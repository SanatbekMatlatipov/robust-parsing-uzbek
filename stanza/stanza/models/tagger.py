"""
Entry point for training and evaluating a POS/morphological features tagger.

This tagger uses highway BiLSTM layers with character and word-level representations, and biaffine classifiers
to produce consistent POS and UFeats predictions.
For details please refer to paper: https://nlp.stanford.edu/pubs/qi2018universal.pdf.
"""

import sys
import os
import shutil
import time
import json
import csv
from datetime import datetime
import argparse
import logging
import numpy as np
import random
import torch
from torch import nn, optim

import stanza.models.pos.data as data
from stanza.models.pos.data import DataLoader
from stanza.models.pos.trainer import Trainer
from stanza.models.pos import scorer
from stanza.models.common import utils
from stanza.models.common import pretrain
from stanza.models.common.data import augment_punct
from stanza.models.common.doc import *
from stanza.utils.conll import CoNLL
from stanza.models import _training_logging

logger = logging.getLogger('stanza')

# Wandb is optional — if not installed, training proceeds without it
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/pos', help='Root dir for saving models.')
    parser.add_argument('--wordvec_dir', type=str, default='extern_data/wordvec', help='Directory of word vectors.')
    parser.add_argument('--wordvec_file', type=str, default=None, help='Word vectors filename.')
    parser.add_argument('--wordvec_pretrain_file', type=str, default=None, help='Exact name of the pretrain file to read')
    parser.add_argument('--train_file', type=str, default=None, help='Input file for data loader.')
    parser.add_argument('--eval_file', type=str, default=None, help='Input file for data loader.')
    parser.add_argument('--output_file', type=str, default=None, help='Output CoNLL-U file.')
    parser.add_argument('--gold_file', type=str, default=None, help='Output CoNLL-U file.')

    parser.add_argument('--mode', default='train', choices=['train', 'predict'])
    parser.add_argument('--lang', type=str, help='Language')
    parser.add_argument('--shorthand', type=str, help="Treebank shorthand")

    parser.add_argument('--hidden_dim', type=int, default=200)
    parser.add_argument('--char_hidden_dim', type=int, default=400)
    parser.add_argument('--deep_biaff_hidden_dim', type=int, default=400)
    parser.add_argument('--composite_deep_biaff_hidden_dim', type=int, default=100)
    parser.add_argument('--word_emb_dim', type=int, default=75)
    parser.add_argument('--char_emb_dim', type=int, default=100)
    parser.add_argument('--tag_emb_dim', type=int, default=50)
    parser.add_argument('--transformed_dim', type=int, default=125)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--char_num_layers', type=int, default=1)
    parser.add_argument('--pretrain_max_vocab', type=int, default=250000)
    parser.add_argument('--word_dropout', type=float, default=0.33)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--rec_dropout', type=float, default=0, help="Recurrent dropout")
    parser.add_argument('--char_rec_dropout', type=float, default=0, help="Recurrent dropout")
    parser.add_argument('--no_char', dest='char', action='store_false', help="Turn off character model.")
    parser.add_argument('--no_pretrain', dest='pretrain', action='store_false', help="Turn off pretrained embeddings.")
    parser.add_argument('--bert_model', type=str, default=None, help='BERT model name for contextual embeddings (e.g., tahrirchi/tahrirchi-bert-base)')
    parser.add_argument('--bert_pooling', type=str, default='last', choices=['last', 'mean'], help='BERT subword pooling strategy: last (default) or mean')
    parser.add_argument('--share_hid', action='store_true', help="Share hidden representations for UPOS, XPOS and UFeats.")
    parser.set_defaults(share_hid=False)

    parser.add_argument('--sample_train', type=float, default=1.0, help='Subsample training data.')
    parser.add_argument('--optim', type=str, default='adam', help='sgd, adagrad, adam or adamax.')
    parser.add_argument('--lr', type=float, default=3e-3, help='Learning rate')
    parser.add_argument('--beta2', type=float, default=0.95)

    parser.add_argument('--max_steps', type=int, default=50000)
    parser.add_argument('--eval_interval', type=int, default=100)
    parser.add_argument('--fix_eval_interval', dest='adapt_eval_interval', action='store_false', \
            help="Use fixed evaluation interval for all treebanks, otherwise by default the interval will be increased for larger treebanks.")
    parser.add_argument('--max_steps_before_stop', type=int, default=3000, help='Changes learning method or early terminates after this many steps if the dev scores are not improving')
    parser.add_argument('--batch_size', type=int, default=5000)
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Gradient clipping.')
    parser.add_argument('--log_step', type=int, default=20, help='Print log every k steps.')
    parser.add_argument('--save_dir', type=str, default='saved_models/pos', help='Root dir for saving models.')
    parser.add_argument('--save_name', type=str, default=None, help="File name to save the model")

    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
    parser.add_argument('--cpu', action='store_true', help='Ignore CUDA.')

    parser.add_argument('--augment_nopunct', type=float, default=None, help='Augment the training data by copying this fraction of punct-ending sentences as non-punct.  Default of None will aim for roughly 10%%')

    parser.add_argument('--wandb', action='store_true', help='Enable Weights & Biases logging.')
    parser.add_argument('--wandb_project', type=str, default='uzbek-pos-tagger', help='W&B project name.')

    args = parser.parse_args(args=args)
    return args

def main(args=None):
    args = parse_args(args=args)

    if args.cpu:
        args.cuda = False
    utils.set_random_seed(args.seed, args.cuda)

    args = vars(args)
    logger.info("Running tagger in {} mode".format(args['mode']))

    # Log GPU/CUDA status explicitly
    if torch.cuda.is_available():
        logger.info("CUDA is available. GPU count: {}. Using GPU: {}".format(
            torch.cuda.device_count(), args['cuda']))
        for i in range(torch.cuda.device_count()):
            logger.info("  GPU {}: {}".format(i, torch.cuda.get_device_name(i)))
    else:
        logger.info("CUDA is NOT available. Training will use CPU.")
    sys.stdout.flush()

    if args['mode'] == 'train':
        train(args)
    else:
        evaluate(args)

def model_file_name(args):
    if args['save_name'] is not None:
        save_name = args['save_name']
    else:
        save_name = args['shorthand'] + "_tagger.pt"

    return os.path.join(args['save_dir'], save_name)

def load_pretrain(args):
    pt = None
    if args['pretrain']:
        pretrain_file = pretrain.find_pretrain_file(args['wordvec_pretrain_file'], args['save_dir'], args['shorthand'], args['lang'])
        if os.path.exists(pretrain_file):
            vec_file = None
        else:
            vec_file = args['wordvec_file'] if args['wordvec_file'] else utils.get_wordvec_file(args['wordvec_dir'], args['shorthand'])
        pt = pretrain.Pretrain(pretrain_file, vec_file, args['pretrain_max_vocab'])
    return pt

def train(args):
    model_file = model_file_name(args)
    utils.ensure_dir(os.path.split(model_file)[0])

    # --- Logging setup: CSV log file alongside saved model ---
    log_base = os.path.splitext(model_file)[0]
    csv_log_file = log_base + '_training_log.csv'
    plots_dir = os.path.join(args['save_dir'], 'plots')
    utils.ensure_dir(plots_dir)

    # --- W&B init (optional) ---
    use_wandb = args.get('wandb', False) and HAS_WANDB
    if args.get('wandb', False) and not HAS_WANDB:
        logger.warning("--wandb flag set but wandb is not installed. Skipping W&B logging.")
    if use_wandb:
        run_name = args.get('save_name', args['shorthand'] + '_tagger')
        if run_name.endswith('.pt'):
            run_name = run_name[:-3]
        wandb.init(
            project=args.get('wandb_project', 'uzbek-pos-tagger'),
            name=run_name,
            config={k: v for k, v in args.items() if isinstance(v, (str, int, float, bool, type(None)))},
            tags=['pos', args.get('shorthand', ''), args.get('lang', '')],
        )
        logger.info("W&B run initialized: {}".format(wandb.run.url))

    # load pretrained vectors if needed
    pretrain = load_pretrain(args)

    # load data
    logger.info("Loading data with batch size {}...".format(args['batch_size']))
    # train_data is now a list of sentences, where each sentence is a
    # list of words, in which each word is a dict of conll attributes
    train_data, _ = CoNLL.conll2dict(input_file=args['train_file'])
    # possibly augment the training data with some amount of fake data
    # based on the options chosen
    logger.info("Original data size: {}".format(len(train_data)))
    train_data.extend(augment_punct(train_data, args['augment_nopunct'],
                                    keep_original_sentences=False))
    logger.info("Augmented data size: {}".format(len(train_data)))
    train_doc = Document(train_data)
    train_batch = DataLoader(train_doc, args['batch_size'], args, pretrain, evaluation=False)
    vocab = train_batch.vocab
    dev_doc = CoNLL.conll2doc(input_file=args['eval_file'])
    dev_batch = DataLoader(dev_doc, args['batch_size'], args, pretrain, vocab=vocab, evaluation=True, sort_during_eval=True)

    # pred and gold path
    system_pred_file = args['output_file']
    if system_pred_file is None:
        system_pred_file = '{}.dev.pred.conllu'.format(log_base)
    gold_file = args['gold_file']
    if gold_file is None:
        gold_file = args['eval_file']

    # skip training if the language does not have training or dev data
    if len(train_batch) == 0 or len(dev_batch) == 0:
        logger.info("Skip training because no data available...")
        return

    logger.info("Training tagger...")
    if args['cuda']:
        logger.info("Model will be placed on GPU (cuda:0)")
    else:
        logger.info("Model will run on CPU (no CUDA)")
    sys.stdout.flush()
    trainer = Trainer(args=args, vocab=vocab, pretrain=pretrain, use_cuda=args['cuda'])

    global_step = 0
    max_steps = args['max_steps']
    dev_score_history = []
    best_dev_preds = []
    current_lr = args['lr']
    global_start_time = time.time()
    format_str = 'Finished STEP {}/{}, loss = {:.6f} ({:.3f} sec/batch), lr: {:.6f}'

    if args['adapt_eval_interval']:
        args['eval_interval'] = utils.get_adaptive_eval_interval(dev_batch.num_examples, 2000, args['eval_interval'])
        logger.info("Evaluating the model every {} steps...".format(args['eval_interval']))

    # --- Metrics tracking lists ---
    metrics_log = []  # list of dicts for CSV
    step_losses = []  # (step, loss) for per-step loss curve
    eval_steps = []
    eval_train_losses = []
    eval_dev_scores = []
    eval_upos_scores = []
    eval_xpos_scores = []
    eval_ufeats_scores = []
    lr_history = []  # (step, lr)

    # Write CSV header
    with open(csv_log_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['step', 'train_loss', 'dev_score', 'upos', 'xpos', 'ufeats', 'alltags', 'lr', 'elapsed_sec'])

    using_amsgrad = False
    last_best_step = 0
    # start training
    train_loss = 0
    while True:
        do_break = False
        for i, batch in enumerate(train_batch):
            start_time = time.time()
            global_step += 1
            loss = trainer.update(batch, eval=False) # update step
            train_loss += loss
            step_losses.append((global_step, loss))
            lr_history.append((global_step, current_lr))

            if global_step % args['log_step'] == 0:
                duration = time.time() - start_time
                logger.info(format_str.format(global_step, max_steps, loss, duration, current_lr))
                if use_wandb:
                    wandb.log({'train/step_loss': loss, 'train/lr': current_lr, 'step': global_step})

            if global_step % args['eval_interval'] == 0:
                # eval on dev
                logger.info("Evaluating on dev set...")
                dev_preds = []
                for batch in dev_batch:
                    preds = trainer.predict(batch)
                    dev_preds += preds
                dev_preds = utils.unsort(dev_preds, dev_batch.data_orig_idx)
                dev_batch.doc.set([UPOS, XPOS, FEATS], [y for x in dev_preds for y in x])
                CoNLL.write_doc2conll(dev_batch.doc, system_pred_file)
                _, _, dev_score = scorer.score(system_pred_file, gold_file)

                # Get detailed scores
                from stanza.models.common.utils import ud_scores
                evaluation = ud_scores(gold_file, system_pred_file)
                upos_f1 = evaluation['UPOS'].f1 * 100
                xpos_f1 = evaluation['XPOS'].f1 * 100
                ufeats_f1 = evaluation['UFeats'].f1 * 100
                alltags_f1 = evaluation['AllTags'].f1 * 100

                train_loss = train_loss / args['eval_interval'] # avg loss per batch
                elapsed = time.time() - global_start_time
                logger.info("step {}: train_loss = {:.6f}, dev_score = {:.4f}".format(global_step, train_loss, dev_score))

                # --- Record metrics ---
                eval_steps.append(global_step)
                eval_train_losses.append(train_loss)
                eval_dev_scores.append(dev_score * 100)
                eval_upos_scores.append(upos_f1)
                eval_xpos_scores.append(xpos_f1)
                eval_ufeats_scores.append(ufeats_f1)

                # Append to CSV
                with open(csv_log_file, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([global_step, f'{train_loss:.6f}', f'{dev_score*100:.2f}',
                                     f'{upos_f1:.2f}', f'{xpos_f1:.2f}', f'{ufeats_f1:.2f}',
                                     f'{alltags_f1:.2f}', f'{current_lr:.6f}', f'{elapsed:.1f}'])

                # W&B logging
                if use_wandb:
                    wandb.log({
                        'eval/dev_score': dev_score * 100,
                        'eval/upos_f1': upos_f1,
                        'eval/xpos_f1': xpos_f1,
                        'eval/ufeats_f1': ufeats_f1,
                        'eval/alltags_f1': alltags_f1,
                        'train/avg_loss': train_loss,
                        'train/lr': current_lr,
                        'step': global_step,
                    })

                train_loss = 0

                # save best model
                if len(dev_score_history) == 0 or dev_score > max(dev_score_history):
                    last_best_step = global_step
                    trainer.save(model_file)
                    logger.info("new best model saved.")
                    best_dev_preds = dev_preds

                dev_score_history += [dev_score]

            if global_step - last_best_step >= args['max_steps_before_stop']:
                if not using_amsgrad:
                    logger.info("Switching to AMSGrad")
                    last_best_step = global_step
                    using_amsgrad = True
                    trainer.optimizer = optim.Adam(trainer.model.parameters(), amsgrad=True, lr=args['lr'], betas=(.9, args['beta2']), eps=1e-6)
                else:
                    logger.info("Early termination: have not improved in {} steps".format(args['max_steps_before_stop']))
                    do_break = True
                    break

            if global_step >= args['max_steps']:
                do_break = True
                break

        if do_break: break

        train_batch.reshuffle()

    logger.info("Training ended with {} steps.".format(global_step))

    if len(dev_score_history) > 0:
        best_f, best_eval = max(dev_score_history)*100, np.argmax(dev_score_history)+1
        logger.info("Best dev F1 = {:.2f}, at iteration = {}".format(best_f, best_eval * args['eval_interval']))
    else:
        logger.info("Dev set never evaluated.  Saving final model.")
        trainer.save(model_file)

    # --- Generate training plots ---
    if HAS_MATPLOTLIB and len(eval_steps) > 0:
        run_label = os.path.splitext(args.get('save_name', args['shorthand']))[0]
        _save_tagger_plots(plots_dir, run_label, eval_steps, eval_train_losses,
                           eval_dev_scores, eval_upos_scores, eval_xpos_scores,
                           eval_ufeats_scores, step_losses, lr_history)
        logger.info("Training plots saved to: {}".format(plots_dir))

    # Save final metrics summary as JSON
    summary = {
        'experiment': args.get('save_name', args['shorthand']),
        'total_steps': global_step,
        'best_dev_score': float(max(dev_score_history) * 100) if dev_score_history else 0,
        'best_step': int((np.argmax(dev_score_history) + 1) * args['eval_interval']) if dev_score_history else 0,
        'final_upos': float(eval_upos_scores[-1]) if eval_upos_scores else 0,
        'final_xpos': float(eval_xpos_scores[-1]) if eval_xpos_scores else 0,
        'final_ufeats': float(eval_ufeats_scores[-1]) if eval_ufeats_scores else 0,
        'training_time_sec': time.time() - global_start_time,
        'csv_log': csv_log_file,
    }
    summary_file = log_base + '_summary.json'
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    logger.info("Training summary saved to: {}".format(summary_file))

    if use_wandb:
        wandb.summary['best_dev_score'] = summary['best_dev_score']
        wandb.summary['best_step'] = summary['best_step']
        wandb.finish()


def _save_tagger_plots(plots_dir, run_label, eval_steps, train_losses,
                       dev_scores, upos_scores, xpos_scores, ufeats_scores,
                       step_losses, lr_history):
    """Generate and save training curves for POS tagger."""
    # 1. Loss curve
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(eval_steps, train_losses, 'b-o', markersize=3, label='Avg Train Loss')
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title(f'Training Loss — {run_label}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, f'{run_label}_loss.png'), dpi=150)
    plt.close(fig)

    # 2. Dev accuracy curves (UPOS, XPOS, UFeats)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(eval_steps, upos_scores, 'g-o', markersize=3, label='UPOS F1')
    ax.plot(eval_steps, xpos_scores, 'r-s', markersize=3, label='XPOS F1')
    ax.plot(eval_steps, ufeats_scores, 'm-^', markersize=3, label='UFeats F1')
    ax.set_xlabel('Step')
    ax.set_ylabel('F1 Score (%%)')
    ax.set_title(f'Dev Accuracy — {run_label}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, f'{run_label}_accuracy.png'), dpi=150)
    plt.close(fig)

    # 3. Learning rate schedule
    if lr_history:
        lr_steps, lr_vals = zip(*lr_history)
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(lr_steps, lr_vals, 'k-', linewidth=0.8)
        ax.set_xlabel('Step')
        ax.set_ylabel('Learning Rate')
        ax.set_title(f'Learning Rate — {run_label}')
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, f'{run_label}_lr.png'), dpi=150)
        plt.close(fig)

    # 4. Combined: Loss + Dev Score on dual axis
    fig, ax1 = plt.subplots(figsize=(10, 5))
    color_loss = '#1f77b4'
    color_score = '#2ca02c'
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Avg Train Loss', color=color_loss)
    ax1.plot(eval_steps, train_losses, color=color_loss, linestyle='-', marker='o', markersize=3, label='Train Loss')
    ax1.tick_params(axis='y', labelcolor=color_loss)
    ax2 = ax1.twinx()
    ax2.set_ylabel('Dev Score (%%)', color=color_score)
    ax2.plot(eval_steps, dev_scores, color=color_score, linestyle='-', marker='s', markersize=3, label='Dev Score')
    ax2.tick_params(axis='y', labelcolor=color_score)
    ax1.set_title(f'Training Overview — {run_label}')
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, f'{run_label}_overview.png'), dpi=150)
    plt.close(fig)


def evaluate(args):
    # file paths
    system_pred_file = args['output_file']
    gold_file = args['gold_file']
    model_file = model_file_name(args)

    pretrain = load_pretrain(args)

    # load model
    logger.info("Loading model from: {}".format(model_file))
    use_cuda = args['cuda'] and not args['cpu']
    trainer = Trainer(pretrain=pretrain, model_file=model_file, use_cuda=use_cuda)
    loaded_args, vocab = trainer.args, trainer.vocab

    # load config
    for k in args:
        if k.endswith('_dir') or k.endswith('_file') or k in ['shorthand'] or k == 'mode':
            loaded_args[k] = args[k]

    # load data
    logger.info("Loading data with batch size {}...".format(args['batch_size']))
    doc = CoNLL.conll2doc(input_file=args['eval_file'])
    batch = DataLoader(doc, args['batch_size'], loaded_args, pretrain, vocab=vocab, evaluation=True, sort_during_eval=True)
    if len(batch) > 0:
        logger.info("Start evaluation...")
        preds = []
        for i, b in enumerate(batch):
            preds += trainer.predict(b)
    else:
        # skip eval if dev data does not exist
        preds = []
    preds = utils.unsort(preds, batch.data_orig_idx)

    # write to file and score
    batch.doc.set([UPOS, XPOS, FEATS], [y for x in preds for y in x])
    CoNLL.write_doc2conll(batch.doc, system_pred_file)

    if gold_file is not None:
        _, _, score = scorer.score(system_pred_file, gold_file)

        logger.info("Tagger score:")
        logger.info("{} {:.2f}".format(args['shorthand'], score*100))

if __name__ == '__main__':
    main()
