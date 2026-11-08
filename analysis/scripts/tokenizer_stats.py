"""
Subword segmentation of Uzbek words by the two encoders' tokenizers (as Stanza calls them:
fast tokenizer, add_prefix_space=True, is_split_into_words=True):
  TahrirchiBERT (byte-level BPE) and BERTbek (WordPiece), and for comparison the multilingual mBERT
  (WordPiece) and XLM-R (SentencePiece unigram).
Suffix analysis on inflected words (FORM = LEMMA + suffix): does the last subword piece lie inside the
suffix chain, and does it contain the whole chain?
"""
import os

import numpy as np
from transformers import AutoTokenizer

from common import RESULTS, UT, UZUDT, normalise, read_conllu, save_json, words

TOKENIZERS = {"TahrirchiBERT": "tahrirchi/tahrirchi-bert-base", "BERTbek": "elmurod1202/bertbek-news-big-cased",
              "mBERT": "bert-base-multilingual-cased", "XLM-R": "xlm-roberta-base"}


def corpus_sentences():
    out = []
    for base, pre in ((UZUDT, "uz_uzudt"), (UT, "uz_ut")):
        for split in ("train", "dev", "test"):
            for s in read_conllu(os.path.join(base, f"{pre}-ud-{split}.conllu")):
                out.append([(r[1], r[2], r[3]) for r in words(s)])
    return out


def main():
    sents = corpus_sentences()
    result = {}
    examples = {}
    for name, hub in TOKENIZERS.items():
        tok = AutoTokenizer.from_pretrained(hub, add_prefix_space=True)
        m_counts, last_in_suffix, suffix_in_last, inflected = [], 0, 0, 0
        ex = []
        for s in sents:
            forms = [w[0] for w in s]
            enc = tok(forms, is_split_into_words=True, return_offsets_mapping=True, add_special_tokens=False)
            word_ids = enc.word_ids()
            offsets = enc["offset_mapping"]
            spans = {}
            for wid, (a, b) in zip(word_ids, offsets):
                if wid is None:
                    continue
                spans.setdefault(wid, []).append((a, b))
            for wid, (form, lemma, upos) in enumerate(s):
                pieces = spans.get(wid, [])
                m_counts.append(len(pieces))
                f, l = normalise(form), normalise(lemma)
                if upos in ("NOUN", "VERB", "ADJ", "PROPN", "PRON") and len(f) > len(l) and f.startswith(l) and l:
                    inflected += 1
                    last_a, last_b = pieces[-1]
                    # offsets are relative to the word in split-into-words mode; with a prefix space they may
                    # include the space: clamp to the word
                    last_a = max(0, min(last_a, len(form)))
                    if last_a >= len(l):
                        last_in_suffix += 1
                    if last_a <= len(l):
                        suffix_in_last += 1
                    if len(ex) < 8 and len(pieces) >= 3:
                        ex.append([form, [form[max(0, a):b] for a, b in pieces]])
        m = np.array(m_counts)
        result[name] = {"words": int(len(m)), "mean_pieces": float(m.mean()), "share_single_piece": float((m == 1).mean()),
                        "share_3plus": float((m >= 3).mean()), "max_pieces": int(m.max()),
                        "inflected_words": inflected,
                        "last_piece_inside_suffix": last_in_suffix / inflected,
                        "last_piece_contains_whole_suffix": suffix_in_last / inflected,
                        "mean_weight_on_last_piece_under_mean_pooling": float(np.mean(1.0 / m))}
        examples[name] = ex
    save_json({"tokenizers": result, "examples": examples}, "tokenizer_stats.json")
    for k, v in result.items():
        print(k, {a: (round(b, 4) if isinstance(b, float) else b) for a, b in v.items()})
    for k, v in examples.items():
        print(k, v[:4])


if __name__ == "__main__":
    main()
