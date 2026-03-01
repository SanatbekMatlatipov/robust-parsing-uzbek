#!/usr/bin/env python3
"""
Compare training curves across multiple experiments.

Reads CSV training logs produced by tagger.py / parser.py and generates
merged line-graph visualizations for the research paper.

Usage:
    # Compare POS tagger experiments
    python scripts/compare_experiments.py --mode pos `
      --logs saved_models/pos/uz_uzudt_E1_tagger_training_log.csv `
            saved_models/pos/uz_uzudt_E4_tagger_training_log.csv `
            saved_models/pos/uz_uzudt_E8_tagger_training_log.csv `
      --labels "E1: FastText" "E4: TahrirchiBERT+mean" "E8: BERTbek+mean" `
      --output_dir saved_models/pos/plots

    # Compare parser experiments
    python scripts/compare_experiments.py --mode depparse `
      --logs saved_models/depparse/uz_uzudt_E1_parser_training_log.csv `
            saved_models/depparse/uz_uzudt_E8_parser_training_log.csv `
      --labels "E1: FastText" "E8: BERTbek+mean" `
      --output_dir saved_models/depparse/plots

    # Generate bar chart from summary JSONs
    python scripts/compare_experiments.py --mode summary `
      --summaries saved_models/pos/*_summary.json `
      --output_dir saved_models/pos/plots
"""

import argparse
import csv
import json
import os
import glob
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def read_csv_log(filepath):
    """Read a training CSV log and return a dict of lists."""
    data = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key, val in row.items():
                if key not in data:
                    data[key] = []
                try:
                    data[key].append(float(val))
                except (ValueError, TypeError):
                    data[key].append(val)
    return data


def plot_pos_comparison(log_files, labels, output_dir):
    """Generate merged comparison plots for POS tagger experiments."""
    os.makedirs(output_dir, exist_ok=True)
    all_data = []
    for f in log_files:
        all_data.append(read_csv_log(f))

    # --- 1. Loss comparison ---
    fig, ax = plt.subplots(figsize=(12, 5))
    for data, label in zip(all_data, labels):
        ax.plot(data['step'], data['train_loss'], '-o', markersize=2, label=label)
    ax.set_xlabel('Step')
    ax.set_ylabel('Avg Training Loss')
    ax.set_title('Training Loss Comparison — POS Tagger')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'comparison_pos_loss.png'), dpi=200)
    plt.close(fig)
    print(f"  Saved: comparison_pos_loss.png")

    # --- 2. UPOS F1 comparison ---
    fig, ax = plt.subplots(figsize=(12, 5))
    for data, label in zip(all_data, labels):
        ax.plot(data['step'], data['upos'], '-o', markersize=2, label=label)
    ax.set_xlabel('Step')
    ax.set_ylabel('UPOS F1 (%)')
    ax.set_title('Dev UPOS F1 Comparison — POS Tagger')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'comparison_pos_upos.png'), dpi=200)
    plt.close(fig)
    print(f"  Saved: comparison_pos_upos.png")

    # --- 3. XPOS F1 comparison ---
    fig, ax = plt.subplots(figsize=(12, 5))
    for data, label in zip(all_data, labels):
        ax.plot(data['step'], data['xpos'], '-s', markersize=2, label=label)
    ax.set_xlabel('Step')
    ax.set_ylabel('XPOS F1 (%)')
    ax.set_title('Dev XPOS F1 Comparison — POS Tagger')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'comparison_pos_xpos.png'), dpi=200)
    plt.close(fig)
    print(f"  Saved: comparison_pos_xpos.png")

    # --- 4. UFeats F1 comparison ---
    fig, ax = plt.subplots(figsize=(12, 5))
    for data, label in zip(all_data, labels):
        ax.plot(data['step'], data['ufeats'], '-^', markersize=2, label=label)
    ax.set_xlabel('Step')
    ax.set_ylabel('UFeats F1 (%)')
    ax.set_title('Dev UFeats F1 Comparison — POS Tagger')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'comparison_pos_ufeats.png'), dpi=200)
    plt.close(fig)
    print(f"  Saved: comparison_pos_ufeats.png")

    # --- 5. All metrics in one figure (2x2 subplots) ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for data, label in zip(all_data, labels):
        axes[0, 0].plot(data['step'], data['train_loss'], '-o', markersize=2, label=label)
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)

    for data, label in zip(all_data, labels):
        axes[0, 1].plot(data['step'], data['upos'], '-o', markersize=2, label=label)
    axes[0, 1].set_title('UPOS F1')
    axes[0, 1].set_ylabel('F1 (%)')
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)

    for data, label in zip(all_data, labels):
        axes[1, 0].plot(data['step'], data['xpos'], '-s', markersize=2, label=label)
    axes[1, 0].set_title('XPOS F1')
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].set_ylabel('F1 (%)')
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)

    for data, label in zip(all_data, labels):
        axes[1, 1].plot(data['step'], data['ufeats'], '-^', markersize=2, label=label)
    axes[1, 1].set_title('UFeats F1')
    axes[1, 1].set_xlabel('Step')
    axes[1, 1].set_ylabel('F1 (%)')
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle('POS Tagger — Experiment Comparison', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'comparison_pos_all.png'), dpi=200)
    plt.close(fig)
    print(f"  Saved: comparison_pos_all.png")


def plot_depparse_comparison(log_files, labels, output_dir):
    """Generate merged comparison plots for dependency parser experiments."""
    os.makedirs(output_dir, exist_ok=True)
    all_data = []
    for f in log_files:
        all_data.append(read_csv_log(f))

    # --- 1. Loss comparison ---
    fig, ax = plt.subplots(figsize=(12, 5))
    for data, label in zip(all_data, labels):
        ax.plot(data['step'], data['train_loss'], '-o', markersize=2, label=label)
    ax.set_xlabel('Step')
    ax.set_ylabel('Avg Training Loss')
    ax.set_title('Training Loss Comparison — Dependency Parser')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'comparison_depparse_loss.png'), dpi=200)
    plt.close(fig)
    print(f"  Saved: comparison_depparse_loss.png")

    # --- 2. UAS comparison ---
    fig, ax = plt.subplots(figsize=(12, 5))
    for data, label in zip(all_data, labels):
        ax.plot(data['step'], data['uas'], '-o', markersize=2, label=label)
    ax.set_xlabel('Step')
    ax.set_ylabel('UAS (%)')
    ax.set_title('Dev UAS Comparison — Dependency Parser')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'comparison_depparse_uas.png'), dpi=200)
    plt.close(fig)
    print(f"  Saved: comparison_depparse_uas.png")

    # --- 3. LAS comparison ---
    fig, ax = plt.subplots(figsize=(12, 5))
    for data, label in zip(all_data, labels):
        ax.plot(data['step'], data['las'], '-s', markersize=2, label=label)
    ax.set_xlabel('Step')
    ax.set_ylabel('LAS (%)')
    ax.set_title('Dev LAS Comparison — Dependency Parser')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'comparison_depparse_las.png'), dpi=200)
    plt.close(fig)
    print(f"  Saved: comparison_depparse_las.png")

    # --- 4. All metrics in one figure (2x2 subplots) ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for data, label in zip(all_data, labels):
        axes[0, 0].plot(data['step'], data['train_loss'], '-o', markersize=2, label=label)
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)

    for data, label in zip(all_data, labels):
        axes[0, 1].plot(data['step'], data['uas'], '-o', markersize=2, label=label)
    axes[0, 1].set_title('UAS')
    axes[0, 1].set_ylabel('Score (%)')
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)

    for data, label in zip(all_data, labels):
        axes[1, 0].plot(data['step'], data['las'], '-s', markersize=2, label=label)
    axes[1, 0].set_title('LAS')
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].set_ylabel('Score (%)')
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)

    # MLAS if available
    has_mlas = all(('mlas' in d) for d in all_data)
    if has_mlas:
        for data, label in zip(all_data, labels):
            axes[1, 1].plot(data['step'], data['mlas'], '-^', markersize=2, label=label)
        axes[1, 1].set_title('MLAS')
    else:
        for data, label in zip(all_data, labels):
            if 'blex' in data:
                axes[1, 1].plot(data['step'], data['blex'], '-^', markersize=2, label=label)
        axes[1, 1].set_title('BLEX')
    axes[1, 1].set_xlabel('Step')
    axes[1, 1].set_ylabel('Score (%)')
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle('Dependency Parser — Experiment Comparison', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'comparison_depparse_all.png'), dpi=200)
    plt.close(fig)
    print(f"  Saved: comparison_depparse_all.png")


def plot_summary_bar_chart(summary_files, output_dir):
    """Generate bar charts from summary JSON files."""
    os.makedirs(output_dir, exist_ok=True)

    summaries = []
    for sf in summary_files:
        with open(sf, 'r', encoding='utf-8') as f:
            s = json.load(f)
            s['_file'] = os.path.basename(sf)
            summaries.append(s)

    if not summaries:
        print("No summary files found.")
        return

    # Detect whether POS or depparse
    is_pos = 'final_upos' in summaries[0]

    names = [s.get('experiment', s['_file']).replace('.pt', '') for s in summaries]
    x = np.arange(len(names))
    width = 0.3

    if is_pos:
        upos = [s.get('final_upos', 0) for s in summaries]
        xpos = [s.get('final_xpos', 0) for s in summaries]
        ufeats = [s.get('final_ufeats', 0) for s in summaries]

        fig, ax = plt.subplots(figsize=(max(10, len(names) * 1.5), 6))
        ax.bar(x - width, upos, width, label='UPOS', color='#2196F3')
        ax.bar(x, xpos, width, label='XPOS', color='#FF9800')
        ax.bar(x + width, ufeats, width, label='UFeats', color='#4CAF50')
        ax.set_ylabel('F1 Score (%)')
        ax.set_title('POS Tagger — Final Dev Scores by Experiment')
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=30, ha='right', fontsize=9)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for i, (u, xp, uf) in enumerate(zip(upos, xpos, ufeats)):
            ax.text(i - width, u + 0.3, f'{u:.1f}', ha='center', va='bottom', fontsize=7)
            ax.text(i, xp + 0.3, f'{xp:.1f}', ha='center', va='bottom', fontsize=7)
            ax.text(i + width, uf + 0.3, f'{uf:.1f}', ha='center', va='bottom', fontsize=7)

        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, 'summary_pos_bar.png'), dpi=200)
        plt.close(fig)
        print(f"  Saved: summary_pos_bar.png")

    else:
        uas = [s.get('final_uas', 0) for s in summaries]
        las = [s.get('final_las', 0) for s in summaries]

        fig, ax = plt.subplots(figsize=(max(10, len(names) * 1.5), 6))
        ax.bar(x - width/2, uas, width, label='UAS', color='#2196F3')
        ax.bar(x + width/2, las, width, label='LAS', color='#FF9800')
        ax.set_ylabel('Score (%)')
        ax.set_title('Dependency Parser — Final Dev Scores by Experiment')
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=30, ha='right', fontsize=9)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        for i, (u, l) in enumerate(zip(uas, las)):
            ax.text(i - width/2, u + 0.3, f'{u:.1f}', ha='center', va='bottom', fontsize=7)
            ax.text(i + width/2, l + 0.3, f'{l:.1f}', ha='center', va='bottom', fontsize=7)

        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, 'summary_depparse_bar.png'), dpi=200)
        plt.close(fig)
        print(f"  Saved: summary_depparse_bar.png")


def main():
    parser = argparse.ArgumentParser(description='Compare training curves across experiments')
    parser.add_argument('--mode', choices=['pos', 'depparse', 'summary'], required=True,
                        help='Type of comparison')
    parser.add_argument('--logs', nargs='+',
                        help='CSV training log files to compare (for pos/depparse mode)')
    parser.add_argument('--labels', nargs='+',
                        help='Labels for each log file (must match --logs count)')
    parser.add_argument('--summaries', nargs='+',
                        help='Summary JSON files (for summary mode). Supports glob patterns.')
    parser.add_argument('--output_dir', type=str, default='saved_models/plots',
                        help='Directory to save comparison plots')
    args = parser.parse_args()

    if args.mode in ('pos', 'depparse'):
        if not args.logs:
            print("Error: --logs is required for pos/depparse mode")
            sys.exit(1)
        if args.labels and len(args.labels) != len(args.logs):
            print(f"Error: --labels count ({len(args.labels)}) must match --logs count ({len(args.logs)})")
            sys.exit(1)
        labels = args.labels or [os.path.basename(f).replace('_training_log.csv', '') for f in args.logs]

        print(f"Comparing {len(args.logs)} experiments ({args.mode} mode)...")
        if args.mode == 'pos':
            plot_pos_comparison(args.logs, labels, args.output_dir)
        else:
            plot_depparse_comparison(args.logs, labels, args.output_dir)

    elif args.mode == 'summary':
        # Expand glob patterns
        files = []
        for pattern in (args.summaries or []):
            expanded = glob.glob(pattern)
            files.extend(expanded)
        if not files:
            print("Error: no summary files found. Use --summaries with JSON file paths or glob patterns.")
            sys.exit(1)
        print(f"Generating bar chart from {len(files)} summary files...")
        plot_summary_bar_chart(files, args.output_dir)

    print(f"\nAll plots saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
