# Future Research — Planned Experiments

**Project:** Towards Robust Uzbek Neural Dependency Parsing  
**Author:** Sanatbek Matlatipov  
**Created:** 2026-02-27

---

## Overview

This document contains experiment configurations that were designed but not yet trained due to time constraints for the initial paper submission. The codebase fully supports all of these configurations — they can be run using the same commands and infrastructure described in `README.md`.

The completed experiments (E1 and E2) establish:
- **E1 (FastText baseline):** UPOS 80.26, LAS 62.40 (best on merged data)
- **E2 (TahrirchiBERT + last-subword):** UPOS 85.08, LAS 63.81 (best on merged data)

The future experiments below explore additional axes: alternative BERT models, pooling strategies, and BERT+FastText fusion.

---

## Planned Experiment Matrix

Each configuration below would be trained on **two data settings** (just like E1 and E2):
- **X.1** — UzUDT only (train 451 / dev 45 / test 188 sentences)
- **X.2** — UzUDT + UT merged (train 781 / dev 78 / test 325 sentences)

| Exp | Transformer | Super-token Fusion | Static Pretrain | Description |
|-----|-------------|-------------------|-----------------|-------------|
| E3 | TahrirchiBERT | Mean pooling | None | Does mean pooling outperform last-subword? |
| E4 | BERTbek | Last-subword | None | Is BERTbek better than TahrirchiBERT? |
| E5 | BERTbek | Mean pooling | None | BERTbek with alternative fusion |
| E6 | TahrirchiBERT | Mean pooling | FastText cc.uz.300 | Does combining BERT + FastText help? |
| E7 | BERTbek | Mean pooling | FastText cc.uz.300 | BERTbek + FastText fusion |

This gives **10 additional runs** (5 configurations × 2 data settings).

---

## Research Questions

1. **Which Uzbek BERT is better?** TahrirchiBERT (`tahrirchi/tahrirchi-bert-base`) vs BERTbek (`elmurod1202/bertbek-news-big-cased`) — compare E2 vs E4.
2. **Does super-token fusion strategy matter?** Last-subword selection vs mean pooling — compare E2 vs E3, E4 vs E5.
3. **Does combining BERT + FastText help?** Compare BERT-only (E3, E5) vs BERT+FastText (E6, E7).

---

## Training Commands

### E3.1: TahrirchiBERT + mean pooling — UzUDT

```powershell
python -m stanza.models.tagger --mode train `
  --lang uz --shorthand uz_uzudt `
  --train_file data/pos/uz_uzudt.train.in.conllu `
  --eval_file data/pos/uz_uzudt.dev.in.conllu `
  --bert_model tahrirchi/tahrirchi-bert-base `
  --bert_pooling mean `
  --no_pretrain `
  --save_dir saved_models/pos --save_name uz_uzudt_E3.1_tagger.pt `
  --wandb

python -m stanza.models.parser --mode train `
  --lang uz --shorthand uz_uzudt `
  --train_file data/depparse/uz_uzudt.train.in.conllu `
  --eval_file data/depparse/uz_uzudt.dev.in.conllu `
  --bert_model tahrirchi/tahrirchi-bert-base `
  --bert_pooling mean `
  --no_pretrain `
  --save_dir saved_models/depparse --save_name uz_uzudt_E3.1_parser.pt `
  --wandb
```

### E3.2: TahrirchiBERT + mean pooling — UzUDT+UT merged

```powershell
python -m stanza.models.tagger --mode train `
  --lang uz --shorthand uz_combined `
  --train_file data/pos/merged/uz_combined.train.in.conllu `
  --eval_file data/pos/merged/uz_combined.dev.in.conllu `
  --bert_model tahrirchi/tahrirchi-bert-base `
  --bert_pooling mean `
  --no_pretrain `
  --save_dir saved_models/pos --save_name uz_combined_E3.2_tagger.pt `
  --wandb

python -m stanza.models.parser --mode train `
  --lang uz --shorthand uz_combined `
  --train_file data/depparse/merged/uz_combined.train.in.conllu `
  --eval_file data/depparse/merged/uz_combined.dev.in.conllu `
  --bert_model tahrirchi/tahrirchi-bert-base `
  --bert_pooling mean `
  --no_pretrain `
  --save_dir saved_models/depparse --save_name uz_combined_E3.2_parser.pt `
  --wandb
```

### E4.1: BERTbek + last-subword — UzUDT

```powershell
python -m stanza.models.tagger --mode train `
  --lang uz --shorthand uz_uzudt `
  --train_file data/pos/uz_uzudt.train.in.conllu `
  --eval_file data/pos/uz_uzudt.dev.in.conllu `
  --bert_model elmurod1202/bertbek-news-big-cased `
  --no_pretrain `
  --save_dir saved_models/pos --save_name uz_uzudt_E4.1_tagger.pt `
  --wandb

python -m stanza.models.parser --mode train `
  --lang uz --shorthand uz_uzudt `
  --train_file data/depparse/uz_uzudt.train.in.conllu `
  --eval_file data/depparse/uz_uzudt.dev.in.conllu `
  --bert_model elmurod1202/bertbek-news-big-cased `
  --no_pretrain `
  --save_dir saved_models/depparse --save_name uz_uzudt_E4.1_parser.pt `
  --wandb
```

### E4.2: BERTbek + last-subword — UzUDT+UT merged

```powershell
python -m stanza.models.tagger --mode train `
  --lang uz --shorthand uz_combined `
  --train_file data/pos/merged/uz_combined.train.in.conllu `
  --eval_file data/pos/merged/uz_combined.dev.in.conllu `
  --bert_model elmurod1202/bertbek-news-big-cased `
  --no_pretrain `
  --save_dir saved_models/pos --save_name uz_combined_E4.2_tagger.pt `
  --wandb

python -m stanza.models.parser --mode train `
  --lang uz --shorthand uz_combined `
  --train_file data/depparse/merged/uz_combined.train.in.conllu `
  --eval_file data/depparse/merged/uz_combined.dev.in.conllu `
  --bert_model elmurod1202/bertbek-news-big-cased `
  --no_pretrain `
  --save_dir saved_models/depparse --save_name uz_combined_E4.2_parser.pt `
  --wandb
```

### E5.1: BERTbek + mean pooling — UzUDT

```powershell
python -m stanza.models.tagger --mode train `
  --lang uz --shorthand uz_uzudt `
  --train_file data/pos/uz_uzudt.train.in.conllu `
  --eval_file data/pos/uz_uzudt.dev.in.conllu `
  --bert_model elmurod1202/bertbek-news-big-cased `
  --bert_pooling mean `
  --no_pretrain `
  --save_dir saved_models/pos --save_name uz_uzudt_E5.1_tagger.pt `
  --wandb

python -m stanza.models.parser --mode train `
  --lang uz --shorthand uz_uzudt `
  --train_file data/depparse/uz_uzudt.train.in.conllu `
  --eval_file data/depparse/uz_uzudt.dev.in.conllu `
  --bert_model elmurod1202/bertbek-news-big-cased `
  --bert_pooling mean `
  --no_pretrain `
  --save_dir saved_models/depparse --save_name uz_uzudt_E5.1_parser.pt `
  --wandb
```

### E5.2: BERTbek + mean pooling — UzUDT+UT merged

```powershell
python -m stanza.models.tagger --mode train `
  --lang uz --shorthand uz_combined `
  --train_file data/pos/merged/uz_combined.train.in.conllu `
  --eval_file data/pos/merged/uz_combined.dev.in.conllu `
  --bert_model elmurod1202/bertbek-news-big-cased `
  --bert_pooling mean `
  --no_pretrain `
  --save_dir saved_models/pos --save_name uz_combined_E5.2_tagger.pt `
  --wandb

python -m stanza.models.parser --mode train `
  --lang uz --shorthand uz_combined `
  --train_file data/depparse/merged/uz_combined.train.in.conllu `
  --eval_file data/depparse/merged/uz_combined.dev.in.conllu `
  --bert_model elmurod1202/bertbek-news-big-cased `
  --bert_pooling mean `
  --no_pretrain `
  --save_dir saved_models/depparse --save_name uz_combined_E5.2_parser.pt `
  --wandb
```

### E6.1: TahrirchiBERT + mean pooling + FastText — UzUDT

```powershell
python -m stanza.models.tagger --mode train `
  --lang uz --shorthand uz_uzudt `
  --train_file data/pos/uz_uzudt.train.in.conllu `
  --eval_file data/pos/uz_uzudt.dev.in.conllu `
  --bert_model tahrirchi/tahrirchi-bert-base `
  --bert_pooling mean `
  --wordvec_pretrain_file wordvec/uz/pretrain/fasttext_cc_uz_300.pt `
  --wordvec_file wordvec/uz/fasttext/cc.uz.300.vec `
  --save_dir saved_models/pos --save_name uz_uzudt_E6.1_tagger.pt `
  --wandb

python -m stanza.models.parser --mode train `
  --lang uz --shorthand uz_uzudt `
  --train_file data/depparse/uz_uzudt.train.in.conllu `
  --eval_file data/depparse/uz_uzudt.dev.in.conllu `
  --bert_model tahrirchi/tahrirchi-bert-base `
  --bert_pooling mean `
  --wordvec_pretrain_file wordvec/uz/pretrain/fasttext_cc_uz_300.pt `
  --wordvec_file wordvec/uz/fasttext/cc.uz.300.vec `
  --save_dir saved_models/depparse --save_name uz_uzudt_E6.1_parser.pt `
  --wandb
```

### E6.2: TahrirchiBERT + mean pooling + FastText — UzUDT+UT merged

```powershell
python -m stanza.models.tagger --mode train `
  --lang uz --shorthand uz_combined `
  --train_file data/pos/merged/uz_combined.train.in.conllu `
  --eval_file data/pos/merged/uz_combined.dev.in.conllu `
  --bert_model tahrirchi/tahrirchi-bert-base `
  --bert_pooling mean `
  --wordvec_pretrain_file wordvec/uz/pretrain/fasttext_cc_uz_300.pt `
  --wordvec_file wordvec/uz/fasttext/cc.uz.300.vec `
  --save_dir saved_models/pos --save_name uz_combined_E6.2_tagger.pt `
  --wandb

python -m stanza.models.parser --mode train `
  --lang uz --shorthand uz_combined `
  --train_file data/depparse/merged/uz_combined.train.in.conllu `
  --eval_file data/depparse/merged/uz_combined.dev.in.conllu `
  --bert_model tahrirchi/tahrirchi-bert-base `
  --bert_pooling mean `
  --wordvec_pretrain_file wordvec/uz/pretrain/fasttext_cc_uz_300.pt `
  --wordvec_file wordvec/uz/fasttext/cc.uz.300.vec `
  --save_dir saved_models/depparse --save_name uz_combined_E6.2_parser.pt `
  --wandb
```

### E7.1: BERTbek + mean pooling + FastText — UzUDT

```powershell
python -m stanza.models.tagger --mode train `
  --lang uz --shorthand uz_uzudt `
  --train_file data/pos/uz_uzudt.train.in.conllu `
  --eval_file data/pos/uz_uzudt.dev.in.conllu `
  --bert_model elmurod1202/bertbek-news-big-cased `
  --bert_pooling mean `
  --wordvec_pretrain_file wordvec/uz/pretrain/fasttext_cc_uz_300.pt `
  --wordvec_file wordvec/uz/fasttext/cc.uz.300.vec `
  --save_dir saved_models/pos --save_name uz_uzudt_E7.1_tagger.pt `
  --wandb

python -m stanza.models.parser --mode train `
  --lang uz --shorthand uz_uzudt `
  --train_file data/depparse/uz_uzudt.train.in.conllu `
  --eval_file data/depparse/uz_uzudt.dev.in.conllu `
  --bert_model elmurod1202/bertbek-news-big-cased `
  --bert_pooling mean `
  --wordvec_pretrain_file wordvec/uz/pretrain/fasttext_cc_uz_300.pt `
  --wordvec_file wordvec/uz/fasttext/cc.uz.300.vec `
  --save_dir saved_models/depparse --save_name uz_uzudt_E7.1_parser.pt `
  --wandb
```

### E7.2: BERTbek + mean pooling + FastText — UzUDT+UT merged

```powershell
python -m stanza.models.tagger --mode train `
  --lang uz --shorthand uz_combined `
  --train_file data/pos/merged/uz_combined.train.in.conllu `
  --eval_file data/pos/merged/uz_combined.dev.in.conllu `
  --bert_model elmurod1202/bertbek-news-big-cased `
  --bert_pooling mean `
  --wordvec_pretrain_file wordvec/uz/pretrain/fasttext_cc_uz_300.pt `
  --wordvec_file wordvec/uz/fasttext/cc.uz.300.vec `
  --save_dir saved_models/pos --save_name uz_combined_E7.2_tagger.pt `
  --wandb

python -m stanza.models.parser --mode train `
  --lang uz --shorthand uz_combined `
  --train_file data/depparse/merged/uz_combined.train.in.conllu `
  --eval_file data/depparse/merged/uz_combined.dev.in.conllu `
  --bert_model elmurod1202/bertbek-news-big-cased `
  --bert_pooling mean `
  --wordvec_pretrain_file wordvec/uz/pretrain/fasttext_cc_uz_300.pt `
  --wordvec_file wordvec/uz/fasttext/cc.uz.300.vec `
  --save_dir saved_models/depparse --save_name uz_combined_E7.2_parser.pt `
  --wandb
```

---

## Prerequisites

All code changes required for these experiments are **already implemented** in the current codebase:

- `--bert_model` argument supported in both `tagger.py` and `parser.py`
- `--bert_pooling mean` option available in `bert_embedding.py`
- Combined BERT + FastText (just pass both `--bert_model` and `--wordvec_pretrain_file`)
- wandb logging with `--wandb` flag
- CSV/JSON/PNG training artifacts generated automatically

No additional code changes are needed — just run the commands above.
