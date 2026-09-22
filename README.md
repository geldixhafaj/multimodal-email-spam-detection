# Hybrid Transformer Architecture for Context-Aware Spam Email Classification
 
Code and data supporting the paper:
 
> Xhafaj, G.; Chaudhry, U.B.; Jahankhani, H. **Hybrid Transformer Architecture for Context-Aware Spam Email Classification.
 
This repository is the three-notebook Google Colab pipeline that builds the MEPC corpus, trains and evaluates all nine models under both protocols, and produces the explainability analysis reported in the paper. It is a Colab/Google-Drive workflow, not a local `src/`-and-`configs/` package — see **How to Run** below before trying to execute anything locally.
 
---
 
## Repository Contents
 
```text
Multi-source Email Phishing Corpus.csv     # raw MEPC corpus (see Data, below)
Thesis_Phase1_EDA_Preprocessing.ipynb      # corpus cleaning, leakage audit, feature engineering, splits
Thesis_Phase2_Modeling.ipynb               # model training + evaluation under every protocol
Thesis_Phase3_Explainability.ipynb         # SHAP / LIME, ablation, multi-seed variance, final comparison
```
 
The notebooks must be run **in this order** — each reads the artefacts the previous one writes to Google Drive (`My Drive/Thesis/`). None of them touches a local filesystem or a `src/` package; every function, config value, and helper lives inside the notebook that uses it.
 
---
 
## Pipeline Overview
 
### Phase 1 — EDA & Preprocessing (`Thesis_Phase1_EDA_Preprocessing.ipynb`)
 
Runtime: CPU, ~25–35 min (High-RAM recommended). No GPU required.
 
| Part | Content |
|---|---|
| A | Environment, corpus ingestion, schema documentation, missing-data structure |
| B | Data quality audit, cleaning, campaign-aware deduplication |
| C | Corpus characterisation — class balance, source/domain leakage, language, temporal drift |
| D | Model input field construction, lexical (log-odds token) analysis, social-engineering cue analysis |
| E | Structural feature engineering — 13 core + 8 extended features, URL forensics |
| F | Statistical validation — Mann–Whitney effect sizes, mutual information, outliers, PCA |
| G | Experimental design — PSI distribution-shift analysis, **Split A** (in-distribution) and **Split B** (cross-source), plus rotating leave-one-source-out folds **C1–C3**, integrity checks |
| H | Leakage stress tests (metadata-only classifiers) and reference baselines — the generalisation gap |
| I | Feature scaling, class weights, artefact persistence, thesis table export |
 
Writes to `My Drive/Thesis/preprocessed/`: the cleaned corpus (`mepc_clean.parquet`), per-split structural feature arrays (`.npy`), fitted scalers, and a `config.json` contract (seed, feature list, split ratios, class weights, leakage upper bound, etc.) that Phase 2 and Phase 3 both load. Also writes 10 thesis tables (CSV + Markdown) and the corpus/leakage/feature figures to `My Drive/Thesis/tables/` and `.../figures/`.
 
### Phase 2 — Modelling (`Thesis_Phase2_Modeling.ipynb`)
 
Runtime: **T4 GPU** (`Runtime → Change runtime type → T4 GPU`, then `Runtime → Run all`). Every heavy step is cached to Drive, so a disconnected Colab session can simply re-run: finished encoders, embeddings, and baselines are reloaded, and an interrupted fine-tune resumes from its last completed epoch.
 
| Part | Content |
|---|---|
| A | Environment, data, all split designs (A, B, C1–C3), integrity audit, source–class confound audit |
| B | Tokenisation, per-language tokenizer coverage, data loaders |
| C | DistilBERT fine-tuning — one encoder per split design, shared trainer, adaptive epoch budget until early stopping fires |
| D | DistilBERT evaluation on every protocol |
| E | Embedding extraction (one encoder per design, cached and fingerprinted) |
| F | MLP heads — embeddings-only (ablation reference) and embeddings + structural (fusion) |
| G | Classical (RF, NB, SVM, LR) and CNN baselines, refitted identically on every protocol |
| H | Unified comparison, generalisation gap, per-language evaluability |
| I | Statistical robustness — full-seed tables, grouped cross-validation, bootstrap confidence intervals, McNemar tests |
| J | Operating-point selection (validation data only) |
| K | Persistence, hyperparameter table, thesis tables, summary |
 
### Phase 3 — Explainability (`Thesis_Phase3_Explainability.ipynb`)
 
Prerequisite: Phase 1 and Phase 2 artefacts in `preprocessed/` and `models/`; this notebook does not touch the raw corpus.
 
| Part | Content |
|---|---|
| A | Environment, artefact loading (installs `shap` and `lime`) |
| B | Naive Bayes, SVM, CNN baselines added for direct comparability |
| C | SHAP explainability — global importance, beeswarm, and two local instances (a correctly classified phishing email and a misclassified one) computed from the actual fitted fusion model on real held-out data |
| D | LIME explainability on the same two instances, for cross-validation against SHAP |
| E | Retraining-based ablation across four progressive structural-feature configurations |
| F | Multi-seed variance for classical baselines, CNN, and the fusion MLP |
| G | Final unified comparison table and literature (prior-work) comparison |
| H | Persistence and reporting |
 
---
 
## Data
 
**File:** `Multi-source Email Phishing Corpus.csv` (raw MEPC corpus — place in `My Drive/Thesis/` before running Phase 1).
 
| Property | Value |
|---|---|
| Raw rows | 101,229 |
| Columns | 21 — `sender`, `sender_domain`, `receiver`, `receiver_domain`, `date`, `subject`, `content_types`, `body`, `urls`, `url_count`, `url_length_max`, `url_length_avg`, `url_subdom_max`, `url_subdom_avg`, `attachment_count`, `has_attachments`, `attachment_types`, `language`, `source`, `label`, `origin` |
| Sources (`source`) | `trec5`, `trec6`, `trec7` (TREC 2005–2007 spam tracks), `phishing_pot` |
| Class balance (raw) | 56,728 legitimate / 44,256 phishing |
| Sender/receiver identifiers | Already hashed/pseudonymised in this file; message text uses typed placeholders (`[NAME]`, `[ORGANIZATION]`, `[URL]`, etc.) in place of PII |
 
Phase 1 cleans this down to the 99,707-email corpus reported in the paper (removing unlabeled/empty records, exact duplicates, and campaign-level near-duplicates) and writes it to `preprocessed/mepc_clean.parquet` for Phase 2 and Phase 3.
 
The TREC 2005–2007 collections are published by NIST (Cormack & Lynam). `phishing_pot` is an MIT-licensed honeypot collection ([`rf-peixoto/phishing_pot`](https://github.com/rf-peixoto/phishing_pot)); the paper uses a fixed snapshot taken on the access date given in its references, since the source repository no longer receives public updates — reproducing exact corpus statistics requires that snapshot rather than the live repository. Per the paper's Data Availability Statement, the assembled corpus is also available from the corresponding author, **Umair B. Chaudhry**, upon reasonable request.
 
---
 
## How to Run
 
1. Open each notebook in **Google Colab**.
2. Mount Google Drive when prompted; the notebooks expect (and create, where missing) `My Drive/Thesis/`, with subfolders `preprocessed/`, `models/`, `figures/`, `tables/`.
3. Place `Multi-source Email Phishing Corpus.csv` in `My Drive/Thesis/`.
4. Run **Phase 1** first (CPU runtime is sufficient). It has no GPU dependency and installs only `pyarrow`.
5. Switch to a **T4 GPU runtime** and run **Phase 2**. It auto-installs `transformers` if missing. `Runtime → Run all` is safe to re-issue after a disconnect — cached artefacts and partially trained encoders are reloaded rather than recomputed.
6. Run **Phase 3** (installs `shap` and `lime`); it only reads Phase 1/Phase 2 artefacts, so a GPU is optional unless you re-run the Part B baselines.
A `QUICK_TEST` flag near the top of Phase 2's configuration cell runs a ~10-minute smoke test on tiny subsamples, writing to `Thesis/quicktest/` so it can never overwrite or be mistaken for a real run.
 
**Fixed seed:** all notebooks seed with `SEED = 42`; Phase 2's statistical-robustness section (Part I) additionally replicates across seeds 42–46.
 
**Key training settings** (full detail in Phase 2, Part K's hyperparameter table): `distilbert-base-uncased`, max sequence length 256, batch size 32, AdamW with decoupled weight decay, linear warmup over the first 10% of steps then linear decay, mixed precision, adaptive epoch budget (starts at 10, extends by 5 to a ceiling of 25, early stopping on validation F1 with patience 3), class-weighted cross-entropy only (no oversampling/undersampling/synthetic data).
 
---
 
## Outputs
 
Running all three notebooks end to end populates `My Drive/Thesis/`:
 
- **`preprocessed/`** — cleaned corpus (`mepc_clean.parquet`), per-split structural feature arrays, fitted scalers, `config.json`.
- **`models/`** — fine-tuned DistilBERT encoders (one per split design), MLP head checkpoints (fusion and ablated), classical/CNN baseline artefacts.
- **`figures/`** — every figure reported in the paper (corpus composition, leakage diagnostics, split design, training curves, confusion matrices, ablation, threshold sensitivity, cross-lingual coverage, SHAP/LIME plots).
- **`tables/`** — every table reported in the paper, exported as both CSV and GitHub-flavoured Markdown (corpus summary, cleaning log, split distributions, structural feature statistics, master model comparison, generalisation gap, bootstrap CIs, McNemar tests, per-language evaluation, tokenizer coverage, and more).
These are the artefacts the paper's tables and figures are generated from directly — nothing in the manuscript is transcribed by hand.
 
---
 
## Results Summary
 
Full results, confidence intervals, and significance tests are in the paper. Headlines, for orientation:
 
- **In-distribution (Split A):** nine models cluster within ~5 F1 points of each other; the fusion model shows **no measurable benefit** over an identical head with the structural block removed (ΔF1 = 0.0002, 95% CI [−0.0003, 0.0007]).
- **Cross-source (Split B, `phishing_pot` withheld entirely):** that null result reverses. The fusion model significantly outperforms its own ablation on both F1 (ΔF1 = 0.0067, 95% CI [0.0043, 0.0092]) and recall (Δ = 0.0102, 95% CI [0.0063, 0.0147]), McNemar-significant after Bonferroni correction. **The contribution of multimodal fusion is conditional on distribution shift** — invisible under standard in-distribution evaluation, measurable only when provenance changes.
- **Ablation:** retraining (not attribution-based) shows structural features cut false negatives by up to 10.8% in-distribution without moving F1 — they buy recall, not overall classification quality, under that protocol.
- **Explainability:** SHAP places 97.6% of total attribution in the text embedding block; the remaining 2.4% (structural features) is exactly what produces the cross-source gain — attribution magnitude computed in-distribution does not predict a feature's conditional contribution under shift.
---
## Funding, Consent & Conflicts of Interest
 
**Funding:** This research received no specific grant from any funding agency in the public, commercial, or not-for-profit sectors.
**Consent for Publication:** All authors give consent for the publication of identifiable details, which can include photograph(s) and/or videos and/or case history and/or details within the text to be published in the above Journal and Article.
**Conflicts of Interest:** All authors declare that they have no conflicts of interest.
 
## License
 
Code and notebooks in this repository are licensed under the MIT License. The license does not cover the third-party corpora used (TREC spam tracks, `phishing_pot`), which retain their original licenses and terms of use.
