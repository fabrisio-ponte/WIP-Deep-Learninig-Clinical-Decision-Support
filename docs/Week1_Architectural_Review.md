# BEHRT Week 1 Architectural Review

Source material: `Ponte_Vela_BEHRT_2025_.pdf`, original BEHRT paper (Li et al. 2020), and the code in `BEHRT_WIP/BEHRT_run/BEHRT_Project` (production pipeline) and `BEHRT_WIP/BEHRT_run_reverse_eng` (ablation experiments).

## Executive summary for publication

This review is meant to serve as a working publication-risk audit, not as a final manuscript. The central finding is that the project already has a plausible BEHRT-style architecture and a credible clinical story, but the current results are not yet publication-ready because several foundational details are inconsistent across code, methods, and reported metrics.

The main issue is not a weak concept; it is a reproducibility and documentation problem. The model has a coherent design, the project already contains cleaned data and cleaned-vocabulary workflows, and the quick/full split gives us a sensible research pipeline. The remaining work is to turn that pipeline into a traceable, reviewer-proof experimental record.

The publication path is straightforward:

1. Use the quick pipeline for iteration and debugging.
2. Fix the data leakage and noise-code issues before any final reporting.
3. Recompute the headline metrics on the cleaned datasets.
4. Regenerate all tables and figures from saved artifacts.
5. Only then write the paper narrative in a way that stays faithful to the code.

This is a project with a realistic path to publication, but not yet with a fully defensible paper draft. The current document identifies what must be fixed before that handoff.

## 1. Design rationale (paper's four contributions, one sentence each)

| Design choice | Why it was made | Alternative considered |
|---|---|---|
| **CCSR code embedding** (`Evisit`, 474-dim vocab → 144-dim) | Aggregating ~89K ICD-10 codes to 474 CCSR categories gives enough samples per class for a multi-label head to learn on a 24.7K-patient cohort. | Keep raw ICD-10 codes (finer-grained but far too sparse per class for this dataset size). |
| **Age-group embedding** (`Eage`) | Adds a life-stage signal that interacts with disease risk (e.g., cardiac risk at 65 vs. 35) independent of visit order, which a pure positional encoding can't express. | Concatenate age as a scalar feature instead of an embedding — cheaper but loses the model's ability to learn non-linear age-disease interactions. |
| **Compact architecture (3 layers, 144-dim, 6 heads)** | Matches model capacity to the ~25K-patient dataset to avoid overfitting versus the original BEHRT's 100M-parameter design built for 1.6M patients. | Use the full-size original config and rely on aggressive regularization/dropout — riskier on this dataset size. |
| **MLM pre-training → fine-tuning** | Bidirectional self-attention pre-trained on the unlabeled masking task learns general disease co-occurrence patterns before the harder multi-label task, standard BERT-family transfer. | Train the multi-label head from a randomly initialized encoder — faster but forgoes the representation-learning benefit BEHRT is built on. |
| **Multi-label next-visit head** (sigmoid over 474 classes) | Frames next-visit prediction as 474 independent binary decisions so the model can output multiple co-occurring diagnoses per visit, matching real comorbidity patterns. | Single-label (top-1) classification — simpler but clinically wrong, since patients present with multiple conditions per visit. |
| **Segment embedding** (A/B alternating per visit — *present in code, not in paper*) | Carried over unmodified from the original BEHRT implementation to mark visit boundaries in the flattened CLS/SEP token stream. | Omit it, since position embeddings already encode visit order — this is what the paper's Eq. 1 implicitly claims was done. |

## 2. Discrepancies between the paper and the actual code

These need to be resolved (or the paper corrected) before Week 2, since a JAMIA reviewer with ML background will check every one of them against a public repo.

**Segment embedding is used but undocumented.** Eq. 1 in the paper lists only three embedding types (`Evisit + Eage + Epos`). `model/NextXVisit.py` and `dataLoader/utils.py` both implement and actively use a fourth: a segment embedding that alternates 0/1 at every visit boundary, exactly as in the original BEHRT. Either add it to Eq. 1 and Table I, or genuinely ablate it out and retrain.

**Two incompatible model configs exist in the repo, and the reported numbers come from the smaller, non-default one.** `MLM.ipynb` / `NextVisit-6month.ipynb` / `NextVisit-12month.ipynb` all use the *original* BEHRT size (hidden 288, 6 layers, 12 heads) — not the compact design described in the paper. The 144/3/6 "compact" config that matches Table I only exists in `MLM_QUICK.ipynb` / `NextXVisit_QUICK.ipynb`, and `TODO.md` explicitly lists "Train Full BEHRT Model" (the 288/6/12 config) as an *unfinished, next-step* item. This means the "full cohort, compact architecture" story in the paper is really the "quick" prototype run — worth being precise about in the methods section.

**The reported result was trained/evaluated on a subsample, not the full cohort.** `NextXVisit_QUICK.ipynb` sets `data_fraction: 0.8` and applies `.sample(frac=0.8)` to *both* the train and test parquet files before training/evaluation. The paper states "evaluated on 24,729 patients (179,716 visits)" without qualifying that the actual reported APS/AUC came from an 80% random subsample of that cohort's test split. Minor, but should be stated exactly for reproducibility.

**The claimed "~5M parameters" doesn't match the actual config.** Computing parameter count directly from the `MLM_QUICK`/`NextXVisit_QUICK` config (hidden=144, 3 layers, 6 heads, vocab≈478, age_vocab=1,322 [ages tracked in months, 0–110 yrs], intermediate_size=**256**, not the 576 written in Eq. 7–8 of the paper) gives roughly **0.8–1.1M trainable parameters**, not 5M — even using the paper's stated intermediate_size of 576 it only reaches ~1.1M. Two separate things to fix: (a) `intermediate_size` in code is 256, but Eq. 7's `W1 ∈ R^{144×576}` says 576 — pick one and make code/paper match; (b) recompute the actual parameter count from the saved checkpoint (`sum(p.numel() for p in model.parameters())` on `behrt_nextvisit_ccsr_quick.pt`) and correct Table I. This doesn't weaken the "compact" story — a true ~1M-parameter model matching BEHRT-level AUC is arguably a *stronger* efficiency claim — but the number as printed is off by ~5x and needs to be right for a journal submission.

**The reported test set still contains the "removed" noise code.** Section IV of the paper states `XXX000` (8.08% frequency, generic catch-all) is dropped in "Phase 1 — Noise Removal" before the final 474-CCSR vocabulary is built. But `comprehensive_disease_analysis_results.json` — whose `sample_wise_aps: 0.4057` and `sample_wise_auc: 0.9190` are the *exact* headline numbers in the paper — shows `CCSR_XXX000` as label index 473 of 474, with 14,866 test-set instances and the single best F1 score of any class (0.898, precision 0.81, recall 1.00). `TODO.md` corroborates this: it records discovering the XXX000 leak and writing `clean_data.py` to remove it *after* this analysis had already been run and reported. In other words, the headline metrics currently in the paper were computed on data that still includes the noise code the methods section says was removed — this one class alone (huge support, trivially easy to predict) is likely inflating both APS and AUC. **This is the highest-priority fix**: rerun the QUICK training + evaluation pipeline on the actually-cleaned data (`clean_data_phase2.py` / `*_clean.parquet` files already exist in `data/processed/`) and replace every metric in the paper with the clean-data numbers before anything gets reformatted for JAMIA.

**Table II traceability is now partially fixed, but still needs manuscript alignment.** The analysis utility now emits both JSON and a flat per-disease CSV with AP/AUC fields (`comprehensive_disease_analysis_results_per_disease_metrics.csv`), so disease-level ranking metrics are now exportable from code artifacts. Remaining risk: the manuscript must explicitly cite the exact checkpoint/config used to generate that table, and should not mix values from different runs (QUICK subsample vs. clean full-data runs).

## 3. Weakest CCSR categories, and why

## Publication-risk summary

We now have a clear distinction between three categories of issues:

- Structural issues: architecture, config, and code/data mismatches.
- Statistical issues: inflated metrics due to generic label leakage and non-cleaned evaluation.
- Editorial issues: values that need to be made reproducible and traceable in the manuscript.

The architecture itself is solid enough to keep. The actual publication risk comes from the mismatch between the story we want to tell and the numbers currently produced by the code. That mismatch is fixable without rewriting the core paper idea, but it must be corrected before submission.

## Verified facts (as of 2026-08-18)

These are now directly validated in the workspace and should be treated as ground truth for drafting:

- Cleaned splits exist and `CCSR_XXX000` is absent in train/val/test (`*_clean.parquet`).
- The most recent low-metric run (`clean_run_20260818_122404`) is a smoke configuration, not a publication baseline:
	- `sample_limit = 2000`
	- `epochs = 1`
	- `sample_wise_aps = 0.0538`
	- `sample_wise_auc = 0.6079`
- Full-style clean runs (no sample limit, 3 epochs) span a much higher range and are not monotonically declining.
- Parameter count from a saved full clean checkpoint (`clean_run_20260813_010458/behrt_nextvisit_ccsr_clean_best.pt`) is `3,650,105` trainable parameters.

## Strict assumptions, breakpoints, and limitations

This section is written in publication-audit style and should map directly into methods/limitations text.

### A. Assumptions required for claim validity

1. **Data hygiene assumption:** all reported final metrics come from `*_clean.parquet` splits with `CCSR_XXX000` removed.
2. **Run comparability assumption:** compared runs share the same data scope (full vs subsample), epoch budget, seed policy, and checkpoint-selection rule.
3. **Architecture fidelity assumption:** manuscript architecture (embeddings, hidden size, layers, heads, intermediate size) exactly matches executed code/config.
4. **Artifact lineage assumption:** every table/figure in the manuscript is generated from a named file artifact, not from notebook output copied manually.

### B. Where the pipeline breaks (failure conditions)

1. **Smoke/full conflation:** if a smoke run (e.g., limited rows or 1 epoch) is compared against full runs, trend interpretation becomes invalid.
2. **Cross-run metric mixing:** if Table I and Table II pull from different checkpoints without explicit disclosure, claims become non-reproducible.
3. **Config drift:** if `analysis_config.json` checkpoint/model settings differ from manuscript settings, reproduced numbers will not match.
4. **Threshold-metric mismatch:** reporting only AUC/APS without the corresponding thresholded behavior (e.g., many classes with F1=0) can overstate usable clinical performance.
5. **One-class label edge cases:** per-disease AUC is undefined when a class has only positives or negatives in the evaluation split; this must be handled and disclosed.

### C. Limitations that must be stated explicitly

1. **Class imbalance remains severe** across hundreds of CCSR labels; ranking metrics can look strong while fixed-threshold utility is sparse for rare labels.
2. **Temporal and institutional generalizability is unproven** (single-source data processing path; no external-site validation shown here).
3. **Calibration is not yet reported**; clinical decision support use requires probability calibration and decision-curve analysis, not only discrimination metrics.
4. **Run-to-run variability exists** (seed and optimization sensitivity), so confidence intervals or repeated-run summaries are needed for robust claims.
5. **Interpretability evidence is partial**; attention/feature narratives should be treated as supportive, not causal explanations.

## Week 2 priority plan

### Phase 1: Stabilize the experimental baseline

- Reproduce the cleaned data training run on the existing `_clean.parquet` files.
- Verify that the generic noise code is absent from train/val/test before evaluation.
- Confirm the quick run remains the default iteration mode while the full run is reserved for final validation.

### Phase 2: Recompute the paper metrics from saved artifacts

- Use a reproducible script to generate APS/AUC and per-disease metrics.
- Save each result set as a file in the project results directory.
- Make sure every figure or table in the paper can be regenerated from that artifact.

### Phase 3: Align the architecture description with the actual code

- Decide whether segment embedding is part of the final architecture.
- Reconcile the model size and intermediate dimension values across code and manuscript.
- State the train/test subsampling explicitly if it remains part of the pipeline.

### Phase 4: Rewrite the manuscript narrative to reflect what is actually true

- Keep the high-level story: compact BEHRT-style model, age-aware multi-label next-visit prediction, strong chronic disease signals.
- Remove or soften any claim that depends on the generic noise code still being present.
- Frame performance as a clinically meaningful subset result rather than a fully cleaned global benchmark unless the clean rerun confirms it.

### Phase 5: Final publication gate

The paper is ready for a clean draft pass only when all of the following are true:

- the clean-data metrics are reproducible;
- the headline numbers come from the same data the methods section claims;
- the architecture description matches the code;
- the tables are generated from saved artifacts, not hand-transcribed from memory;
- the quick/full distinction is clearly explained as an iteration workflow, not a hidden model mismatch.

If those gate conditions are met, the project becomes publishable in a strong, transparent way without changing the core scientific direction.

From `comprehensive_disease_analysis_results.json` (F1 at the default 0.5 threshold, computed on the QUICK model / 80%-subsampled test set):

**465 of 474 CCSR categories score F1 = 0.0.** Only 9 classes have any nonzero F1 at all. That's a much starker picture than the paper's macro APS (40.6%) and AUC (91.9%) suggest on their own — those are threshold-free ranking metrics, so a model can rank the true label reasonably high (good AUC/APS) while still never crossing a fixed 0.5 probability threshold for the vast majority of rare classes (F1 = 0). The two metrics aren't contradictory, but the F1 collapse is worth stating explicitly in the limitations section rather than only showing AUC/APS.

The 9 classes with signal, and what they have in common:

| CCSR code | Disease | F1 | Precision | Recall | Support |
|---|---|---|---|---|---|
| CCSR_XXX000 | *(noise code — should not be a label; see §2)* | 0.898 | 0.81 | 1.00 | 14,866 |
| CIR019 | Heart failure | 0.616 | 0.65 | 0.59 | 4,023 |
| END010 | Disorders of lipid metabolism | 0.613 | 0.58 | 0.66 | 6,218 |
| GEN003 | Chronic kidney disease | 0.604 | 0.64 | 0.57 | 4,061 |
| CIR011 | Coronary atherosclerosis / other heart disease | 0.600 | 0.60 | 0.60 | 4,142 |
| FAC025 | Other specified status (administrative/residual bucket) | 0.413 | 0.56 | 0.33 | 5,100 |
| CIR007 | Essential hypertension | 0.402 | 0.63 | 0.30 | 6,198 |
| CIR017 | Cardiac dysrhythmias | 0.222 | 0.59 | 0.14 | 3,800 |
| END003 | Diabetes mellitus with complication | 0.007 | 0.50 | 0.003 | 3,180 |

Why these and not others: all 8 real disease categories with signal are high-prevalence, high-support chronic conditions (support 3,180–6,218, among the largest classes in the 474-way label set) that also have strong self-referential temporal structure — once diagnosed, they tend to reappear at every subsequent visit, which is exactly the "long-range head" pattern the paper's attention analysis (§VIII) describes. Everything below roughly 3,000 supporting examples appears to fall below the model's learnable signal-to-noise floor at this dataset size and training budget, and gets predicted at low enough probability that it never crosses the 0.5 threshold — consistent with what the reverse-engineering ablation (`REVERSE_ENGINEERING_RESULTS.md`) found separately: performance tracks how clean/simple/high-signal the pattern is, and degrades sharply as realistic multi-class noise increases.

One correction worth making in the demo code while you're in there: `demo/backend_api.py`'s `get_code_description()` mislabels `CIR007` as "Heart failure" — per the official CMS CCSR reference table, CIR007 is **Essential hypertension** and CIR019 is the actual Heart failure code.

## 4. Recommended order of operations before Week 2

1. Freeze one publication baseline checkpoint and explicitly tag all smoke runs as non-comparable diagnostics.
2. Generate Table II only from the saved per-disease artifact file, and record the exact generating checkpoint/config in the caption or supplement.
3. Update Table I parameter-count claim using the checkpoint-derived count for the selected baseline (do not reuse stale approximate values).
4. Resolve architecture text/code parity: segment embedding status plus `intermediate_size` mismatch.
5. Add a limitations paragraph that explicitly separates threshold-free ranking performance from thresholded clinical utility.

None of this changes the paper's core story — compact BEHRT variant, age embeddings, strong performance on chronic cardiometabolic disease — it just needs the numbers underneath it to be the numbers a reviewer would get by cloning the repo and rerunning the QUICK notebooks.
