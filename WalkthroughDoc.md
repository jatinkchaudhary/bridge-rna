# ExpressionPerformer — `examples/` Walkthrough

A scientific reading of the five evaluation notebooks in `examples/`: what each one asks, which checkpoint and data it touches, what it actually found, and where the design has oversights that are likely hampering the results.

---

## 0. The shared system under test

Every notebook (except the legacy `5b_`) loads one pretrained artifact and probes it. Understanding the model first makes the notebooks legible.

**Model — `ExpressionPerformer` (`train.py`).** A transcriptome is treated as a *set* of ~15,165 genes in a fixed canonical order (`data/archs4/train_orthologs/canonical_genes.csv`), not a sequence. The input hidden state for each gene slot is the **sum** of two embeddings:

- a learned **gene-identity** embedding (`nn.Embedding(num_genes, hidden_dim)`) — "which gene is this slot";
- a **Rotary Expression Embedding (REE)** — a sinusoidal encoding driven by the *magnitude* of expression, injecting the continuous value the way rotary encodings inject position.

These pass through Performer (linear-attention) layers and are trained with a single **masked-language-modeling (MLM)** objective: random genes are replaced with mask token `-10` and the model predicts their `log1p(TPM)` value. Output is `[batch, genes, hidden_dim]`.

**Three ways the model is consumed:**
1. `predict_expression` → MLM head, for **imputation** scoring.
2. `extract_transcriptome_embeddings` → pools per-gene hidden states into one **512-d sample embedding** (mean / multi-scale pooling).
3. Raw per-gene hidden states `[G, D]` → **interpretability** (saliency, per-gene shifts).

**Two checkpoints appear across the notebooks — keep them straight:**

| Run | Used by | Params | Layers / hidden / ffn | Val loss | Epoch | Present in this copy? |
|---|---|---|---|---|---|---|
| `20jo1hdd` (small) | `tcga_example`, `embeddings`, `osdr_example` | 19.3 M | 4 / 512 / 2048 | 0.611 | 5 | ✅ yes |
| `s66qfh36` (large) | `osdr_example_large` | 45.6 M | deeper | **0.161** | 10 | ❌ **directory missing** |

Both map the same 15,165-gene dictionary. The large model has a far better pretraining loss — the central question below is whether that translated downstream.

**The recurring three-part scientific question** (asked of every dataset):
1. *Imputation* — can the MLM head recover masked expression? (tests the pretraining objective directly)
2. *Representation* — does the embedding space organize biology (PCA/t-SNE) beyond raw features?
3. *Probe* — do embeddings **beat the raw `log1p(TPM)` baseline** under honest cross-validation?

---

## 1. `tcga_example.ipynb` — the clean in-domain template

This is the reference design pattern; the other notebooks are elaborations of it.

**Checkpoint:** `20jo1hdd`. **Data:** 3,481 TCGA samples, input `log1p(tpm_unstranded)` to match training. Task: BRCA vs LUAD (balanced).

**Pipeline / checkpoints in the code:** build TCGA matrix from raw GDC files → assert gene dictionary matches `gene_embedding.weight.shape[0]` (hard gate) → imputation → PCA + t-SNE on raw vs log-TPM vs embeddings → RandomForest probe (StratifiedKFold, F1-macro).

**Key results:**

| Probe (F1-macro, RF, 5-fold) | Score |
|---|---|
| log1p(TPM) raw baseline | **0.998 ± 0.002** |
| PCA(64) on log1p(TPM) | 0.987 ± 0.009 |
| ExpressionPerformer embeddings | 0.940 ± 0.026 |

Imputation: global Pearson **0.838**, Spearman 0.834 at 15% masking.

**Read:** the embeddings carry strong cancer-type signal (0.94 F1) but **do not beat the raw baseline** — and on a two-cancer problem the raw baseline is near-perfect, so headroom is small. First appearance of the pattern that holds everywhere: *embeddings < raw*.

---

## 2. `embeddings.ipynb` — what do the gene tokens encode?

**Scientific question:** stripped of any expression input, has the **gene-identity** embedding learned biology — do co-embedded genes share pathways?

**Method:** feed an all-zero transcriptome so only the gene embedding speaks → 20-cluster KMeans over the 15,165 gene vectors → KEGG/GO enrichment (`gseapy`/Enrichr) per cluster.

**Key results:** 17 cluster–library enrichments at adj-p < 0.05, 24 at < 0.10. Strongest: cluster 8 → *ABC transporters* (adj-p ≈ 3e-55) and *Fatty Acid Metabolic Process* (≈7e-21); others recover proteasome, glycosaminoglycan biosynthesis, neuroactive ligand–receptor.

**Read (the notebook's own, appropriately hedged):** structure is real but *selective, not globally clean* — "moderate rather than spectacular." Several clusters remain broad/mixed. Honest and correct.

---

## 3. `osdr_example.ipynb` — the full case study (NASA spaceflight)

Same small checkpoint (`20jo1hdd`), a much harder, confounded problem: mouse spaceflight-vs-ground RNA-seq.

**New design elements (and why):**
- **Cross-species ingestion** — 60,888 mouse genes → one-to-one human orthologs (44,780 dropped for no mapping), TPM from GENCODE mouse exon lengths, QC `≥14,000` nonzero genes (7 samples dropped). Final: **2,101 samples (775 spaceflight / 1,326 control), 15,165 genes**, ~90% of model genes observed per sample. All logged in `ingest_diagnostics` — strong practice.
- **Study-aware validation (`GroupKFold` on study accession)** — the central safeguard against the classifier memorizing batch/study identity.
- Low-label curves, tissue-stratified probes, per-gene embedding shift, fine-tuning, contrastive saliency with explicit de-biasing.

**Key results:**

| Probe (F1-macro) | Embeddings | log1p(TPM) | PCA(64) |
|---|---|---|---|
| StratifiedKFold RF | 0.658 ± 0.013 | **0.729 ± 0.020** | 0.712 |
| **GroupKFold RF** (honest) | 0.548 ± 0.039 | **0.621 ± 0.065** | 0.597 |
| Linear probe (LogReg) | 0.679 | **0.805** | 0.648 |

- Imputation (simple cell): global Pearson **0.898**. **But the richer control cell is the important one:** per-gene Pearson is only **0.687**, and a trivial *predict-the-gene-mean* baseline already scores global Pearson **0.816** — so the model's apparent 0.90 is inflated by between-gene dynamic range, and its true lift over a mean-predictor is modest. A shuffled control sits at ~0.0 (sanity passes).
- **Known spaceflight-gene enrichment: 0/8 in every category** (radiation stress, muscle atrophy, immune, oxidative, circadian, bone) appear in the top-40 embedding-shift genes — a clean **negative result**.
- Fine-tuning (frozen MLP ≈ 0.51, frozen RF ≈ 0.50) **underperforms even the frozen-embedding RF**, which underperforms raw.

**Read:** under honest GroupKFold, embeddings trail raw by ~7 F1 points, and the model does not surface canonical spaceflight biology. The notebook is methodologically careful; the model is simply not winning here.

---

## 4. `osdr_example_large.ipynb` — scaled, exhaustive, large checkpoint

Same OSDR data, the **45.6 M-param `s66qfh36`** checkpoint (val loss 0.16 — 4× better pretraining than the small model), multi-GPU inference, and a deep battery: linear probe, embedding-dim→gene mapping, MLP heads, per-fold top-K selection, adversarial de-confounding, XGBoost over multiple targets.

**Key results:**

- **Imputation jumps to global Pearson 0.979 / Spearman 0.976** — the better pretraining loss shows up here, as expected.
- **…but downstream classification does NOT improve.** Embeddings F1 = **0.667** vs raw log1p(TPM) **0.729** (StratifiedKFold) — essentially identical to the small model. *A 4× better MLM loss bought zero downstream gain.* This is the single most important finding in the set.
- **Compression headline:** 15,165 genes → 512 dims = **29.6× compression** retaining **88.4%** of raw-TPM F1. True, but note it is **retention, not improvement** — a lossy compressor, not a better representation.
- **Dimension 13** is pitched as a "mitochondria vs stress" axis. Statistics: Cohen's d = **−0.105**, single-dim **AUC = 0.534**, Mann–Whitney p = 0.009. Significant but the effect size is near-chance (AUC 0.53); the biological narrative in the markdown cells (cells 17/21) **over-interprets a very weak signal**.
- XGBoost across targets: embeddings ≈ raw on Sex (0.990 vs 0.996) and Strain (0.900), and **below** raw on Spaceflight (0.684 vs 0.773). The model preserves coarse biological axes (sex, strain) better than the subtle perturbation (spaceflight).
- Adversarial fine-tuning and MLP heads land ~0.56–0.60 grouped — no better than frozen.

---

## 5. `5b_tcga_analysis.ipynb` — legacy / different model lineage

**This is NOT the ExpressionPerformer.** It loads `bulkformer_checkpoints/best_model.pt`, a global `checkpoints/config.json`, ESM-2 gene embeddings (`esm2_t6_8M_UR50D_gene_embeddings.pt`) and a GNN `edge_index_top20.pt`. It is a **BulkFormer**-style pipeline (ESM-2 + graph), an earlier/parallel comparison. Its standout component is a **TPM self-validation**: it recomputes TPM from GENCODE exon lengths and checks it against GDC's own TPM on a log-log diagonal — a good normalization sanity check that the ExpressionPerformer notebooks lack. It then runs the same impute/PCA/t-SNE/RF battery.

**Status in this repo:** the paths it needs (`bulkformer_checkpoints/`, `checkpoints/config.json`, `data/embeddings/…`, `graph/…`) are **absent** from the synced tree, so it will not run as-is. Treat it as historical reference, not a live benchmark.

---

## 6. Consolidated scoreboard (F1-macro, embeddings vs raw)

| Dataset / setting | Embeddings | Raw log1p(TPM) | Winner |
|---|---|---|---|
| TCGA BRCA/LUAD, RF | 0.940 | 0.998 | raw |
| OSDR, RF Stratified | 0.658 | 0.729 | raw |
| OSDR, RF GroupKFold | 0.548 | 0.621 | raw |
| OSDR, linear probe | 0.679 | 0.805 | raw |
| OSDR large, RF Stratified | 0.667 | 0.729 | raw |
| OSDR large, XGB Spaceflight | 0.684 | 0.773 | raw |
| OSDR large, XGB Sex | 0.990 | 0.996 | ~tie |

**The embeddings lose to the raw baseline in every supervised comparison.** They impute well (especially the large model) and compress ~30× at ~88% retention, but as a downstream feature they are currently a *lossy re-encoding of the input*, not an enrichment of it.

---

## 7. Oversights likely hampering the output

Ordered by how much they affect the conclusions.

**A. Headline imputation numbers are inflated (high impact on interpretation).** The "simple" imputation cells report a single **global** Pearson pooled over all masked entries, which is dominated by between-gene dynamic range and by structural zeros (masking a zero and predicting ~zero is trivial). The richer cell shows per-gene Pearson is ~0.69 and a *gene-mean* predictor already reaches ~0.82 global. **Fix:** lead with per-gene / per-sample correlations and always report the gene-mean baseline alongside; restrict masking to expressed genes (or report expressed-only separately).

**B. Better pretraining loss did not improve downstream features (the key scientific gap).** `s66qfh36` cut val loss 4× and pushed imputation to 0.98, yet downstream F1 was flat (0.667 ≈ 0.658) and still below raw. This is the result to investigate, not paper over — it suggests the MLM objective and/or the pooling are not aligned with sample-level discrimination. **Fix:** test alternative readouts (attention-pool, `[CLS]`-style token, or supervised contrastive pretraining), and report the gap explicitly rather than foregrounding compression.

**C. Fine-tuning backbone is configured from the WRONG config (correctness bug).** In `osdr_example_large` the fine-tuning cell loads `RUN_DIR.parent/"config.json"` = the **global** `checkpoints_performer/config.json` (`num_layers=2, hidden_dim=128, num_genes=16109`), which matches **neither** the frozen-embedding backbone (`s66qfh36`, taken from `payload["config"]`) **nor** the data (15,165 genes). The printed banner even says "genes=16109" while `X_sub` is 15,165. So "frozen vs fine-tuned" is not an apples-to-apples comparison, and the fine-tune/adversarial numbers may be running on a mis-specified or partially-loaded backbone. **Fix:** load the run-specific config from `payload["config"]` (or `RUN_DIR/config.json`) and assert `num_genes` and layer dims match both the checkpoint state-dict and the data before training.

**D. TCGA probe is not study/patient-aware (leakage risk).** `tcga_example` uses `StratifiedKFold`, not `GroupKFold`. TCGA can have multiple aliquots per patient; the near-perfect 0.998 may be partly optimistic. OSDR correctly uses GroupKFold — apply the same standard to TCGA (group by patient/`case_id`). The clean gap between Stratified (0.658) and Group (0.548) in OSDR shows how much this matters.

**E. Interpretability is statistically fragile and over-narrated.** (i) Saliency top-genes were 90% "A"-initial — a pure alphabetical/index artifact of the canonical gene order. The author caught this and added a de-biasing pass, but the de-biased list still shows residual letter enrichment (G/D), so rankings remain unstable; treat the gene lists as exploratory, not findings. (ii) The dimension-13 "mitochondria vs stress" story rests on AUC 0.53 / Cohen's d −0.11 — essentially chance-level effect size dressed in strong biological language (markdown cells 17, 21). **Fix:** gate any biological claim on a minimum effect size (e.g. AUC ≥ 0.65) and on stability across folds/seeds; soften the narrative cells.

**F. Reproducibility / hygiene (low impact, but real).** Leftover `print('hello')` debug cells (osdr_example cells 1, 6); many empty cells in `osdr_example_large` (12, 13, 39–44); deprecation warnings (`plt.cm.get_cmap`, `Series.fillna(method=…)`, `torch.cuda.amp.GradScaler`); a `DataLoader` worker-shutdown traceback (cell 32); LogReg/SVM `ConvergenceWarning` and ill-conditioned RidgeClassifier on the 15k-dim raw inputs (increase `max_iter`, or standardize/scale-reduce before linear models). The `s66qfh36` checkpoint is absent from this synced copy, so the large notebook cannot be re-run here — pin checkpoints or document their location.

**G. No normalization cross-check in the ExpressionPerformer notebooks.** The TPM-vs-GDC diagonal validation only exists in the legacy `5b_` notebook. Since mouse OSDR TPM is computed in-house from exon lengths, a comparable sanity check (e.g. correlate a held-out recomputed TPM against any reference) would harden the cross-species ingestion claims.

---

## 8. One-paragraph takeaway

The harness is well-designed and unusually honest — group-aware CV, shuffle controls, mean-predictor baselines, saliency de-biasing, and full ingest diagnostics are all present. The substantive problem is the **result**, not the methodology: across TCGA and OSDR, at two model scales, the ExpressionPerformer's sample embeddings reliably **trail the raw `log1p(TPM)` baseline** on every supervised task, even as imputation and compression look strong. The most valuable next step is diagnosing **why a 4× better MLM loss yields no downstream gain** (oversight B), after fixing the fine-tuning config mismatch (C) and reporting the de-inflated imputation metrics (A).
<img width="154" height="150" alt="expressionperformer_system_design" src="https://github.com/user-attachments/assets/b13f55a4-2b65-4b9c-b999-c7d4b9373a85" />
