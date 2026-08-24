# ACT

**Auditable CT phenotyping through report-derived radiological observations**

ACT maps a native-resolution chest or abdominal CT volume to similarities with
376,194 report-derived radiological observations, then projects those scores
through a large language model embedding space. Predictions become decomposable
into named observations. Probes trained on this representation can be audited
against the observation bank, and the bank can be restricted to clinically
trusted observations without retraining the image encoder.

The manuscript is under submission. Model weights and redistributable
observation-bank artefacts will be available for academic research at
[huggingface.co/peterhan91/clip_3d_ct](https://huggingface.co/peterhan91/clip_3d_ct)
upon publication (the repository is private until then).

## Repository layout

| Directory | Contents |
|---|---|
| `model/` | Volume-report model: architecture, DDP training, inference, zero-shot evaluation, Hugging Face checkpoint loading. Flat module layout, import-compatible with the analysis code (`CLIP3D_REPO` points here). |
| `preprocessing/` | Native CT preprocessing (NIfTI to HDF5), report cleaning, MedGemma impression generation, split construction for the training datasets. External evaluation datasets are prepared by `analysis/preprocess_new/`. |
| `analysis/` | Observation-bank extraction and embedding, the concept-anchored representation (`clear3d/`), phenotype probing, probe-observation audit, observation-bank restriction, statistics, and figure generation (`analysis/experiments/`). Mirrors the source layout so bare-name imports and relative paths keep working. |
| `phenotypes/` | 221-phenotype label building (phecode mapping, INSPECT extraction) and rule manifests. |

No patient-level data ships here. Datasets come from their own gated sources
under their own terms (see Data availability in the paper). Configure your
local paths via CLI flags or `.env` (template: `.env.example`). Cluster paths
in scripts were replaced with `/path/to/...` placeholders.

## Installation

Python 3.10 or newer.

```bash
pip install -r requirements.txt
```

Optional stacks (figures, concept extraction, external preprocessing,
baselines) are listed as commented extras in `requirements.txt`. Training ran
on x86 with CUDA 12.4 (2x A100, torchrun DDP); analysis ran on Python 3.12,
torch 2.11, NVIDIA GH200. Concept extraction needs a separately launched vLLM
OpenAI-compatible server. Each zero-shot baseline documents its own
environment in its `setup_env.sh`.

## Loading the pretrained model

```python
from load_pretrained import load_act  # in model/

model, preprocess_text = load_act()   # downloads clip_3d_ctrate_merlin_v1 from Hugging Face
```

The checkpoint instantiates a DINOv2 ViT-B/14 slice encoder with register
tokens, a two-layer Transformer slice-fusion module (12 heads, feed-forward
multiplier 2, rotary embeddings), a 768-d projection, and a CLIP-style text
tower (width 512, 12 layers, context length 77).
`model/run_scripts/train_release_v1.sh` documents the full training recipe
recovered for this checkpoint (symmetric InfoNCE computed per GPU without
cross-device gathering, AdamW lr 1e-4, weight decay 0.2, batch 4 per GPU with
gradient accumulation 32, seed 42, checkpoint selection on CT-RATE validation).

## Reproducing the paper

Pipeline order, from raw dataset access to figures:

1. **Preprocess**: split and path CSVs via
   `preprocessing/run_scripts/generate_*_split_csvs.py` (external datasets:
   `analysis/preprocess_new/`), report cleaning via
   `preprocessing/clean_impressions.py` and `preprocessing/impression_section.py`,
   volumes to HDF5 via `preprocessing/run_preprocess.py` (RAS reorientation,
   HU clip [-1000, 1000], resize and pad to 160 x 224 x 224, uint8). CT-RATE
   validation carve: `preprocessing/split_validation.py` (seed 42, summarized
   in `preprocessing/manifests/ctrate_split_summary.json`).
2. **Train**: `model/run_scripts/train_release_v1.sh`; evaluate zero-shot with
   `model/run_test.py`.
3. **Observation bank**: `analysis/run_full.sh` drives
   `analysis/get_concepts_ct.py` (extraction prompts are inline in that
   script); embed with `analysis/get_embed_ct.py` (F2LLM is the paper's
   embedder).
4. **Representation and probing**: `analysis/run_pipeline.sh`, the zero-shot
   benchmark under `analysis/experiments/exp1_zeroshot/` (per-baseline faithful
   preprocessing; the f-VLM scorer includes the y-column fix), phenotype labels
   via `phenotypes/extract_inspect_per_ct_phenotypes.py` with
   `phenotypes/phecode_mapping.py` (Phecode Map v1.2; run with
   `cwd=phenotypes/`), probes via `analysis/run_v1_phenotype_lbfgs_f2llm.sh`
   and `analysis/experiments/exp4_confounder_audit/run_pheno_probe.sh`.
5. **Audit and restriction**: authoritative Equation 2 exports in
   `analysis/experiments/exp4_confounder_audit/` (`export_f2llm_adam_top25.py`,
   `build_f2llm_natural_clinical_audit.py`); restriction rules in
   `analysis/trusted_concept_space.py` and probes in
   `analysis/trusted_concept_probe.py`; evaluation and SI tables in
   `analysis/experiments/supplementary/` (patient-clustered bootstraps with
   shared draws).

### Figure map

Fig. 2b/2c: `analysis/experiments/exp1_zeroshot/analysis/plots/`
(`bar_plot_refstyle.py`, `circular_map.py` via `make_circular_maps.py`,
`stack_panels.py`). Fig. 2d: `analysis/experiments/exp3_concept_retrieval/`
plus `analysis/plot_concept_retrieval_grid.py`. Fig. 3:
`analysis/plot_concept_latent.py` (UMAP artifacts from
`analysis/concept_latent_umap.py`), `analysis/plot_radlex_embedding_distance.py`,
`analysis/plot_pmbb_combined_atlas.py`. Fig. 4:
`analysis/experiments/exp1_zeroshot/inspect_pheno/forest/plot_radar.py`.
Fig. 5: `analysis/experiments/exp4_confounder_audit/plot_proxy_family_grid.py`.
Fig. 6: `analysis/experiments/exp4_confounder_audit/plot_grid_2x5.py`.
SI figures: `plot_concept_retrieval_bars.py` (exp3), `plot_concept_wordclouds.py`,
`plot_shared_string_reuse.py`, `plot_clinically_aligned.py`,
`plot_refinement_paired_audits.py` (exp4). Figures 1 and the SI architecture
schematic are manual artwork with no script. The Figure 3 UMAP was fitted
without a fixed seed, so exact coordinates are not reproducible by design.

## Audit-score provenance

The probe-observation alignment score of Figures 5 and 6 (paper Equation 2) is
the dot product of the seed-averaged, **unnormalized** probe weight with the
bank-mean-centred, L2-normalized observation embedding, implemented in
`analysis/experiments/exp4_confounder_audit/export_f2llm_adam_top25.py` and
`build_f2llm_natural_clinical_audit.py`. A legacy variant
(`analysis/concept_audit_20seeds.py` and `cosine_align_audit` in
`analysis/exp_phenotype_ct.py`) normalizes the weight into a cosine and is
retained, clearly marked, only because its ranking selects the top-100 subset
for the `original_topk` baseline arm of Figure 6. No reported alignment number
comes from it. Before regenerating Figure 5 or 6, confirm the plotting script
reads the authoritative exports.

## Known gaps

A few dependencies live only on the training cluster and are being added:
`probe_features.py` (fairA probe head used by `analysis/perseed_probe.py`),
the `clip_3d_eval` loader (`eval_all.py`, `configs.json`; until it lands at
`analysis/eval/`, importing `clear3d.features` and therefore the four
`analysis/exp_*.py` drivers requires your own copy on `CLIP3D_EVAL_REPO`),
the PMBB bank builders, their extraction prompt, and the
`analysis/experiments/exp2_retrieval/pmbb_manifests/` builder (users with PMBB
access recreate the four per-scan manifest CSVs from their own PMBB export),
`clear3d/_gteqwen2_worker.py` (gte-Qwen2 baseline only),
`refine_probe_llmspace.py`, and `make_trusted_vs_baseline.py`.
The shipped trainer reproduces the released objective up to accumulation
semantics: the July 2025 run summed 32 independent per-GPU 4-sample InfoNCE
losses, while `run_train.py --accum_freq` uses OpenCLIP two-pass accumulation
over the full accumulated batch. `model/run_scripts/train_release_v1.sh`
documents the difference.
Not retained: the RSNA-2023 split generator (its seed-42 split is
deterministically reconstructed and verified by
`analysis/experiments/supplementary/reconstruct_rsna2023_test_manifest.py`),
the TeX composition files for Figures 3 and 4, the Figure 2d montage curation
run, and the exact run behind the 20,751 strict PMBB-only count in Figure 3c.
Result JSONs, concept banks, and probe bundles contain report-derived strings
or patient-level rows and are regenerated by users with dataset access.

## Licenses

Apache License 2.0 (`LICENSE`) for original code. Vendored components keep
their original licenses and per-file headers: OpenAI CLIP (MIT:
`model/model.py`, `model/clip.py`, `model/simple_tokenizer.py`, BPE vocab),
Meta V-JEPA 2 (MIT: `model/attentive_pooler.py`), CheXzero (MIT:
`model/eval.py` and the forest bootstrap CIs), OpenCLIP (MIT: `model/loss.py`).
`phenotypes/Phecode_map_v1_2_icd9_icd10cm.csv` and
`phenotypes/phecode_definitions1.2.csv` are PheWAS Catalog resources
([phewascatalog.org](https://phewascatalog.org/phecodes)), redistributed with
attribution. RadLex 4.3 is not redistributed: download it from
[radlex.org](https://radlex.org) under the RSNA license for
`analysis/plot_radlex_embedding_distance.py`. DINOv2 backbone weights are
fetched at runtime via `torch.hub`. Model weights and observation-bank
artefacts on Hugging Face are for academic research; the source datasets'
licenses and data-use terms apply.

## Citation

See `CITATION.cff`. A journal reference will be added upon publication.
