# Answer or Defer? Testing VLMs' Stated Confidence on Brain MRI

<p align="center">
  <img src="docs/logos/trends.png" alt="TReNDS Center" height="42">
  &nbsp;&nbsp;&nbsp;
  <img src="https://amir-sbg.github.io/FlexSplat/assets/logos/gsu.png" alt="Georgia State University" height="42">
</p>

<p align="center">
  <a href="https://wacv.thecvf.com/"><img src="https://img.shields.io/badge/WACV-2027-4b2e83.svg" alt="WACV 2027"></a>
  <a href="https://sites.google.com/view/fabric-wacv"><img src="https://img.shields.io/badge/FABRIC-Workshop-7c5cc4.svg" alt="FABRIC Workshop"></a>
  <a href="https://arxiv.org/abs/2611.11833"><img src="https://img.shields.io/badge/arXiv-2611.11833-b31b1b.svg" alt="arXiv 2611.11833"></a>
  <img src="https://img.shields.io/badge/Python-3.11-3776AB.svg" alt="Python 3.11">
</p>

Research code for the paper:

> **Answer or Defer? Testing VLMs' Stated Confidence on Brain MRI**<br>
> Amir Sabbaghziarani<sup>1,2</sup>, Parsa Ghazvinian<sup>2</sup>, Sepehr Salem<sup>2</sup>, and Sergey Plis<sup>1,2</sup><br>
> <sup>1</sup>Joint Georgia Institute of Technology / Georgia State University / Emory University Center for Translational Research in Data Science and Neuroimaging<br>
> <sup>2</sup>Georgia State University<br>
> Accepted to [WACV 2027](https://wacv.thecvf.com/) —
> [FABRIC Workshop](https://sites.google.com/view/fabric-wacv)

A vision-language model that reads a brain MRI slice also tells you how sure it
is. This repository asks whether that number can decide when to act on the
model's answer and when to refer the case to a clinician -- and, where it
cannot, what in the confidence signal makes it fail.

> This is research evaluation code. It is **not** a clinical diagnostic system
> and must not be used for patient-care decisions.

## Summary

Eight open-weight VLMs, two of them reasoning models, assess **3,888 brain MRI
slices** for a tumor or gross abnormality:

- **1,206 slices** from 150 post-treatment glioma volumes (BraTS 2024), with a
  mask-derived reference per slice
- **2,682 slices** from 100 healthy volunteers (IXI), treated as negatives
- Multiple choice with neutral wording, an explicit confidence from 0 to 100,
  greedy decoding

<p align="center">
  <img src="docs/study-design.png" alt="Answer-or-defer evaluation design for VLM confidence on brain MRI" width="900">
</p>

A gate accepts the answers whose confidence clears a threshold and defers the
rest. The question is whether any threshold leaves a set of answers reliable
enough to act on.

## Key Findings

- On glioma patients, **no threshold selects a nonempty set with at most 5%
  empirical error** for seven of the eight models, even chosen retrospectively.
- At a **10% budget**, Gemma-4-12B retains 24% of slices and every other model
  at most 2.5%.
- Confidence differs far more **between a model's two answers** than between
  its right and wrong answers to the same question: Gemma-4-12B states 0.969
  on its true positives and 0.970 on its false positives.
- A covariance decomposition (Lemma 1) splits the confidence-correctness
  association into a part tied to the answer and a part within each answer, and
  explains how the correlation **reverses between populations**.
- At a **20% budget**, a rule that ignores confidence entirely keeps more
  slices than the gate for two models.
- Mean confidence barely moves with lesion area, even where sensitivity nearly
  doubles.

<p align="center">
  <img src="docs/confidence-failure-example.png" alt="Example showing the same high stated confidence on correct and incorrect brain MRI answers" width="960">
</p>

## Confidence Failure Analysis

<p align="center">
  <img src="docs/outcome-level-confidence.png" alt="Mean stated confidence for true positives, false positives, true negatives, and false negatives" width="760">
</p>

True and false predictions receive nearly the same confidence once the answer
is fixed. The larger gap is between *yes* and *no* answers, indicating that
stated confidence often reflects answer preference more than correctness.

<p align="center">
  <img src="docs/population-correlation-shift.png" alt="Confidence-correctness correlation reverses between the BraTS and IXI populations" width="760">
</p>

The confidence-correctness correlation changes sign between the glioma and
healthy cohorts for all eight models. On IXI, where every reference answer is
negative, the sign mainly records which answer a model states more confidently;
it is not evidence that confidence generalizes as a reliability signal.

<p align="center">
  <img src="docs/lesion-area-sensitivity.png" alt="VLM sensitivity increases with lesion area on BraTS positive slices" width="760">
</p>

Sensitivity improves sharply as in-plane lesion area grows, but mean stated
confidence changes little across the same bins. The models therefore provide
limited confidence-based warning on the small lesions they are most likely to
miss.

## Benchmark

- **Data**: BraTS 2024 (post-treatment glioma) and IXI (healthy volunteers).
  Neither is redistributed here.
- **Reference**: derived per slice from the BraTS segmentation. A slice is
  positive if the mask covers at least 25 px on that slice; any annotated
  compartment counts -- enhancing tissue, non-enhancing core, surrounding FLAIR
  hyperintensity, resection cavity.
- **Models**: `InternVL2.5-8B`, `Qwen2.5-VL-3B`, `Qwen2.5-VL-7B`, `Gemma-3-4B`,
  `MedGemma-4B`, `Gemma-4-12B`, `GLM-4.6V-Flash`, `Qwen3.6-27B`.
- **Metrics**: sensitivity, specificity, balanced accuracy, predictive values,
  coverage and selective risk, C_alpha at 5 / 10 / 20% error budgets, the
  answer decomposition of Cov(c, z), case-mix reweighting under label shift,
  and subject-clustered bootstrap intervals.

## Headline Results

Recomputed by this code from the per-slice decisions, not copied from the
paper. `C_alpha` is the largest share of slices any threshold accepts with at
most `alpha` error; deferred slices count in the denominator.

| Model | Sens. BraTS+ | Spec. BraTS− | Spec. IXI | BA BraTS | Acc. | C<sub>0.05</sub> | C<sub>0.10</sub> | C<sub>0.20</sub> |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| InternVL2.5-8B | .423 | .615 | .957 | .519 | .481 | 0 | 0 | 0 |
| Qwen2.5-VL-3B | .198 | .872 | .707 | .535 | .402 | 0 | 0 | 0 |
| Qwen2.5-VL-7B | .042 | .984 | .837 | .513 | .328 | 0 | 0 | 0 |
| Gemma-3-4B | .619 | .473 | .274 | .546 | .575 | 0 | 0 | 0 |
| MedGemma-4B | .424 | .814 | .727 | .619 | .542 | 0 | 0 | 0 |
| Gemma-4-12B | .616 | .702 | .525 | .659 | .642 | 0 | **.243** | .480 |
| GLM-4.6V-Flash | .637 | .573 | .626 | .605 | .617 | 0 | .025 | .025 |
| Qwen3.6-27B † | .788 | .555 | .486 | .671 | .720 | .021 | .021 | .021 |

† The released transcripts for Qwen3.6-27B are a 2,048-token run that answers
65% of slices, not the 8,192-token run the paper describes. See
[docs/reproducing.md](docs/reproducing.md), which also records the two places
where the released data does not support a sentence in the abstract.

Three patterns run through the table:

- The models differ mostly in how readily they say "yes", and most trade
  sensitivity against specificity rather than improving both.
- A gate validated on healthy brains looks safest for the models that miss the
  most tumors: the two most specific models clear a 5% target on IXI without
  deferring anything, and are among the least sensitive on BraTS.
- Calibration does not decide this. C_alpha depends only on how confidence
  orders the slices, so any strictly increasing recalibration leaves it
  unchanged.

## Repository

- `config/`: datasets, models, prompts, tasks, and `study.yaml` -- every
  threshold, error budget, bin edge and bootstrap setting the analysis uses
- `src/data/`: slice extraction and the mask-derived per-slice reference
- `src/prompts/`: the prompt matrix and the logged option shuffle
- `src/inference/`: vLLM and Transformers runners, resumable and shardable
- `src/scoring/`: response parsing, including reasoning-trace handling
- `src/analysis/`: the study table, the gate, the decomposition, case mix, the
  restricted subset, the bootstrap, and the table / figure / verify drivers
- `src/figures/`: the paper's figures
- `slurm/`: cluster launchers
- `tests/`: 72 tests over parsing, the gate, the decomposition, case mix, the
  subset and the bootstrap

Nothing generated is tracked. The decision table, the tables, the figures and
the verification report are all rebuilt by `make paper`, and the images, the
segmentations and the raw transcripts are not redistributed here. Per-slice
outputs will be released with the camera-ready version.

## Quick Start

```bash
git clone https://github.com/amir-sbg/answer-or-defer-vlm-brain-mri.git
cd answer-or-defer-vlm-brain-mri

conda env create -f environment.yml
conda activate deferral
export PYTHONPATH="$PWD"
```

Run the test suite, which needs no data:

```bash
make test
```

Rebuilding the paper needs `results/tumor_decisions.csv.gz`, which `make study`
produces from the transcripts (see the next section). Once it exists:

```bash
make paper
```

That writes `results/tables/*.csv` (and LaTeX for the four numbered tables),
`figures/fig2..fig5`, and `results/verification.csv`, in about a minute and
without a GPU.

## Full Pipeline

This repository does not redistribute medical data. Place BraTS 2024 under
`data/brats2024/` and IXI under `data/ixi/`, then:

```bash
python -m src.data.slice_extract \
  --config config/datasets.yaml \
  --dataset brats ixi \
  --output-dir data/slices \
  --provenance-out data/slice_provenance.jsonl
```

Derive the per-slice reference and build the exam manifest:

```bash
python -m src.data.build_manifest \
  --provenance data/slice_provenance.jsonl \
  --config config/datasets.yaml \
  --study config/study.yaml \
  --output data/exam_set.csv
```

Run one model over the exam:

```bash
python -m src.inference.run_inference \
  --model google/gemma-4-12B-it \
  --manifest data/exam_set.csv \
  --output-dir raw_responses \
  --phrasing neutral \
  --format MC OE
```

Parse the transcripts once into the per-slice decision table -- one row per
(model, slice) with the reference, the answer, the correctness and the stated
confidence -- then rebuild the paper from it:

```bash
python -m src.analysis.build_study_table \
  --responses-dir raw_responses \
  --manifest data/exam_set.csv \
  --output results/tumor_decisions.csv.gz

make paper
```

Every number in Section 4 comes from that one file, so the transcripts are
parsed exactly once and each result is traceable to a single artifact.

For the full sweep, `sbatch slurm/inference.slurm` runs one model per array
task across four GPUs. Runs resume: each shard owns a disjoint, stably hashed
set of prompts and skips what it has already written, so a job killed by the
wall clock is restarted by resubmitting the same array.

## Reproducibility Notes

- **Deferral is explicit.** A response with no parseable option, an UNSURE, a
  reasoning trace cut off at the token limit, or an answer with no parseable
  confidence is deferred. Deferred slices count in the coverage denominator but
  are never accepted, so coverage measures how much of the slice task could be
  automated. Rates are computed on answered slices.
- **A threshold cannot split a tie.** Most models put hundreds of slices on one
  confidence value, so the threshold search tries exactly the distinct values a
  model used and takes a whole level or none of it.
- **C_alpha is chosen in sample.** A zero is strong evidence that nothing
  works; a positive value is the best coverage observed here, not a guarantee
  for new data.
- **The bootstrap clusters on the patient.** Nine slices come from each sampled
  volume and several volumes can come from one patient.
- **Decoding is greedy**; the option shuffle and the prompt sharding are seeded
  from the item identity with sha256, not Python's salted `hash()`.
- **`make verify`** recomputes every number the paper prints and reports the
  differences. [docs/reproducing.md](docs/reproducing.md) accounts for all of
  them.

## Scope and Responsible Use

This is a negative result about one signal on one binary decision, measured
against a mask-derived reference rather than a radiologist's read of each
displayed slice. It is not a verdict on any model and makes no diagnostic
claim. The main risk it carries is over-trust in the opposite direction: a
model's statement about its own reliability should be validated like any other
clinical signal, on the population where it will be used, before anyone acts
on it.

## Citation

```bibtex
@inproceedings{sabbaghziarani2027answer,
  title     = {Answer or Defer? Testing {VLMs}' Stated Confidence on Brain {MRI}},
  author    = {Sabbaghziarani, Amir and Ghazvinian, Parsa and Salem, Sepehr and Plis, Sergey},
  booktitle = {FABRIC Workshop at the IEEE/CVF Winter Conference on Applications of Computer Vision},
  year      = {2027},
  note      = {Accepted to the FABRIC Workshop}
}
```

## Acknowledgments

Developed in the TReNDS research environment at Georgia State University. The
study builds on public neuroimaging resources -- BraTS 2024 and IXI -- and
derives its reference from released segmentation masks rather than new manual
annotation. The exam protocol follows the behavioral safety audit in
[confident-unreliable-vlm-brain-mri-audit](https://github.com/amir-sbg/confident-unreliable-vlm-brain-mri-audit),
from which the six instruction-tuned models' transcripts are taken; the two
reasoning models were run for this study.
