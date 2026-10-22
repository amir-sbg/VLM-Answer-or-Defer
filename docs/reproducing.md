# Reproducing the paper

Everything in Section 4 is rebuilt from one file,
`results/tumor_decisions.csv.gz`: one row per (model, slice) with the
reference, the answer, whether the answer is right, and the stated confidence.
Once that file exists, neither the images nor the 650 MB of raw transcripts are
needed again.

```bash
make study     # transcripts -> the decision table
make paper     # tables, figures, and the verification report
make test      # the test suite
```

No generated file is tracked in this repository. To produce the decision table
you need the slices and the transcripts: `make data` and `make inference` are
the steps that build them, and `slurm/inference.slurm` runs the eight models
four shards to a GPU.

## What `make verify` does

`docs/reported_values.yaml` holds every number the submitted paper prints,
transcribed by hand from its tables and figures. `src/analysis/verify.py`
recomputes each one from the released decisions and writes
`results/verification.csv`, one row per comparison. It exits non-zero on a
disagreement unless `--allow-mismatch` is passed.

On the transcripts released here, **115 of 174 reported values reproduce**:

| block | reproduces |
|---|---|
| Table 1, strata | 4 / 4 |
| Table 2, rates by stratum | 28 / 32 |
| Table 3, the gate on BraTS | 27 / 32 |
| Table 4, gate against the answer-only rule | 35 / 40 |
| Table 5, the restricted subset | 9 / 30 |
| Figure 3a, outcome-level confidence | 8 / 16 |
| Figure 3b, correlation by population | 4 / 16 |
| prose claims | 0 / 4 |

The rest of this file accounts for the differences. They are not rounding, and
two of them contradict sentences in the abstract.

## 1. Qwen3.6-27B was run at a shorter token budget than the paper describes

The paper states that Qwen3.6-27B generated up to 8,192 tokens so its reasoning
trace could finish, and reports 99%-plus answer coverage for both reasoning
models. The transcripts released here answer **785 of 1,206 BraTS slices
(65%)**, which matches the paper's own remark that at 2,048 tokens the model
answered only 66.8%. A trace that runs out of tokens never reaches an answer
and is counted as deferred, so these are the 2,048-token run, not the one
behind the paper's main numbers.

Every Qwen3.6-27B row therefore differs, including its sensitivity (0.788 here
against 0.690 reported) and its accuracy (0.720 against 0.663).
`config/models.yaml` carries the budget the paper describes, so re-running that
model reproduces its intended configuration; the transcripts have not been
re-generated here.

This matters beyond one row, because **Qwen3.6-27B reaches `C_0.05 = 0.021` on
these transcripts** -- 25 slices accepted at confidence 0.98 with 4% error.
That contradicts the abstract's "no threshold selects a nonempty set with at
most 5% empirical error for any model". The claim should be re-checked against
the 8,192-token run before the sentence is repeated.

## 2. GLM-4.6V-Flash does not flip the sign of its correlation

The abstract and Figure 3b say the confidence-correctness correlation reverses
between the two populations for all eight models. Seven do. GLM-4.6V-Flash has
**rho = +0.086 on BraTS and +0.198 on IXI** -- the same sign on both. The paper
plots +0.16 and -0.41.

Its rates are not in question: sensitivity, both specificities and balanced
accuracy all reproduce to three decimals, so the answers in these transcripts
are the answers behind Table 2. Only the confidences differ, and with them the
outcome-level means (0.809 / 0.783 / 0.830 / 0.780 here against 0.821 / 0.819 /
0.789 / 0.791 reported).

The "all eight models" phrasing should either be re-derived from whichever
confidences Figure 3b was drawn from, or weakened to the seven models that do
flip.

## 3. The restricted subset is 616 slices, not 620

Section 4.6 keeps BraTS+ slices with at least 400 px of annotated lesion on T2,
FLAIR or T1ce, and BraTS- slices with no mask voxel within 10 mm, and reports
480 positive and 140 negative. Applying that rule here gives **466 positive and
150 negative**.

Two details the prose leaves open:

- The sequence restriction is attached to the positive clause. Applying it to
  the negatives as well gives 150; leaving the negatives unrestricted gives
  199. `restrict_negatives_to_sequences` in `config/study.yaml` selects between
  them, and the default is the former, which is closest to the reported sizes.
- "No mask voxel within 10 mm" is implemented as the out-of-plane distance from
  the slice to the nearest annotated voxel. For a whole slice plane that *is*
  the minimum distance: the in-plane position of the nearest voxel already lies
  in the plane, so no in-plane term survives the minimum.

Because the subset differs, every per-model row in Table 5 differs with it.
Two findings in that table do not survive: **GLM-4.6V-Flash reaches
`C_0.05 = 0.037` on the subset** and Qwen3.6-27B reaches 0.023, against the
paper's statement that the threshold search keeps nothing at a 5% target for
all four models.

## 4. Qwen2.5-VL-7B keeps nothing at a 20% budget

Table 3 gives this model `C_0.20 = 0.034` (41 slices), the same value as its
answer-only rule in Table 4. On these transcripts no threshold reaches it: the
model's 41 "yes" answers carry confidences from 0.50 to 0.80, and 0.70 is also
the confidence of more than a thousand "no" answers, so no cut isolates them.
The operating points are 4 slices at 0.80 (25% error) and then 1,124 at 0.70
(67% error).

A tie-aware search cannot do better than 0, and the paper states that its
search is tie-aware. The reported value looks like the answer-only coverage
having been carried into the gate column.

## 5. Figure 3b's correlations differ by more than rounding

Beyond GLM, the correlations for the instruction-tuned models differ from the
plotted values by up to 0.19 (MedGemma-4B, +0.09 reported against -0.097 here;
Qwen2.5-VL-3B, +0.23 against +0.098) even though every one of those models
reproduces its Table 2 rates exactly. Identical answers with different
correlations means the confidences the figure was drawn from are not the ones
in these transcripts. The two near-zero cases (MedGemma-4B, Qwen2.5-VL-7B) are
sign-unstable in any case; the paper already says so.

## 6. Smaller things, stated for completeness

- **Gemma-4-12B's accepted sets** are 293 and 579 slices here against 294 and
  582 reported, at the same thresholds. One to three slices; the direction is
  towards slightly less coverage.
- **The excluded slices.** Section 3.1 reports excluding 74 BraTS slices with
  no usable mask and 70 whose mask covers 1-24 px. The data has 144 with no
  usable mask and 27 in the 1-24 px band. The two reported figures sum to 144,
  and Table 1's BraTS- count of 366 is reproduced only by treating the 27
  sub-threshold slices as negatives, which is what the pipeline does. The
  prose, which calls them "too little to call either way", does not match the
  arithmetic of its own table.
- **"150 subjects"** in Table 1 counts sampled volumes: one sequence file each,
  drawn from 138 distinct patients. The manifest keeps `volume_id` for the
  sampling unit and `subject_id` for the patient, and the bootstrap clusters on
  the patient.
- **The positivity-threshold check.** Raising the threshold from 25 to 100 px
  moves sensitivity by at most 0.026 and specificity by at most 0.011 across
  all eight models, against "at most 0.021" and "at most 0.005" for the four
  the paper checked. The conclusion holds; the bounds are over a larger set.
- **The smallest lesion bin.** Sensitivity there spans 0.033 to 0.450, printed
  as 0.03-0.46.

## Determinism

Decoding is greedy, so inference is deterministic given a model, a prompt and a
token budget. The option shuffle is seeded from the item identity with sha256,
not Python's `hash()`, which is salted per process. Sharding hashes the same
way, so a resumed or re-sharded run lands every prompt in the file it was in
before. The bootstrap takes its seed from `config/study.yaml`.

What is *not* pinned is the model weights: a Hugging Face repository can move
under its name. The runs here were made with the revisions current in June to
August 2026.
