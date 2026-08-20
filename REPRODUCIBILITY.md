# Reproducibility

## Frozen protocol

- Exercises: squat and deadlift for the primary supported LOCO experiment; lunge is retained in the data audit but has no feasible target under the final support rule.
- Targets: 12 naturally observed diagnosis vectors listed in `configs/final_protocol.json`.
- Seeds: 7, 42, and 123.
- Split: remove every paired recording group containing the target diagnosis, then form a deterministic group-disjoint train/validation split.
- Support: each criterion state must have at least 8 training examples and at least 1 validation example.
- Input: paired frontal/lateral 33-joint 3D poses, interpolated for whole missing frames, pelvis-centered, robust-scale normalized, and resampled to 16 frames.
- Threshold: fixed at 0.5; no test-fold calibration.

The exact split indices are frozen in `configs/split_manifest_v2.json`.

## Reproduce reported statistics without the dataset

```bash
python src/aggregate_runs.py
python src/analyze_research_questions.py
python scripts/generate_figures.py
python scripts/verify_release.py
```

The RQ2 blocked permutation uses 50,000 randomizations and shuffles target-bit accuracy within each held-out diagnosis. This preserves criterion dependence inside diagnoses and avoids treating 67 criterion rows as independent.

## Retrain

After following `data/README.md`:

```bash
python src/run_context_evidence_sweep.py \
  --models tcn,gru,transformer,ssm,fact \
  --seeds 7,42,123 \
  --outdir results/reproduction
```

Training is intentionally a fixed-budget reproduction, not a hyperparameter sweep. The default FACT configuration uses AdamW, learning rate 0.0015, weight decay 0.0002, batch size 32, up to 55 epochs, and patience 9.

## Build the paper

From `paper/`, run `pdflatex main.tex`, `bibtex main`, then `pdflatex main.tex` twice. The checked-in PDF is the submission snapshot.

## Claim boundaries

This release tests recording-pair-disjoint natural error-composition shift on one dataset. It does not establish participant-disjoint generalization, causal reliance, universal FACT superiority, or external-dataset replication.

