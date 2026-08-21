# Contributing

Bug reports, leakage checks, independent reproductions, and additional datasets with concurrent criterion-level error labels are especially welcome. Please open an issue before a large change.

For code changes, run:

```bash
python scripts/verify_release.py
python src/analyze_research_questions.py --out build/rq_stats.json
```

Do not commit ALEX-GYM data, trained checkpoints, or credentials.

This repository is publicly hosted under its authors' account and is **not**
anonymized; its own URL identifies them. Author-blinding applies only to the
submission artifacts: `paper/Beyond_Seen_Mistakes.pdf` and the author fields of
`CITATION.cff`. Do not add author names, affiliations, or acknowledgements to
those two while the paper is under review.

