# Contributing

Bug reports, leakage checks, independent reproductions, and additional datasets with concurrent criterion-level error labels are especially welcome. Please open an issue before a large change.

For code changes, run:

```bash
python scripts/verify_release.py
python src/analyze_research_questions.py --out build/rq_stats.json
```

Do not commit ALEX-GYM data, trained checkpoints, credentials, or identifying author metadata while the paper is under double-blind review.

