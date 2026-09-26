# Feedback system

Human feedback is the only bridge between a teacher's opinion and future model
improvements. It is collected on every analysis, reviewed by a human, and turned
into a versioned dataset. It never changes the production model by itself.

## Principles

1. Feedback is a **review queue entry**, not a training signal.
2. A human must set the status to `approved` before anything can be exported.
3. Training consent (`allow_training`) is a separate, affirmative decision.
4. The production model changes only through `tools/promote_model.py`.

## Labels and statuses

User labels:

| Label | Meaning |
| --- | --- |
| `human` | The teacher is confident the text is human-written |
| `ai` | The teacher is confident the text is AI-generated |
| `ai_assisted` | Written by a human with AI help (not a trainable class yet) |
| `unsure` | The teacher cannot tell |

Review statuses:

| Status | Meaning |
| --- | --- |
| `pending` | Default for new feedback; waiting for review |
| `approved` | Accepted by an admin; eligible for export when consented |
| `rejected` | Disagreed with or unusable |
| `needs_review` | Flagged for a second look |
| `duplicate` | Same text already present |
| `invalid` | Failed validation |

Optional confidence is `certain` or `not_sure`.

## Hard-case mining

Every submission is scored by `src/priority.py` so the queue shows the most
valuable mistakes first:

- `false_positive_candidate` — human text predicted as AI (most damaging to a teacher)
- `false_negative_candidate` — AI text predicted as human
- `high_disagreement` — prediction far from the user label
- `low_confidence` — probability close to 50
- `ai_assisted` — label that cannot be trained on but still needs a decision
- `edge_case` — fallback bucket, including `unsure`

`priority_score` is 0-100; higher means review first. `hard_case_categories`
stores every matching category as a JSON list so a record can appear in several
queues at once.

## Review queue

`GET /admin/feedback` renders the queue. It is protected:

- no `ADMIN_TOKEN` configured → `403`
- missing/wrong token → `401` plus a login form (`POST /admin/login` sets a cookie)
- `X-Admin-Token: <token>` header is also accepted (used by scripts and tests)

Filters: `all`, `pending`, `needs_review`, `approved`, `rejected`,
`high_disagreement`, `false_positives`, `false_negatives`, `ai_assisted`.

`POST /admin/feedback/<id>` with `{"status": "approved"}` records the decision,
the review timestamp, and returns the new status.

## Export

`python tools/export_feedback_dataset.py` writes only records that satisfy all
of:

- status `approved`
- `allow_training` true
- raw text still retained
- label in `{human, ai}`

Output goes to `data/feedback/approved/{human,ai}/*.txt` plus `metadata.csv`.
`ai_assisted` and `unsure` are reviewable but never exported.

## Contract

`POST /api/feedback`:

```json
{
  "analysis_id": "uuid",
  "label": "human",
  "allow_training": false,
  "user_confidence": "certain",
  "text": "optional, required when the analysis text was not retained",
  "prediction_incorrect": true
}
```

- `201` for a new pending record, `200` for a duplicate
- `400` for an invalid label, confidence, or consent type
- `404` for an unknown analysis id
