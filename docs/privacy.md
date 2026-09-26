# Privacy

Two independent controls exist and must never be conflated.

## `store_text` - retention

Set per prediction (checkbox or `store_text` JSON field), defaulting to
`STORE_TEXT_DEFAULT` (true).

- `true` - the raw analyzed text is stored with the analysis event
- `false` - only the SHA-256 hash, normalized hash, metadata and prediction are
  stored; the raw text never reaches the database

Retention is required for the duplicate check and for the human review queue, so
it defaults to on. It is an operational decision, not a consent decision.

```json
POST /api/check
{ "text": "...", "store_text": false }
```

When retention is off, feedback can still be submitted by sending `text`
explicitly; that text is stored with the feedback record for review and, if
consented, export.

## `allow_training` - consent

Set when the teacher submits feedback (`allow_training` /
`consent_for_training`). Feedback about a prediction is always allowed;
permission to reuse the raw text for training is not implied by leaving
feedback.

`src/privacy.py::can_export_for_training` requires **all** of:

- status `approved`
- `allow_training` true
- raw text still retained
- label in `{human, ai}`

## What is never done

- The production model is not retrained or modified by feedback.
- Pending, rejected or unconsented feedback is never exported.
- `ai_assisted` and `unsure` labels are never exported as training data.
- Raw text is not included in evaluation reports, monitoring metrics or gate
  reports; only counts and numbers are.
- `data/datasets/`, `data/feedback/` and `data/feedback.db` are gitignored so
  raw text is not committed.

## Admin access

`/admin/feedback` requires `ADMIN_TOKEN`:

- unset → `403` (the endpoint is disabled rather than silently open)
- wrong or missing header → `401` plus a login form
- correct `X-Admin-Token` header or logged-in cookie → queue access

Without a configured token there is no review queue, and therefore no path from
feedback to training data.
