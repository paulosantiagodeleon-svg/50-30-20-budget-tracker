# 50/30/20 Budget, Debt & Savings Tracker (PHP ₱)

A Streamlit dashboard that splits your net pay into **50% Needs / 30% Wants / 20% Savings & Debt**, logs daily
transactions, routes savings into digital-bank buckets (GoTyme, MariBank, Maya, Emergency Fund) and tracks debt payoff.
Data lives in **Google Cloud Firestore**, with an automatic **local JSON mock** so it runs out of the box.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

On first launch there is no `serviceAccountKey.json`, so the app uses `local_mock_db.json` (git-ignored).
Click **🌱 Seed Sample PHP Data** in the sidebar to populate a sample cycle, savings accounts, debts and transactions.

## Connecting to real Firestore

1. In the [Firebase console](https://console.firebase.google.com/) create a project and enable **Firestore Database**.
2. Go to **Project settings → Service accounts → Generate new private key**.
3. Save the downloaded file as **`serviceAccountKey.json` in the project root** (next to `app.py`).
4. Restart the app. The sidebar badge turns 🟢 **Live Cloud Firestore**.

The key is covered by `.gitignore` (`*serviceAccountKey*.json`). **Never commit it.** If the key exists but the
connection fails (bad credentials, offline), the app falls back to the mock and shows the reason in the sidebar.

## Project layout

| Path | Purpose |
|---|---|
| `app.py` | Streamlit UI: sidebar navigation + Dashboard, Transactions, Savings, Debts, Developer pages |
| `src/firebase_client.py` | `LiveFirestore`, `LocalMockFirestore` (same method signatures), `get_database()`, query logger |
| `src/models.py` | Dataclasses that validate every document before it is written |
| `src/services.py` | 50/30/20 math, savings routing, debt deduction, seeding |
| `tests/test_budget_logic.py` | Pytest suite |

## Firestore collections

- `income_cycles`: `{ id, cycle_name, net_salary, needs_cap, wants_cap, savings_debt_cap, start_date, end_date, is_active }`
- `transactions`: `{ id, cycle_id, date, amount, category, bucket, payment_channel, target_account_id, notes, created_at }`
- `debts`: `{ id, name, total_amount, remaining_balance, minimum_payment, due_date, status }`
- `savings_accounts`: `{ id, bank_name, goal_name, current_balance, interest_rate }`

Dates are ISO strings (`YYYY-MM-DD`); money is stored as floats rounded to 2 decimals.

## How it behaves

- **Caps:** the 20% bucket absorbs rounding so the three caps always sum exactly to net salary. Savings and Debt
  transactions both count against the single 20% cap.
- **Savings routing:** a `Savings` transaction needs a target account and increments its `current_balance`.
- **Debt payoff:** a `Debt` transaction needs a target debt, deducts from `remaining_balance` and marks it `paid` at zero.
  A payment larger than the remaining balance is rejected.
- **Atomic writes:** the transaction document and the balance update are committed in one batch
  (`db.batch().commit()` on Firestore, all-or-nothing on the mock).
- **Deleting a transaction** reverses its balance effect.
- **Income bucket:** an extra `Income` bucket lets you log side income. It is shown separately and does not count
  against the caps.
- **Developer tab:** shows each Firestore query (`.collection().where()...`) with its JSON payload, plus a live view of
  every collection.
- **Seed data** uses ids prefixed `seed-`; re-seeding resets only those documents. It also makes the sample cycle active.

## Tests

```bash
pytest
```

## Notes

- Equality-only `where` filters are used so no Firestore composite indexes are needed; sorting happens client-side.
- Before real use, lock down your Firestore security rules. The Admin SDK bypasses rules, but never expose the key.
