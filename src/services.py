"""Business logic: 50/30/20 math, savings routing and debt payoff.

Pure functions (``calculate_split``, ``summarize_cycle`` ...) hold the math so
they are trivially testable; ``BudgetService`` orchestrates the repository and
commits every multi-document change as one atomic batch write.
"""
from __future__ import annotations

import calendar
import math
from datetime import date, datetime, timedelta, timezone
from typing import Any, Optional

from . import firebase_client as _fc
from .firebase_client import DEBTS, INCOME_CYCLES, SAVINGS_ACCOUNTS, TRANSACTIONS
from .models import (
    BUCKETS,
    INCOME_BUCKET,
    LINKED_BUCKETS,
    Debt,
    IncomeCycle,
    RecurringTemplate,
    SavingsAccount,
    Transaction,
    ValidationError,
)

NEEDS_PCT, WANTS_PCT = 0.50, 0.30  # the remaining 20% is savings & debt repayment
FREQUENCIES = ("Monthly", "Bi-weekly")
SEED_PREFIX = "seed-"
PH_FINAL_WITHHOLDING_TAX = 0.20  # 20% final withholding tax on Philippine bank deposit interest
WIPE_BATCH_SIZE = 400  # Firestore allows at most 500 writes per batch; stay safely under it
# st.session_state key prefixes owned by the UI forms; cleared when all data is wiped.
UI_STATE_PREFIXES = ("rec_", "topup_", "new_cycle_", "cycle_edit_", "wipe_")

# v1.1 collection. firebase_client.COLLECTIONS doubles as a whitelist inside LocalMockFirestore
# (_load / reset / snapshot), so without this guard the mock would silently drop the new
# collection on the next restart. Once COLLECTIONS in firebase_client.py includes it, this is a no-op.
RECURRING_TEMPLATES = getattr(_fc, "RECURRING_TEMPLATES", "recurring_templates")
if RECURRING_TEMPLATES not in _fc.COLLECTIONS:
    _fc.COLLECTIONS = tuple(_fc.COLLECTIONS) + (RECURRING_TEMPLATES,)
ALL_COLLECTIONS = tuple(_fc.COLLECTIONS)


# ---------------------------------------------------------------- pure math
def _to_date(value: Any) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value))
    except ValueError:
        raise ValidationError("Dates must be valid (YYYY-MM-DD).") from None


def calculate_split(net_salary: float) -> dict[str, float]:
    """Return the 50/30/20 caps. The 20% bucket absorbs rounding so caps sum to net."""
    try:
        net = round(float(net_salary), 2)
    except (TypeError, ValueError):
        raise ValidationError("Net salary must be a number.") from None
    if net <= 0:
        raise ValidationError("Net salary must be greater than 0.")
    needs = round(net * NEEDS_PCT, 2)
    wants = round(net * WANTS_PCT, 2)
    return {
        "needs_cap": needs,
        "wants_cap": wants,
        "savings_debt_cap": round(net - needs - wants, 2),
    }


def default_end_date(start: date, frequency: str = "Monthly") -> date:
    """Last day of a pay cycle that begins on ``start``."""
    if frequency == "Bi-weekly":
        return start + timedelta(days=13)
    if frequency != "Monthly":
        raise ValidationError(f"Frequency must be one of: {', '.join(FREQUENCIES)}.")
    year, month = (start.year + 1, 1) if start.month == 12 else (start.year, start.month + 1)
    next_start = date(year, month, min(start.day, calendar.monthrange(year, month)[1]))
    return next_start - timedelta(days=1)


def _line(spent: float, cap: float) -> dict[str, Any]:
    spent, cap = round(spent, 2), round(cap, 2)
    return {
        "spent": spent,
        "cap": cap,
        "remaining": round(cap - spent, 2),
        "pct": (spent / cap) if cap > 0 else 0.0,
        "over": spent > cap,
    }


def summarize_cycle(cycle: dict, transactions: list[dict]) -> dict[str, Any]:
    """Compare logged spending with the 50/30/20 caps for one pay cycle."""
    spent = {bucket: 0.0 for bucket in BUCKETS}
    other_income = 0.0
    for tx in transactions:
        if tx["bucket"] in spent:
            spent[tx["bucket"]] += tx["amount"]
        elif tx["bucket"] == INCOME_BUCKET:
            other_income += tx["amount"]
    spent = {k: round(v, 2) for k, v in spent.items()}
    total = round(sum(spent.values()), 2)
    return {
        "spent": spent,
        "lines": {
            "Needs": _line(spent["Needs"], cycle["needs_cap"]),
            "Wants": _line(spent["Wants"], cycle["wants_cap"]),
            "Savings & Debt": _line(spent["Savings"] + spent["Debt"], cycle["savings_debt_cap"]),
        },
        "total_allocated": total,
        "unallocated": round(cycle["net_salary"] - total, 2),
        "other_income": round(other_income, 2),
    }


def debt_progress(total_amount: float, remaining_balance: float) -> float:
    """Fraction of a debt already paid off (0.0 - 1.0)."""
    if total_amount <= 0:
        return 0.0
    return max(0.0, min(1.0, (total_amount - remaining_balance) / total_amount))


def _clamped_day(year: int, month: int, day: int) -> date:
    return date(year, month, min(day, calendar.monthrange(year, month)[1]))


def next_due_date(due_day: int, today: date) -> date:
    """Next date on or after ``today`` falling on ``due_day`` (short months use their last day)."""
    this_month = _clamped_day(today.year, today.month, due_day)
    if this_month >= today:
        return this_month
    year, month = (today.year + 1, 1) if today.month == 12 else (today.year, today.month + 1)
    return _clamped_day(year, month, due_day)


def calculate_interest_projection(balance: float, annual_rate_percent: float) -> dict[str, float]:
    """Project interest on a flat annual rate with the 20% PH final withholding tax applied.

    Simple (non-compounding) estimate; values are unrounded so net == gross * 0.80 exactly.
    Daily uses 365 days, monthly uses 12 months.
    """
    try:
        bal, rate = float(balance), float(annual_rate_percent)
    except (TypeError, ValueError):
        raise ValidationError("Balance and interest rate must be numbers.") from None
    if not (math.isfinite(bal) and math.isfinite(rate)):
        raise ValidationError("Balance and interest rate must be finite numbers.")
    if bal < 0:
        raise ValidationError("Balance cannot be negative.")
    if not 0 <= rate <= 100:
        raise ValidationError("Interest rate must be between 0 and 100 (%).")
    annual_gross = bal * (rate / 100)
    annual_net = annual_gross * (1 - PH_FINAL_WITHHOLDING_TAX)
    return {
        "daily_gross": annual_gross / 365,
        "monthly_gross": annual_gross / 12,
        "annual_gross": annual_gross,
        "daily_net": annual_net / 365,
        "monthly_net": annual_net / 12,
        "annual_net": annual_net,
    }


def portfolio_interest_projection(accounts: list[dict]) -> dict[str, float]:
    """Sum :func:`calculate_interest_projection` across savings accounts."""
    total = {k: 0.0 for k in calculate_interest_projection(0, 0)}
    for acct in accounts:
        proj = calculate_interest_projection(acct["current_balance"], acct.get("interest_rate", 0))
        for key, value in proj.items():
            total[key] += value
    return total


# ------------------------------------------------------------------ service
class BudgetService:
    def __init__(self, db) -> None:
        self.db = db

    # ---- pay cycles ---------------------------------------------------
    def _deactivate_ops(self, except_id: Optional[str] = None) -> list[tuple]:
        active = self.db.query(INCOME_CYCLES, [("is_active", "==", True)])
        return [("update", INCOME_CYCLES, c["id"], {"is_active": False})
                for c in active if c["id"] != except_id]

    def create_cycle(self, cycle_name: str, net_salary: float, start_date: Any,
                     end_date: Any = None, frequency: str = "Monthly",
                     activate: bool = True, cycle_id: Optional[str] = None) -> dict:
        """Create a cycle. ``end_date`` may be any date on/after ``start_date`` (irregular payroll
        cutoffs); ``frequency`` only supplies the default end date when ``end_date`` is omitted."""
        start = _to_date(start_date)
        end = _to_date(end_date) if end_date else default_end_date(start, frequency)
        cycle = IncomeCycle.new(
            id=cycle_id, cycle_name=cycle_name, net_salary=net_salary,
            start_date=start.isoformat(), end_date=end.isoformat(),
            is_active=activate, **calculate_split(net_salary),
        )
        ops = self._deactivate_ops(except_id=cycle.id) if activate else []
        ops.append(("set", INCOME_CYCLES, cycle.id, cycle.to_dict()))
        self.db.batch_write(ops)
        return cycle.to_dict()

    def get_active_cycle(self) -> Optional[dict]:
        active = self.db.query(INCOME_CYCLES, [("is_active", "==", True)])
        return max(active, key=lambda c: c["start_date"]) if active else None

    def list_cycles(self) -> list[dict]:
        return sorted(self.db.query(INCOME_CYCLES), key=lambda c: c["start_date"], reverse=True)

    def activate_cycle(self, cycle_id: str) -> None:
        if self.db.get(INCOME_CYCLES, cycle_id) is None:
            raise ValidationError("Pay cycle not found.")
        ops = self._deactivate_ops(except_id=cycle_id)
        ops.append(("update", INCOME_CYCLES, cycle_id, {"is_active": True}))
        self.db.batch_write(ops)

    def cycle_summary(self, cycle: dict) -> dict:
        return summarize_cycle(cycle, self.list_transactions(cycle_id=cycle["id"]))

    # ---- savings accounts ---------------------------------------------
    def add_savings_account(self, bank_name: str, goal_name: str = "",
                            current_balance: float = 0.0, interest_rate: float = 0.0,
                            account_id: Optional[str] = None) -> dict:
        account = SavingsAccount.new(id=account_id, bank_name=bank_name, goal_name=goal_name,
                                     current_balance=current_balance, interest_rate=interest_rate)
        self.db.set(SAVINGS_ACCOUNTS, account.id, account.to_dict())
        return account.to_dict()

    def list_savings_accounts(self) -> list[dict]:
        return sorted(self.db.query(SAVINGS_ACCOUNTS),
                      key=lambda a: (a["bank_name"].lower(), a.get("goal_name", "").lower()))

    # ---- debts --------------------------------------------------------
    def add_debt(self, name: str, total_amount: float, minimum_payment: float, due_date: Any,
                 remaining_balance: Optional[float] = None, debt_id: Optional[str] = None) -> dict:
        debt = Debt.new(
            id=debt_id, name=name, total_amount=total_amount,
            remaining_balance=total_amount if remaining_balance is None else remaining_balance,
            minimum_payment=minimum_payment, due_date=_to_date(due_date).isoformat(),
        )
        self.db.set(DEBTS, debt.id, debt.to_dict())
        return debt.to_dict()

    def list_debts(self) -> list[dict]:
        return sorted(self.db.query(DEBTS), key=lambda d: (d["status"] == "paid", d["due_date"]))

    # ---- transactions -------------------------------------------------
    def _require(self, collection: str, doc_id: Optional[str], label: str) -> dict:
        if not doc_id:
            raise ValidationError(f"Select a {label} for this transaction.")
        doc = self.db.get(collection, doc_id)
        if doc is None:
            raise ValidationError(f"The selected {label} no longer exists.")
        return doc

    def add_transaction(self, cycle_id: str, date: Any, amount: float, category: str,
                        bucket: str, payment_channel: str,
                        target_account_id: Optional[str] = None, notes: str = "",
                        tx_id: Optional[str] = None) -> dict:
        """Log a transaction and route its effect on savings/debt atomically."""
        if self.db.get(INCOME_CYCLES, cycle_id) is None:
            raise ValidationError("Pay cycle not found.")
        if tx_id and self.db.get(TRANSACTIONS, tx_id) is not None:
            raise ValidationError(f"Transaction {tx_id} already exists.")
        tx = Transaction.new(
            id=tx_id, cycle_id=cycle_id, date=date, amount=amount, category=category,
            bucket=bucket, payment_channel=payment_channel, notes=notes,
            target_account_id=target_account_id if bucket in LINKED_BUCKETS else None,
        )
        ops: list[tuple] = [("set", TRANSACTIONS, tx.id, tx.to_dict())]

        if tx.bucket == "Savings":
            account = self._require(SAVINGS_ACCOUNTS, tx.target_account_id, "savings account")
            ops.append(("update", SAVINGS_ACCOUNTS, account["id"],
                        {"current_balance": round(account["current_balance"] + tx.amount, 2)}))
        elif tx.bucket == "Debt":
            debt = self._require(DEBTS, tx.target_account_id, "debt")
            if tx.amount > debt["remaining_balance"] + 1e-9:
                raise ValidationError(
                    f"Payment exceeds the remaining balance of {debt['name']} "
                    f"(₱{debt['remaining_balance']:,.2f}).")
            remaining = round(debt["remaining_balance"] - tx.amount, 2)
            ops.append(("update", DEBTS, debt["id"], {
                "remaining_balance": remaining,
                "status": "paid" if remaining <= 0 else "active"}))

        self.db.batch_write(ops)
        return tx.to_dict()

    def delete_transaction(self, tx_id: str) -> None:
        """Delete a transaction and reverse its savings/debt balance effect."""
        tx = self.db.get(TRANSACTIONS, tx_id)
        if tx is None:
            raise ValidationError("Transaction not found.")
        ops: list[tuple] = [("delete", TRANSACTIONS, tx_id, None)]
        target = tx.get("target_account_id")

        if tx["bucket"] == "Savings" and target:
            account = self.db.get(SAVINGS_ACCOUNTS, target)
            if account:
                ops.append(("update", SAVINGS_ACCOUNTS, target, {
                    "current_balance": max(0.0, round(account["current_balance"] - tx["amount"], 2))}))
        elif tx["bucket"] == "Debt" and target:
            debt = self.db.get(DEBTS, target)
            if debt:
                remaining = min(debt["total_amount"], round(debt["remaining_balance"] + tx["amount"], 2))
                ops.append(("update", DEBTS, target, {
                    "remaining_balance": remaining,
                    "status": "paid" if remaining <= 0 else "active"}))
        self.db.batch_write(ops)

    def list_transactions(self, cycle_id: Optional[str] = None, bucket: Optional[str] = None,
                          payment_channel: Optional[str] = None) -> list[dict]:
        filters = []
        if cycle_id:
            filters.append(("cycle_id", "==", cycle_id))
        if bucket:
            filters.append(("bucket", "==", bucket))
        if payment_channel:
            filters.append(("payment_channel", "==", payment_channel))
        # Equality-only filters need no composite index; sort client-side.
        return sorted(self.db.query(TRANSACTIONS, filters),
                      key=lambda t: (t["date"], t["created_at"]), reverse=True)

    # ---- v1.1: salary top-ups & recurring bills (module-level functions below) ----
    def top_up_cycle_salary(self, cycle_id: str, additional_amount: float, note: str = "") -> dict:
        return top_up_cycle_salary(self.db, cycle_id, additional_amount, note)

    def create_recurring_template(self, *args: Any, **kwargs: Any) -> dict:
        return create_recurring_template(self.db, *args, **kwargs)

    def list_recurring_templates(self) -> list[dict]:
        return list_recurring_templates(self.db)

    def delete_recurring_template(self, template_id: str) -> None:
        delete_recurring_template(self.db, template_id)

    def post_recurring_templates(self, cycle_id: str, template_ids: list[str], date_str: Any) -> dict:
        return post_recurring_templates(self.db, cycle_id, template_ids, date_str)

    # ---- v1.1.1: cycle dates & wipe ----------------------------------
    def update_cycle_dates(self, cycle_id: str, start_date: Any, end_date: Any) -> dict:
        return update_cycle_dates(self.db, cycle_id, start_date, end_date)

    def wipe_all_data(self) -> dict[str, int]:
        return wipe_all_data(self.db)

    # ---- sample data --------------------------------------------------
    def seed_sample_data(self, today: Optional[date] = None) -> dict[str, int]:
        """Insert (or reset) a fictional PHP dataset. Idempotent: ids start with 'seed-'."""
        today = today or date.today()
        start = today.replace(day=1)
        cycle_id = f"{SEED_PREFIX}cycle"
        tx_plan = [  # (day offset, amount, category, bucket, channel, target id, notes)
            (0, 9000, "Rent", "Needs", "Maya", None, "Monthly rent"),
            (1, 2450, "Groceries", "Needs", "Cash", None, "Weekly market run"),
            (3, 3200, "Electricity & Water", "Needs", "Maya", None, ""),
            (4, 1350, "Transportation", "Needs", "Cash", None, "Commute / fare"),
            (6, 1299, "Internet", "Needs", "Credit Card", None, ""),
            (2, 1850, "Dining out", "Wants", "Credit Card", None, "Dinner with friends"),
            (5, 549, "Streaming", "Wants", "Credit Card", None, ""),
            (7, 780, "Coffee", "Wants", "Cash", None, ""),
            (9, 3200, "Shopping", "Wants", "Maya", None, "Sale"),
            (1, 2000, "Monthly savings", "Savings", "Maya", f"{SEED_PREFIX}gotyme", "Go Save top-up"),
            (1, 1000, "Monthly savings", "Savings", "Maya", f"{SEED_PREFIX}maribank", ""),
            (1, 500, "Emergency fund", "Savings", "Cash", f"{SEED_PREFIX}emergency", ""),
            (8, 1500, "Credit card payment", "Debt", "GoTyme", f"{SEED_PREFIX}card", ""),
            (8, 2000, "Salary loan payment", "Debt", "GoTyme", f"{SEED_PREFIX}loan", ""),
            (10, 2500, "Freelance gig", "Income", "GoTyme", None, "Extra income"),
        ]
        tx_ids = [f"{SEED_PREFIX}tx-{i:02d}" for i in range(len(tx_plan))]

        reset_ops = self._deactivate_ops(except_id=cycle_id)
        reset_ops.append(("set", INCOME_CYCLES, cycle_id, IncomeCycle.new(
            id=cycle_id, cycle_name=f"{start:%B %Y} (Sample)", net_salary=40000,
            start_date=start.isoformat(), end_date=default_end_date(start).isoformat(),
            is_active=True, **calculate_split(40000)).to_dict()))
        accounts = [("gotyme", "GoTyme", "Go Save", 12000, 4.0),
                    ("maribank", "MariBank", "Travel Fund", 8000, 3.0),
                    ("maya", "Maya", "Maya Savings", 5000, 3.5),
                    ("emergency", "Emergency Fund", "6-month cushion", 25000, 0.0)]
        for key, bank, goal, balance, rate in accounts:
            reset_ops.append(("set", SAVINGS_ACCOUNTS, f"{SEED_PREFIX}{key}", SavingsAccount.new(
                id=f"{SEED_PREFIX}{key}", bank_name=bank, goal_name=goal,
                current_balance=balance, interest_rate=rate).to_dict()))
        for key, name, total, remaining, minimum, due_day in [
                ("card", "Credit Card", 25000, 18500, 1500, 25),
                ("loan", "Salary Loan", 30000, 22000, 2500, 30)]:
            due = start.replace(day=min(due_day, calendar.monthrange(start.year, start.month)[1]))
            reset_ops.append(("set", DEBTS, f"{SEED_PREFIX}{key}", Debt.new(
                id=f"{SEED_PREFIX}{key}", name=name, total_amount=total,
                remaining_balance=remaining, minimum_payment=minimum,
                due_date=due.isoformat()).to_dict()))
        bills = [  # (name, amount, category, bucket, channel, target id, due day)
            ("Meralco Electricity", 3200, "Electricity", "Needs", "Maya", None, 15),
            ("Fiber Internet", 1299, "Internet", "Needs", "Credit Card", None, 6),
            ("GoTyme Auto-Save", 1000, "Auto-save", "Savings", "GoTyme", f"{SEED_PREFIX}gotyme", 30)]
        for n, (name, amount, category, bucket, channel, target, due_day) in enumerate(bills, start=1):
            reset_ops.append(("set", RECURRING_TEMPLATES, f"{SEED_PREFIX}rec-{n}", RecurringTemplate.new(
                id=f"{SEED_PREFIX}rec-{n}", name=name, amount=amount, category=category, bucket=bucket,
                payment_channel=channel, target_account_id=target, due_day=due_day).to_dict()))
        reset_ops += [("delete", TRANSACTIONS, tid, None) for tid in tx_ids]
        self.db.batch_write(reset_ops)  # balances back to their baseline, old seed tx removed

        for tid, (offset, amount, category, bucket, channel, target, notes) in zip(tx_ids, tx_plan):
            self.add_transaction(
                cycle_id=cycle_id, date=min(start + timedelta(days=offset), today),
                amount=amount, category=category, bucket=bucket, payment_channel=channel,
                target_account_id=target, notes=notes, tx_id=tid)
        return {"cycles": 1, "savings_accounts": len(accounts), "debts": 2, "transactions": len(tx_plan),
                "recurring_templates": len(bills)}


# ------------------------------------------------- v1.1 module-level functions
def log_transaction(db, cycle_id: str, date: Any, amount: float, category: str, bucket: str,
                    payment_channel: str, target_account_id: Optional[str] = None,
                    notes: str = "", tx_id: Optional[str] = None) -> dict:
    """Log one transaction (same atomic savings/debt routing as ``BudgetService.add_transaction``)."""
    return BudgetService(db).add_transaction(
        cycle_id, date, amount, category, bucket, payment_channel,
        target_account_id=target_account_id, notes=notes, tx_id=tx_id)


def top_up_cycle_salary(db, cycle_id: str, additional_amount: float, note: str = "") -> dict:
    """Add a mid-cycle paycheck (e.g. the 15th/30th) and re-split the new total 50/30/20.

    The 20% bucket absorbs rounding, so the caps always sum exactly to the new net salary.
    Returns the updated cycle document.
    """
    try:
        amount = round(float(additional_amount), 2)
    except (TypeError, ValueError):
        raise ValidationError("Top-up amount must be a number.") from None
    if not math.isfinite(amount) or amount <= 0:
        raise ValidationError("Top-up amount must be greater than 0.")
    cycle = db.get(INCOME_CYCLES, cycle_id)
    if cycle is None:
        raise ValidationError("Pay cycle not found.")
    new_net = round(cycle["net_salary"] + amount, 2)
    updates = {
        **calculate_split(new_net),
        "net_salary": new_net,
        "top_ups": [*(cycle.get("top_ups") or []), {
            "amount": amount,
            "note": str(note or "").strip(),
            "net_salary_after": new_net,
            "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }],
    }
    db.update(INCOME_CYCLES, cycle_id, updates)
    return {**cycle, **updates}


def create_recurring_template(db, name: str, amount: float, category: str, bucket: str,
                              payment_channel: str, due_day: int,
                              target_account_id: Optional[str] = None,
                              template_id: Optional[str] = None) -> dict:
    template = RecurringTemplate.new(
        id=template_id, name=name, amount=amount, category=category, bucket=bucket,
        payment_channel=payment_channel, target_account_id=target_account_id, due_day=due_day)
    if template_id and db.get(RECURRING_TEMPLATES, template.id) is not None:
        raise ValidationError(f"Recurring template {template.id} already exists.")
    if template.bucket == "Savings" and db.get(SAVINGS_ACCOUNTS, template.target_account_id) is None:
        raise ValidationError("The selected savings account no longer exists.")
    if template.bucket == "Debt" and db.get(DEBTS, template.target_account_id) is None:
        raise ValidationError("The selected debt no longer exists.")
    db.set(RECURRING_TEMPLATES, template.id, template.to_dict())
    return template.to_dict()


def list_recurring_templates(db) -> list[dict]:
    return sorted(db.query(RECURRING_TEMPLATES), key=lambda t: (t["due_day"], t["name"].lower()))


def delete_recurring_template(db, template_id: str) -> None:
    if db.get(RECURRING_TEMPLATES, template_id) is None:
        raise ValidationError("Recurring template not found.")
    db.delete(RECURRING_TEMPLATES, template_id)


def post_recurring_templates(db, cycle_id: str, template_ids: list[str], date_str: Any) -> dict:
    """Post the selected templates as transactions dated ``date_str`` in ``cycle_id``.

    Each template goes through :func:`log_transaction`, so savings/debt balances and cap tracking
    behave exactly like manual entries. Unknown ids or a bad date/cycle raise before anything is
    posted. A template that fails on its own (e.g. a debt payment above the remaining balance) is
    reported in ``failed`` and the rest are still posted.
    """
    posting_date = _to_date(date_str).isoformat()
    if db.get(INCOME_CYCLES, cycle_id) is None:
        raise ValidationError("Pay cycle not found.")
    ordered = list(dict.fromkeys(template_ids))  # de-duplicate so nothing posts twice
    templates = {tid: db.get(RECURRING_TEMPLATES, tid) for tid in ordered}
    missing = [tid for tid, doc in templates.items() if doc is None]
    if missing:
        raise ValidationError(f"Unknown recurring template(s): {', '.join(missing)}.")
    posted: list[dict] = []
    failed: list[dict] = []
    for tid in ordered:
        tpl = templates[tid]
        try:
            posted.append(log_transaction(
                db, cycle_id, posting_date, tpl["amount"], tpl["category"], tpl["bucket"],
                tpl["payment_channel"], target_account_id=tpl.get("target_account_id"),
                notes=f"Recurring: {tpl['name']}"))
        except ValidationError as exc:
            failed.append({"template_id": tid, "name": tpl["name"], "error": str(exc)})
    return {"posted": posted, "failed": failed}


# ------------------------------------------------- v1.1.1 module-level functions
def update_cycle_dates(db, cycle_id: str, start_date: Any, end_date: Any) -> dict:
    """Change a cycle's start/end dates (any range with end >= start). Returns the updated cycle.

    Transactions belong to a cycle by id, not by date, so existing entries are unaffected.
    """
    cycle = db.get(INCOME_CYCLES, cycle_id)
    if cycle is None:
        raise ValidationError("Pay cycle not found.")
    updates = {"start_date": _to_date(start_date).isoformat(), "end_date": _to_date(end_date).isoformat()}
    IncomeCycle.from_dict({**cycle, **updates})  # raises ValidationError if end < start
    db.update(INCOME_CYCLES, cycle_id, updates)
    return {**cycle, **updates}


def wipe_all_data(db) -> dict[str, int]:
    """Delete every document in every collection and verify the database is empty.

    Works through the repository API only, so it is identical for ``LiveFirestore`` and
    ``LocalMockFirestore``: the mock is cleared with its own ``reset()``; any other backend is
    emptied with batched deletes (``WIPE_BATCH_SIZE`` per batch). Returns ``{collection: deleted}``.
    Batches are atomic individually, not across the whole wipe, so after a mid-way failure simply
    run it again. Raises ``RuntimeError`` if anything is left afterwards.
    """
    before = db.snapshot()
    counts = {name: len(before.get(name, [])) for name in ALL_COLLECTIONS}
    if getattr(db, "supports_reset", False):
        db.reset()
    else:
        ops = [("delete", name, doc["id"], None)
               for name in ALL_COLLECTIONS for doc in before.get(name, []) if doc.get("id")]
        for start in range(0, len(ops), WIPE_BATCH_SIZE):
            db.batch_write(ops[start:start + WIPE_BATCH_SIZE])
    remaining = sum(len(docs) for docs in db.snapshot().values())
    if remaining:
        raise RuntimeError(f"Wipe incomplete: {remaining} document(s) are still in the database.")
    return counts


def clear_ui_state(state: Any, prefixes: tuple = UI_STATE_PREFIXES) -> list[str]:
    """Remove form-related keys from a ``st.session_state``-like mapping; returns the removed keys."""
    doomed = [key for key in list(state.keys()) if isinstance(key, str) and key.startswith(prefixes)]
    for key in doomed:
        del state[key]
    return doomed
