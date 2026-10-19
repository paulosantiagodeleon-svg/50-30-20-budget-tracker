"""Business logic: 50/30/20 math, savings routing and debt payoff.

Pure functions (``calculate_split``, ``summarize_cycle`` ...) hold the math so
they are trivially testable; ``BudgetService`` orchestrates the repository and
commits every multi-document change as one atomic batch write.
"""
from __future__ import annotations

import calendar
from datetime import date, datetime, timedelta
from typing import Any, Optional

from .firebase_client import DEBTS, INCOME_CYCLES, SAVINGS_ACCOUNTS, TRANSACTIONS
from .models import (
    BUCKETS,
    INCOME_BUCKET,
    LINKED_BUCKETS,
    Debt,
    IncomeCycle,
    SavingsAccount,
    Transaction,
    ValidationError,
)

NEEDS_PCT, WANTS_PCT = 0.50, 0.30  # the remaining 20% is savings & debt repayment
FREQUENCIES = ("Monthly", "Bi-weekly")
SEED_PREFIX = "seed-"


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
        reset_ops += [("delete", TRANSACTIONS, tid, None) for tid in tx_ids]
        self.db.batch_write(reset_ops)  # balances back to their baseline, old seed tx removed

        for tid, (offset, amount, category, bucket, channel, target, notes) in zip(tx_ids, tx_plan):
            self.add_transaction(
                cycle_id=cycle_id, date=min(start + timedelta(days=offset), today),
                amount=amount, category=category, bucket=bucket, payment_channel=channel,
                target_account_id=target, notes=notes, tx_id=tid)
        return {"cycles": 1, "savings_accounts": len(accounts), "debts": 2, "transactions": len(tx_plan)}
