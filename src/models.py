"""Strict document schemas for the four Firestore collections.

Every document is validated *before* it is pushed to Firestore (or the local
mock), so malformed data can never reach the database. Dates are stored as ISO
strings (YYYY-MM-DD) and money as floats rounded to 2 decimals (PHP ₱).
"""
from __future__ import annotations

import math
import uuid
from dataclasses import asdict, dataclass, fields
from datetime import date, datetime, timezone
from typing import Any, Optional

BUCKETS = ("Needs", "Wants", "Savings", "Debt")
INCOME_BUCKET = "Income"  # logged income; never counts against the 50/30/20 caps
LOG_BUCKETS = BUCKETS + (INCOME_BUCKET,)
PAYMENT_CHANNELS = ("Cash", "Maya", "GoTyme", "MariBank", "Credit Card", "Other")
LINKED_BUCKETS = ("Savings", "Debt")  # buckets that must point at an account/debt


class ValidationError(ValueError):
    """Raised when a document or request fails validation."""


def new_id() -> str:
    return uuid.uuid4().hex[:20]


def _money(value: Any, name: str, *, positive: bool = False) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValidationError(f"{name} must be a number.") from None
    if not math.isfinite(number):
        raise ValidationError(f"{name} must be a finite number.")
    if positive and number <= 0:
        raise ValidationError(f"{name} must be greater than 0.")
    if number < 0:
        raise ValidationError(f"{name} cannot be negative.")
    return round(number, 2)


def _iso_date(value: Any, name: str) -> str:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    try:
        return date.fromisoformat(str(value)).isoformat()
    except ValueError:
        raise ValidationError(f"{name} must be a valid date (YYYY-MM-DD).") from None


def _text(value: Any, name: str, *, required: bool = True) -> str:
    text = "" if value is None else str(value).strip()
    if required and not text:
        raise ValidationError(f"{name} is required.")
    return text


class _Document:
    """Shared helpers for all document dataclasses."""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)  # type: ignore[call-overload]

    @classmethod
    def from_dict(cls, data: dict[str, Any]):
        names = {f.name for f in fields(cls)}  # type: ignore[arg-type]
        return cls(**{k: v for k, v in data.items() if k in names})

    @classmethod
    def new(cls, **kwargs: Any):
        """Create a document with a freshly generated id (or the one supplied)."""
        doc_id = kwargs.pop("id", None) or new_id()
        return cls(id=doc_id, **kwargs)


@dataclass
class IncomeCycle(_Document):
    id: str
    cycle_name: str
    net_salary: float
    needs_cap: float
    wants_cap: float
    savings_debt_cap: float
    start_date: str
    end_date: str
    is_active: bool = True

    def __post_init__(self) -> None:
        self.cycle_name = _text(self.cycle_name, "Cycle name")
        self.net_salary = _money(self.net_salary, "Net salary", positive=True)
        self.needs_cap = _money(self.needs_cap, "Needs cap")
        self.wants_cap = _money(self.wants_cap, "Wants cap")
        self.savings_debt_cap = _money(self.savings_debt_cap, "Savings & debt cap")
        self.start_date = _iso_date(self.start_date, "Start date")
        self.end_date = _iso_date(self.end_date, "End date")
        if self.end_date < self.start_date:
            raise ValidationError("End date cannot be before the start date.")
        self.is_active = bool(self.is_active)


@dataclass
class Transaction(_Document):
    id: str
    cycle_id: str
    date: str
    amount: float
    category: str
    bucket: str
    payment_channel: str
    target_account_id: Optional[str] = None
    notes: str = ""
    created_at: str = ""

    def __post_init__(self) -> None:
        self.cycle_id = _text(self.cycle_id, "Pay cycle")
        self.date = _iso_date(self.date, "Date")
        self.amount = _money(self.amount, "Amount", positive=True)
        self.category = _text(self.category, "Category")
        if self.bucket not in LOG_BUCKETS:
            raise ValidationError(f"Bucket must be one of: {', '.join(LOG_BUCKETS)}.")
        self.payment_channel = _text(self.payment_channel, "Payment channel")
        self.target_account_id = _text(self.target_account_id, "Target account", required=False) or None
        self.notes = _text(self.notes, "Notes", required=False)
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Debt(_Document):
    id: str
    name: str
    total_amount: float
    remaining_balance: float
    minimum_payment: float
    due_date: str
    status: str = "active"

    def __post_init__(self) -> None:
        self.name = _text(self.name, "Debt name")
        self.total_amount = _money(self.total_amount, "Total amount", positive=True)
        self.remaining_balance = _money(self.remaining_balance, "Remaining balance")
        if self.remaining_balance > self.total_amount:
            raise ValidationError("Remaining balance cannot exceed the total amount.")
        self.minimum_payment = _money(self.minimum_payment, "Minimum payment")
        self.due_date = _iso_date(self.due_date, "Due date")
        self.status = "paid" if self.remaining_balance == 0 else "active"


@dataclass
class SavingsAccount(_Document):
    id: str
    bank_name: str
    goal_name: str = ""
    current_balance: float = 0.0
    interest_rate: float = 0.0  # % per annum

    def __post_init__(self) -> None:
        self.bank_name = _text(self.bank_name, "Bank name")
        self.goal_name = _text(self.goal_name, "Goal name", required=False)
        self.current_balance = _money(self.current_balance, "Current balance")
        rate = _money(self.interest_rate, "Interest rate")
        if rate > 100:
            raise ValidationError("Interest rate must be between 0 and 100 (%).")
        self.interest_rate = rate
