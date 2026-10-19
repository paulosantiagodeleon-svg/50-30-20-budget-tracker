"""Unit tests: 50/30/20 math, savings routing, debt payoff, atomicity, persistence."""
from __future__ import annotations

import tempfile
from datetime import date
from pathlib import Path

import pytest

from src.firebase_client import (
    DEBTS,
    SAVINGS_ACCOUNTS,
    TRANSACTIONS,
    LocalMockFirestore,
    QueryLogger,
    get_database,
)
from src.models import Debt, Transaction, ValidationError
from src.services import (
    BudgetService,
    calculate_split,
    debt_progress,
    default_end_date,
    summarize_cycle,
)


def make_service(tmp: str):
    db = LocalMockFirestore(Path(tmp) / "db.json", logger=QueryLogger())
    return BudgetService(db), db


def setup_world(service: BudgetService):
    cycle = service.create_cycle("Oct 2026", 30000, "2026-10-01")
    account = service.add_savings_account("GoTyme", "Go Save", 1000)
    debt = service.add_debt("Credit Card", 10000, 500, "2026-10-25", remaining_balance=8000)
    return cycle, account, debt


# ------------------------------------------------------------- 50/30/20 math
def test_split_is_exactly_50_30_20():
    assert calculate_split(30000) == {
        "needs_cap": 15000.0, "wants_cap": 9000.0, "savings_debt_cap": 6000.0}


def test_split_always_sums_to_net_salary():
    for net in (12345.67, 999.99, 0.01 + 100, 18250.55):
        caps = calculate_split(net)
        assert round(sum(caps.values()), 2) == pytest.approx(net)


def test_split_rejects_non_positive_salary():
    for bad in (0, -5, "abc"):
        with pytest.raises(ValidationError):
            calculate_split(bad)


def test_default_end_dates():
    assert default_end_date(date(2026, 10, 1)) == date(2026, 10, 31)
    assert default_end_date(date(2026, 12, 15)) == date(2027, 1, 14)
    assert default_end_date(date(2026, 1, 31)) == date(2026, 2, 27)
    assert default_end_date(date(2026, 10, 1), "Bi-weekly") == date(2026, 10, 14)


def test_summarize_cycle_compares_spend_to_caps():
    cycle = {"net_salary": 30000, **calculate_split(30000)}
    txs = [
        {"bucket": "Needs", "amount": 16000},   # over the 15,000 cap
        {"bucket": "Wants", "amount": 3000},
        {"bucket": "Savings", "amount": 2000},
        {"bucket": "Debt", "amount": 1000},
        {"bucket": "Income", "amount": 500},
    ]
    s = summarize_cycle(cycle, txs)
    assert s["lines"]["Needs"]["over"] is True
    assert s["lines"]["Needs"]["remaining"] == -1000
    assert s["lines"]["Wants"]["pct"] == pytest.approx(3000 / 9000)
    assert s["lines"]["Savings & Debt"]["spent"] == 3000
    assert s["lines"]["Savings & Debt"]["over"] is False
    assert s["total_allocated"] == 22000
    assert s["unallocated"] == 8000
    assert s["other_income"] == 500


# ----------------------------------------------------------------- savings
def test_savings_transaction_increments_account_balance():
    with tempfile.TemporaryDirectory() as tmp:
        service, db = make_service(tmp)
        cycle, account, _ = setup_world(service)
        service.add_transaction(cycle["id"], "2026-10-02", 2500.50, "Monthly savings",
                                "Savings", "Maya", target_account_id=account["id"])
        assert db.get(SAVINGS_ACCOUNTS, account["id"])["current_balance"] == pytest.approx(3500.50)


def test_savings_transaction_requires_valid_account():
    with tempfile.TemporaryDirectory() as tmp:
        service, db = make_service(tmp)
        cycle, _, _ = setup_world(service)
        for target in (None, "does-not-exist"):
            with pytest.raises(ValidationError):
                service.add_transaction(cycle["id"], "2026-10-02", 100, "Save", "Savings", "Cash",
                                        target_account_id=target)
        assert db.query(TRANSACTIONS) == []  # nothing was written


# -------------------------------------------------------------------- debt
def test_debt_transaction_deducts_balance_and_updates_progress():
    with tempfile.TemporaryDirectory() as tmp:
        service, db = make_service(tmp)
        cycle, _, debt = setup_world(service)
        service.add_transaction(cycle["id"], "2026-10-03", 2000, "Card payment",
                                "Debt", "GoTyme", target_account_id=debt["id"])
        updated = db.get(DEBTS, debt["id"])
        assert updated["remaining_balance"] == pytest.approx(6000)
        assert debt_progress(updated["total_amount"], updated["remaining_balance"]) == pytest.approx(0.4)
        assert updated["status"] == "active"


def test_debt_fully_paid_is_marked_paid():
    with tempfile.TemporaryDirectory() as tmp:
        service, db = make_service(tmp)
        cycle, _, debt = setup_world(service)
        service.add_transaction(cycle["id"], "2026-10-03", 8000, "Final payment",
                                "Debt", "Cash", target_account_id=debt["id"])
        paid = db.get(DEBTS, debt["id"])
        assert paid["remaining_balance"] == 0
        assert paid["status"] == "paid"


def test_debt_overpayment_is_rejected_without_side_effects():
    with tempfile.TemporaryDirectory() as tmp:
        service, db = make_service(tmp)
        cycle, _, debt = setup_world(service)
        with pytest.raises(ValidationError):
            service.add_transaction(cycle["id"], "2026-10-03", 8000.01, "Too much",
                                    "Debt", "Cash", target_account_id=debt["id"])
        assert db.get(DEBTS, debt["id"])["remaining_balance"] == 8000
        assert db.query(TRANSACTIONS) == []


def test_debt_progress_bounds():
    assert debt_progress(10000, 10000) == 0.0
    assert debt_progress(10000, 0) == 1.0
    assert debt_progress(0, 0) == 0.0


# ---------------------------------------------------- reversal on deletion
def test_deleting_transactions_reverses_balances():
    with tempfile.TemporaryDirectory() as tmp:
        service, db = make_service(tmp)
        cycle, account, debt = setup_world(service)
        save = service.add_transaction(cycle["id"], "2026-10-02", 500, "Save", "Savings", "Cash",
                                       target_account_id=account["id"])
        pay = service.add_transaction(cycle["id"], "2026-10-02", 700, "Pay", "Debt", "Cash",
                                      target_account_id=debt["id"])
        service.delete_transaction(save["id"])
        service.delete_transaction(pay["id"])
        assert db.get(SAVINGS_ACCOUNTS, account["id"])["current_balance"] == 1000
        assert db.get(DEBTS, debt["id"])["remaining_balance"] == 8000
        assert db.query(TRANSACTIONS) == []


def test_needs_and_wants_do_not_touch_savings_or_debt():
    with tempfile.TemporaryDirectory() as tmp:
        service, db = make_service(tmp)
        cycle, account, debt = setup_world(service)
        tx = service.add_transaction(cycle["id"], "2026-10-02", 300, "Lunch", "Wants", "Cash",
                                     target_account_id=account["id"])  # ignored for Wants
        assert tx["target_account_id"] is None
        assert db.get(SAVINGS_ACCOUNTS, account["id"])["current_balance"] == 1000
        assert db.get(DEBTS, debt["id"])["remaining_balance"] == 8000


# ------------------------------------------------------- cycles & filtering
def test_only_one_active_cycle_at_a_time():
    with tempfile.TemporaryDirectory() as tmp:
        service, _ = make_service(tmp)
        first = service.create_cycle("Oct", 30000, "2026-10-01")
        second = service.create_cycle("Nov", 32000, "2026-11-01")
        assert service.get_active_cycle()["id"] == second["id"]
        service.activate_cycle(first["id"])
        assert [c["id"] for c in service.db.query("income_cycles", [("is_active", "==", True)])] == [first["id"]]


def test_transaction_filters_by_bucket_and_channel():
    with tempfile.TemporaryDirectory() as tmp:
        service, _ = make_service(tmp)
        cycle, _, _ = setup_world(service)
        service.add_transaction(cycle["id"], "2026-10-02", 100, "Fare", "Needs", "Cash")
        service.add_transaction(cycle["id"], "2026-10-03", 200, "Coffee", "Wants", "Maya")
        service.add_transaction(cycle["id"], "2026-10-04", 300, "Rice", "Needs", "Maya")
        assert len(service.list_transactions(cycle["id"])) == 3
        assert len(service.list_transactions(cycle["id"], bucket="Needs")) == 2
        assert len(service.list_transactions(cycle["id"], payment_channel="Maya")) == 2
        both = service.list_transactions(cycle["id"], bucket="Needs", payment_channel="Maya")
        assert [t["category"] for t in both] == ["Rice"]
        assert service.list_transactions(cycle["id"])[0]["date"] == "2026-10-04"  # newest first


# ------------------------------------------------------- models validation
def test_models_reject_invalid_documents():
    with pytest.raises(ValidationError):
        Transaction.new(cycle_id="c", date="2026-10-01", amount=-5, category="x",
                        bucket="Needs", payment_channel="Cash")
    with pytest.raises(ValidationError):
        Transaction.new(cycle_id="c", date="2026-10-01", amount=5, category="x",
                        bucket="Nonsense", payment_channel="Cash")
    with pytest.raises(ValidationError):
        Debt.new(name="Loan", total_amount=100, remaining_balance=150,
                 minimum_payment=10, due_date="2026-10-25")
    with pytest.raises(ValidationError):
        Debt.new(name="Loan", total_amount=100, remaining_balance=50,
                 minimum_payment=10, due_date="not-a-date")


# ------------------------------------------- mock database behaviour
def test_batch_write_is_atomic_when_one_op_fails():
    with tempfile.TemporaryDirectory() as tmp:
        _, db = make_service(tmp)
        with pytest.raises(KeyError):
            db.batch_write([
                ("set", DEBTS, "a", {"name": "A"}),
                ("update", DEBTS, "missing", {"name": "B"}),
            ])
        assert db.get(DEBTS, "a") is None


def test_mock_persists_to_json_and_reloads():
    with tempfile.TemporaryDirectory() as tmp:
        service, _ = make_service(tmp)
        service.add_savings_account("MariBank", "Travel", 750)
        reloaded = BudgetService(LocalMockFirestore(Path(tmp) / "db.json", logger=QueryLogger()))
        assert [a["bank_name"] for a in reloaded.list_savings_accounts()] == ["MariBank"]


def test_query_operators_and_logging():
    with tempfile.TemporaryDirectory() as tmp:
        service, db = make_service(tmp)
        service.add_debt("A", 1000, 50, "2026-10-10")
        service.add_debt("B", 5000, 50, "2026-10-10")
        big = db.query(DEBTS, [("total_amount", ">=", 2000)])
        assert [d["name"] for d in big] == ["B"]
        assert [d["name"] for d in db.query(DEBTS, [("name", "in", ["A", "B"])])] == ["A", "B"]
        last = db.logger.recent(1)[0]
        assert last["operation"] == "QUERY"
        assert ".where('name', 'in', ['A', 'B'])" in last["query"]
        assert last["result_count"] == 2
        assert last["payload"][0]["name"] == "A"


def test_firebase_client_falls_back_to_mock_without_key():
    with tempfile.TemporaryDirectory() as tmp:
        db, note = get_database(tmp, logger=QueryLogger())
        assert isinstance(db, LocalMockFirestore)
        assert db.is_live is False
        assert "serviceAccountKey.json" in note


# --------------------------------------------------------------- seeding
def test_seed_sample_data_is_idempotent_and_consistent():
    with tempfile.TemporaryDirectory() as tmp:
        service, db = make_service(tmp)
        today = date(2026, 10, 20)
        service.seed_sample_data(today)
        first = db.snapshot()
        service.seed_sample_data(today)
        second = db.snapshot()
        assert {k: len(v) for k, v in first.items()} == {k: len(v) for k, v in second.items()}
        assert first["savings_accounts"] == second["savings_accounts"]
        balances = {a["bank_name"]: a["current_balance"] for a in second["savings_accounts"]}
        assert balances["GoTyme"] == 14000 and balances["MariBank"] == 9000
        assert balances["Emergency Fund"] == 25500
        remaining = {d["name"]: d["remaining_balance"] for d in second["debts"]}
        assert remaining == {"Credit Card": 17000, "Salary Loan": 20000}
        summary = service.cycle_summary(service.get_active_cycle())
        assert summary["lines"]["Needs"]["cap"] == 20000
        assert summary["spent"]["Needs"] == 17299
