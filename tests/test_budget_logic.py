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
from src.models import Debt, RecurringTemplate, Transaction, ValidationError
from src.services import (
    ALL_COLLECTIONS,
    RECURRING_TEMPLATES,
    WIPE_BATCH_SIZE,
    BudgetService,
    calculate_interest_projection,
    calculate_split,
    clear_ui_state,
    create_recurring_template,
    debt_progress,
    default_end_date,
    delete_recurring_template,
    list_recurring_templates,
    log_transaction,
    next_due_date,
    portfolio_interest_projection,
    post_recurring_templates,
    summarize_cycle,
    top_up_cycle_salary,
    update_cycle_dates,
    wipe_all_data,
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


# =====================================================================  v1.1.0
# ------------------------------------------------ 15th/30th paycheck top-up
def test_top_up_adds_salary_and_resplits_caps_exactly():
    with tempfile.TemporaryDirectory() as tmp:
        service, db = make_service(tmp)
        cycle = service.create_cycle("Oct 2026", 30000, "2026-10-01")
        updated = top_up_cycle_salary(db, cycle["id"], 15000.55, "30th paycheck")
        assert updated["net_salary"] == pytest.approx(45000.55)
        assert updated["needs_cap"] == pytest.approx(round(45000.55 * 0.5, 2))
        assert updated["wants_cap"] == pytest.approx(round(45000.55 * 0.3, 2))
        total = updated["needs_cap"] + updated["wants_cap"] + updated["savings_debt_cap"]
        assert round(total, 2) == pytest.approx(updated["net_salary"])  # remainder absorbs rounding
        stored = db.get("income_cycles", cycle["id"])
        assert stored["net_salary"] == updated["net_salary"]
        assert stored["savings_debt_cap"] == updated["savings_debt_cap"]
        assert stored["top_ups"][0]["note"] == "30th paycheck"
        assert stored["top_ups"][0]["amount"] == pytest.approx(15000.55)


def test_top_up_round_numbers_and_repeated_top_ups_accumulate():
    with tempfile.TemporaryDirectory() as tmp:
        service, db = make_service(tmp)
        cycle = service.create_cycle("Oct 2026", 20000, "2026-10-01")
        service.top_up_cycle_salary(cycle["id"], 10000)
        final = service.top_up_cycle_salary(cycle["id"], 10000, "bonus")
        assert (final["net_salary"], final["needs_cap"], final["wants_cap"], final["savings_debt_cap"]) == \
            (40000, 20000, 12000, 8000)
        assert len(final["top_ups"]) == 2


def test_top_up_rejects_bad_amounts_and_unknown_cycle_without_changes():
    with tempfile.TemporaryDirectory() as tmp:
        service, db = make_service(tmp)
        cycle = service.create_cycle("Oct 2026", 30000, "2026-10-01")
        for bad in (0, -100, "abc", None, float("nan"), 0.001):
            with pytest.raises(ValidationError):
                top_up_cycle_salary(db, cycle["id"], bad)
        with pytest.raises(ValidationError):
            top_up_cycle_salary(db, "no-such-cycle", 500)
        stored = db.get("income_cycles", cycle["id"])
        assert stored["net_salary"] == 30000
        assert stored["top_ups"] == []


def test_top_up_changes_caps_used_by_cycle_summary():
    with tempfile.TemporaryDirectory() as tmp:
        service, _ = make_service(tmp)
        cycle = service.create_cycle("Oct 2026", 20000, "2026-10-01")
        service.add_transaction(cycle["id"], "2026-10-02", 12000, "Rent", "Needs", "Cash")
        assert service.cycle_summary(service.db.get("income_cycles", cycle["id"]))["lines"]["Needs"]["over"]
        service.top_up_cycle_salary(cycle["id"], 10000)
        needs = service.cycle_summary(service.db.get("income_cycles", cycle["id"]))["lines"]["Needs"]
        assert needs["cap"] == 15000 and needs["over"] is False and needs["remaining"] == 3000


# ------------------------------------------------------ recurring templates
def test_recurring_template_model_validation():
    ok = RecurringTemplate.new(name="Netflix", amount=549, category="Streaming", bucket="Wants",
                               payment_channel="Credit Card", due_day=31, target_account_id="ignored")
    assert ok.due_day == 31 and ok.target_account_id is None  # unlinked buckets drop the target
    for bad_day in (0, 32, 1.5, "x", None, -3):
        with pytest.raises(ValidationError):
            RecurringTemplate.new(name="Bad", amount=100, category="c", bucket="Needs",
                                  payment_channel="Cash", due_day=bad_day)
    with pytest.raises(ValidationError):
        RecurringTemplate.new(name="Bad", amount=0, category="c", bucket="Needs",
                              payment_channel="Cash", due_day=5)
    with pytest.raises(ValidationError):
        RecurringTemplate.new(name="Bad", amount=10, category="c", bucket="Nonsense",
                              payment_channel="Cash", due_day=5)
    with pytest.raises(ValidationError):
        RecurringTemplate.new(name="Auto-save", amount=10, category="c", bucket="Savings",
                              payment_channel="Cash", due_day=5)  # linked bucket needs a target


def test_recurring_template_create_list_delete_and_persistence():
    with tempfile.TemporaryDirectory() as tmp:
        service, db = make_service(tmp)
        cycle, account, debt = setup_world(service)
        late = create_recurring_template(db, "Water", 450, "Water", "Needs", "Cash", 28)
        early = create_recurring_template(db, "Meralco", 3200, "Electricity", "Needs", "Maya", 15)
        create_recurring_template(db, "Auto-Save", 1000, "Auto-save", "Savings", "GoTyme", 30,
                                  target_account_id=account["id"])
        assert [t["name"] for t in list_recurring_templates(db)] == ["Meralco", "Water", "Auto-Save"]
        with pytest.raises(ValidationError):
            create_recurring_template(db, "Ghost", 100, "x", "Savings", "Cash", 5, target_account_id="nope")
        with pytest.raises(ValidationError):
            create_recurring_template(db, "Ghost", 100, "x", "Debt", "Cash", 5, target_account_id="nope")

        # survives a restart of the local mock (the collection is not dropped on reload)
        reloaded = LocalMockFirestore(Path(tmp) / "db.json", logger=QueryLogger())
        assert len(list_recurring_templates(reloaded)) == 3

        delete_recurring_template(db, late["id"])
        assert [t["name"] for t in list_recurring_templates(db)] == ["Meralco", "Auto-Save"]
        with pytest.raises(ValidationError):
            delete_recurring_template(db, late["id"])
        assert early["id"] in {t["id"] for t in service.list_recurring_templates()}


def test_post_recurring_templates_logs_transactions_and_routes_balances():
    with tempfile.TemporaryDirectory() as tmp:
        service, db = make_service(tmp)
        cycle, account, debt = setup_world(service)
        bills = service.create_recurring_template("Meralco", 3200, "Electricity", "Needs", "Maya", 15)
        save = service.create_recurring_template("Auto-Save", 1000, "Auto-save", "Savings", "GoTyme", 30,
                                                 target_account_id=account["id"])
        card = service.create_recurring_template("Card payment", 500, "Credit card", "Debt", "GoTyme", 25,
                                                 target_account_id=debt["id"])
        service.create_recurring_template("Netflix", 549, "Streaming", "Wants", "Credit Card", 3)  # not ticked
        result = post_recurring_templates(db, cycle["id"],
                                          [bills["id"], save["id"], card["id"], bills["id"]], "2026-10-15")
        assert result["failed"] == []
        assert len(result["posted"]) == 3  # the duplicate id is only posted once
        txs = service.list_transactions(cycle["id"])
        assert {t["category"] for t in txs} == {"Electricity", "Auto-save", "Credit card"}
        assert all(t["date"] == "2026-10-15" and t["notes"].startswith("Recurring: ") for t in txs)
        assert db.get(SAVINGS_ACCOUNTS, account["id"])["current_balance"] == pytest.approx(2000)
        assert db.get(DEBTS, debt["id"])["remaining_balance"] == pytest.approx(7500)
        summary = service.cycle_summary(db.get("income_cycles", cycle["id"]))
        assert summary["spent"]["Needs"] == 3200
        assert summary["lines"]["Savings & Debt"]["spent"] == 1500


def test_post_recurring_templates_reports_failures_and_posts_the_rest():
    with tempfile.TemporaryDirectory() as tmp:
        service, db = make_service(tmp)
        cycle, account, debt = setup_world(service)
        fine = service.create_recurring_template("Meralco", 3200, "Electricity", "Needs", "Maya", 15)
        too_big = service.create_recurring_template("Big payment", 9000, "Loan", "Debt", "Cash", 20,
                                                    target_account_id=debt["id"])  # remaining is 8,000
        result = post_recurring_templates(db, cycle["id"], [too_big["id"], fine["id"]], "2026-10-15")
        assert [t["category"] for t in result["posted"]] == ["Electricity"]
        assert [f["template_id"] for f in result["failed"]] == [too_big["id"]]
        assert "exceeds" in result["failed"][0]["error"]
        assert db.get(DEBTS, debt["id"])["remaining_balance"] == 8000


def test_post_recurring_templates_validates_before_posting_anything():
    with tempfile.TemporaryDirectory() as tmp:
        service, db = make_service(tmp)
        cycle, _, _ = setup_world(service)
        fine = service.create_recurring_template("Meralco", 3200, "Electricity", "Needs", "Maya", 15)
        with pytest.raises(ValidationError):
            post_recurring_templates(db, cycle["id"], [fine["id"], "missing"], "2026-10-15")
        with pytest.raises(ValidationError):
            post_recurring_templates(db, cycle["id"], [fine["id"]], "not-a-date")
        with pytest.raises(ValidationError):
            post_recurring_templates(db, "no-cycle", [fine["id"]], "2026-10-15")
        assert post_recurring_templates(db, cycle["id"], [], "2026-10-15") == {"posted": [], "failed": []}
        assert db.query(TRANSACTIONS) == []


def test_log_transaction_matches_manual_transaction_behaviour():
    with tempfile.TemporaryDirectory() as tmp:
        service, db = make_service(tmp)
        cycle, account, _ = setup_world(service)
        tx = log_transaction(db, cycle["id"], "2026-10-02", 250, "Save", "Savings", "Cash",
                             target_account_id=account["id"])
        assert tx["bucket"] == "Savings"
        assert db.get(SAVINGS_ACCOUNTS, account["id"])["current_balance"] == 1250


def test_next_due_date_handles_month_ends_and_rollover():
    assert next_due_date(15, date(2026, 10, 10)) == date(2026, 10, 15)
    assert next_due_date(15, date(2026, 10, 15)) == date(2026, 10, 15)
    assert next_due_date(15, date(2026, 10, 20)) == date(2026, 11, 15)
    assert next_due_date(6, date(2026, 12, 20)) == date(2027, 1, 6)
    assert next_due_date(31, date(2027, 2, 10)) == date(2027, 2, 28)
    assert next_due_date(30, date(2026, 1, 31)) == date(2026, 2, 28)


def test_recurring_collection_is_whitelisted_for_the_mock_and_wiped_by_reset():
    assert RECURRING_TEMPLATES == "recurring_templates"
    assert RECURRING_TEMPLATES in ALL_COLLECTIONS
    with tempfile.TemporaryDirectory() as tmp:
        service, db = make_service(tmp)
        service.create_recurring_template("Water", 450, "Water", "Needs", "Cash", 28)
        assert len(db.snapshot()[RECURRING_TEMPLATES]) == 1  # visible in the Developer tab snapshot
        db.reset()
        assert list_recurring_templates(db) == []


# ------------------------------------------- PH digital bank interest projector
def test_interest_projection_applies_exact_20_percent_withholding_tax():
    p = calculate_interest_projection(100000, 4)
    assert p["annual_gross"] == pytest.approx(4000)
    assert p["annual_net"] == pytest.approx(3200)
    assert p["monthly_gross"] == pytest.approx(4000 / 12)
    assert p["monthly_net"] == pytest.approx(3200 / 12)
    assert p["daily_gross"] == pytest.approx(4000 / 365)
    assert p["daily_net"] == pytest.approx(3200 / 365)
    for period in ("daily", "monthly", "annual"):
        assert p[f"{period}_net"] == pytest.approx(p[f"{period}_gross"] * 0.80)
        assert p[f"{period}_gross"] - p[f"{period}_net"] == pytest.approx(p[f"{period}_gross"] * 0.20)


def test_interest_projection_edge_cases_and_validation():
    zero = calculate_interest_projection(0, 5)
    assert set(zero) == {"daily_gross", "monthly_gross", "annual_gross",
                         "daily_net", "monthly_net", "annual_net"}
    assert all(v == 0 for v in zero.values())
    assert calculate_interest_projection(12345.67, 0)["annual_net"] == 0
    assert calculate_interest_projection(1000, 100)["annual_net"] == pytest.approx(800)
    for balance, rate in ((-1, 4), (100, -0.5), (100, 101), ("x", 4), (100, None), (float("inf"), 4)):
        with pytest.raises(ValidationError):
            calculate_interest_projection(balance, rate)


def test_portfolio_interest_projection_sums_accounts():
    accounts = [{"current_balance": 100000, "interest_rate": 4.0},
                {"current_balance": 50000, "interest_rate": 3.0},
                {"current_balance": 25000}]  # missing rate counts as 0%
    total = portfolio_interest_projection(accounts)
    assert total["annual_gross"] == pytest.approx(4000 + 1500)
    assert total["annual_net"] == pytest.approx((4000 + 1500) * 0.80)
    assert total["monthly_net"] == pytest.approx((4000 + 1500) * 0.80 / 12)
    assert total["daily_net"] == pytest.approx((4000 + 1500) * 0.80 / 365)
    assert portfolio_interest_projection([])["annual_net"] == 0


# ---------------------------------------------------- v1.1 seed data
def test_seed_includes_three_idempotent_recurring_bills_that_post_cleanly():
    with tempfile.TemporaryDirectory() as tmp:
        service, db = make_service(tmp)
        today = date(2026, 10, 20)
        counts = service.seed_sample_data(today)
        service.seed_sample_data(today)
        assert counts["recurring_templates"] == 3
        templates = {t["id"]: t for t in list_recurring_templates(db)}
        assert set(templates) == {"seed-rec-1", "seed-rec-2", "seed-rec-3"}
        assert templates["seed-rec-1"]["name"] == "Meralco Electricity"
        assert templates["seed-rec-2"]["name"] == "Fiber Internet"
        assert templates["seed-rec-3"]["name"] == "GoTyme Auto-Save"
        assert templates["seed-rec-3"]["target_account_id"] == "seed-gotyme"
        cycle = service.get_active_cycle()
        before = db.get(SAVINGS_ACCOUNTS, "seed-gotyme")["current_balance"]
        result = post_recurring_templates(db, cycle["id"], list(templates), today)
        assert len(result["posted"]) == 3 and result["failed"] == []
        assert db.get(SAVINGS_ACCOUNTS, "seed-gotyme")["current_balance"] == before + 1000


# =====================================================================  v1.1.1
# ------------------------------------------------ custom cycle end dates
def test_create_cycle_accepts_irregular_custom_range():
    with tempfile.TemporaryDirectory() as tmp:
        service, db = make_service(tmp)
        cycle = service.create_cycle("Oct 26 - Nov 30", 36000, "2026-10-26", "2026-11-30")
        stored = db.get("income_cycles", cycle["id"])
        assert (stored["start_date"], stored["end_date"]) == ("2026-10-26", "2026-11-30")
        assert stored["needs_cap"] == 18000 and stored["wants_cap"] == 10800
        assert stored["savings_debt_cap"] == 7200  # caps still sum to net salary
        assert stored["is_active"] is True


def test_create_cycle_without_end_date_still_uses_the_frequency_default():
    with tempfile.TemporaryDirectory() as tmp:
        service, _ = make_service(tmp)
        monthly = service.create_cycle("M", 30000, "2026-10-26")
        biweekly = service.create_cycle("B", 30000, date(2026, 10, 26), frequency="Bi-weekly")
        assert monthly["end_date"] == "2026-11-25"
        assert biweekly["end_date"] == "2026-11-08"
        with pytest.raises(ValidationError):
            service.create_cycle("X", 30000, "2026-10-26", frequency="Weekly")


def test_create_cycle_allows_single_day_and_accepts_date_objects():
    with tempfile.TemporaryDirectory() as tmp:
        service, _ = make_service(tmp)
        one_day = service.create_cycle("Bonus day", 5000, date(2026, 12, 15), date(2026, 12, 15))
        assert one_day["start_date"] == one_day["end_date"] == "2026-12-15"


def test_create_cycle_rejects_end_before_start_and_changes_nothing():
    with tempfile.TemporaryDirectory() as tmp:
        service, _ = make_service(tmp)
        first = service.create_cycle("Oct", 30000, "2026-10-01")
        with pytest.raises(ValidationError):
            service.create_cycle("Bad", 30000, "2026-11-30", "2026-11-01")
        with pytest.raises(ValidationError):
            service.create_cycle("Bad", 30000, "2026-11-01", "not-a-date")
        assert [c["id"] for c in service.list_cycles()] == [first["id"]]
        assert service.get_active_cycle()["id"] == first["id"]  # the old cycle was not deactivated


def test_update_cycle_dates_changes_only_the_dates():
    with tempfile.TemporaryDirectory() as tmp:
        service, db = make_service(tmp)
        cycle = service.create_cycle("Oct", 30000, "2026-10-01")
        service.top_up_cycle_salary(cycle["id"], 5000, "bonus")
        tx = service.add_transaction(cycle["id"], "2026-10-05", 400, "Fare", "Needs", "Cash")
        updated = update_cycle_dates(db, cycle["id"], "2026-10-26", date(2026, 11, 30))
        assert (updated["start_date"], updated["end_date"]) == ("2026-10-26", "2026-11-30")
        stored = db.get("income_cycles", cycle["id"])
        assert (stored["start_date"], stored["end_date"]) == ("2026-10-26", "2026-11-30")
        assert stored["net_salary"] == 35000 and len(stored["top_ups"]) == 1
        assert stored["needs_cap"] == 17500
        assert [t["id"] for t in service.list_transactions(cycle["id"])] == [tx["id"]]  # entries untouched
        assert service.update_cycle_dates(cycle["id"], "2026-10-26", "2026-12-01")["end_date"] == "2026-12-01"


def test_update_cycle_dates_validation_leaves_the_cycle_unchanged():
    with tempfile.TemporaryDirectory() as tmp:
        service, db = make_service(tmp)
        cycle = service.create_cycle("Oct", 30000, "2026-10-01")
        for start, end in (("2026-11-30", "2026-11-01"), ("oops", "2026-11-01"), ("2026-11-01", "")):
            with pytest.raises(ValidationError):
                update_cycle_dates(db, cycle["id"], start, end)
        with pytest.raises(ValidationError):
            update_cycle_dates(db, "no-such-cycle", "2026-10-01", "2026-10-31")
        stored = db.get("income_cycles", cycle["id"])
        assert (stored["start_date"], stored["end_date"]) == ("2026-10-01", "2026-10-31")


# ------------------------------------------------------- wipe all data
def _populated(service, db):
    service.seed_sample_data(date(2026, 10, 20))
    service.top_up_cycle_salary(service.get_active_cycle()["id"], 1000)
    return db.snapshot()


def test_wipe_all_data_clears_all_five_collections_on_the_mock():
    with tempfile.TemporaryDirectory() as tmp:
        service, db = make_service(tmp)
        before = _populated(service, db)
        assert all(before[c] for c in ("income_cycles", "transactions", "debts",
                                       "savings_accounts", "recurring_templates"))
        counts = wipe_all_data(db)
        assert counts == {name: len(before[name]) for name in ALL_COLLECTIONS}
        assert counts["recurring_templates"] == 3 and counts["transactions"] == 15
        assert all(docs == [] for docs in db.snapshot().values())
        assert service.get_active_cycle() is None and service.list_savings_accounts() == []
        assert service.list_debts() == [] and service.list_recurring_templates() == []
        # the JSON file on disk is empty too, so a restart really starts from a fresh slate
        reloaded = LocalMockFirestore(Path(tmp) / "db.json", logger=QueryLogger())
        assert all(docs == [] for docs in reloaded.snapshot().values())


def test_wipe_all_data_leaves_a_fully_usable_empty_slate():
    with tempfile.TemporaryDirectory() as tmp:
        service, db = make_service(tmp)
        _populated(service, db)
        service.wipe_all_data()
        cycle = service.create_cycle("Fresh", 20000, "2026-11-01")
        assert service.get_active_cycle()["id"] == cycle["id"]
        service.seed_sample_data(date(2026, 11, 3))  # seeding after a wipe works
        assert len(service.list_recurring_templates()) == 3
        assert wipe_all_data(db)["income_cycles"] == 2  # the fresh cycle + the sample cycle


def test_wipe_all_data_on_an_empty_database_is_a_safe_noop():
    with tempfile.TemporaryDirectory() as tmp:
        _, db = make_service(tmp)
        assert wipe_all_data(db) == {name: 0 for name in ALL_COLLECTIONS}


class _NoResetMock(LocalMockFirestore):
    """Behaves like LiveFirestore from the wipe's point of view: no reset(), only batched deletes."""
    supports_reset = False


def test_wipe_all_data_uses_chunked_batch_deletes_when_reset_is_unavailable():
    with tempfile.TemporaryDirectory() as tmp:
        db = _NoResetMock(Path(tmp) / "db.json", logger=QueryLogger())
        total = WIPE_BATCH_SIZE * 2 + 50
        docs = [("set", DEBTS, f"d{i}", {"name": f"D{i}"}) for i in range(total)]
        for start in range(0, total, 200):
            db.batch_write(docs[start:start + 200])
        db.set("income_cycles", "c1", {"cycle_name": "x"})
        db.set(RECURRING_TEMPLATES, "r1", {"name": "y"})
        db.logger.clear()
        counts = wipe_all_data(db)
        assert counts[DEBTS] == total and counts["income_cycles"] == 1 and counts[RECURRING_TEMPLATES] == 1
        assert all(docs == [] for docs in db.snapshot().values())
        batches = [e for e in db.logger.recent() if e["operation"].startswith("BATCH")]
        assert len(batches) == 3 and all(e["result_count"] <= WIPE_BATCH_SIZE for e in batches)


def test_wipe_all_data_raises_if_documents_survive():
    class _Stubborn(_NoResetMock):
        def _raw_commit(self, ops):  # silently ignores deletes
            return None

    with tempfile.TemporaryDirectory() as tmp:
        db = _Stubborn(Path(tmp) / "db.json", logger=QueryLogger())
        db._data[DEBTS]["a"] = {"id": "a", "name": "A"}
        with pytest.raises(RuntimeError):
            wipe_all_data(db)


def test_clear_ui_state_removes_only_form_keys():
    state = {"rec_chk_1": True, "rec_post_date": 1, "topup_amount": 5.0, "new_cycle_end": 2,
             "cycle_edit_start_abc": 3, "wipe_confirm_text": "WIPE", "_flash": "hi", "other": 1}
    removed = clear_ui_state(state)
    assert sorted(removed) == ["cycle_edit_start_abc", "new_cycle_end", "rec_chk_1", "rec_post_date",
                               "topup_amount", "wipe_confirm_text"]
    assert state == {"_flash": "hi", "other": 1}
