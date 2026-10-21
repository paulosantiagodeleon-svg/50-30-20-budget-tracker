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
from src.models import Debt, RecurringTemplate, SplitPart, Transaction, ValidationError
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
from src.services import (  # v1.2 additions
    asset_allocation,
    collect_tags,
    daily_spend_series,
    extract_hashtags,
    filter_by_tags,
    find_possible_duplicates,
    pacing_summary,
    parse_amount,
    parse_bank_csv,
    parse_date_text,
    sweep_unspent_to_savings,
    tag_totals,
    transactions_to_csv,
    unspent_capacity,
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


# =====================================================================  v1.2.0
OCT_CYCLE = {"start_date": "2026-10-01", "end_date": "2026-10-10", "net_salary": 20000, **calculate_split(20000)}


# --------------------------------------------------------- hashtag extraction
def test_extract_hashtags_finds_lowercase_unique_tags_in_order():
    assert extract_hashtags("New GPU #pc-build and fan #PC-Build #vacation_2026") == ["pc-build", "vacation_2026"]
    assert extract_hashtags("Trip #vacation, hotel (#vacation) #Cebu.") == ["vacation", "cebu"]
    assert extract_hashtags("#goals2026!") == ["goals2026"]
    assert extract_hashtags("#piñata-party #Pasko") == ["piñata-party", "pasko"]


def test_extract_hashtags_ignores_things_that_are_not_tags():
    for text in ("C# is fun", "see page#top", "item #1", "##double", "me@x.com#frag", "no tags", "", None):
        assert extract_hashtags(text) == [], text
    assert extract_hashtags("#trip- next #a--b") == ["trip", "a"]  # trailing/double separators are not part of a tag


def test_collect_tags_counts_transactions_not_occurrences():
    txs = [{"notes": "#a #b"}, {"notes": "#a #a"}, {"notes": "x #c"}, {"notes": ""}, {}]
    tags = collect_tags(txs)
    assert tags == {"a": 2, "b": 1, "c": 1}
    assert list(tags) == ["a", "b", "c"]  # most used first, ties alphabetical


def test_filter_by_tags_any_all_and_normalisation():
    txs = [{"id": 1, "notes": "#pc-build GPU"}, {"id": 2, "notes": "#vacation #pc-build"}, {"id": 3, "notes": "lunch"}]
    assert [t["id"] for t in filter_by_tags(txs, ["pc-build"])] == [1, 2]
    assert [t["id"] for t in filter_by_tags(txs, ["#PC-Build "])] == [1, 2]
    assert [t["id"] for t in filter_by_tags(txs, ["vacation", "pc-build"])] == [1, 2]  # any
    assert [t["id"] for t in filter_by_tags(txs, ["vacation", "pc-build"], match_all=True)] == [2]
    assert [t["id"] for t in filter_by_tags(txs, [])] == [1, 2, 3]
    assert filter_by_tags(txs, ["nope"]) == []


def test_tag_totals_skip_income_and_sweeps():
    txs = [
        {"bucket": "Needs", "amount": 100, "category": "x", "notes": "#pc"},
        {"bucket": "Wants", "amount": 50, "category": "x", "notes": "#pc #game"},
        {"bucket": "Income", "amount": 999, "category": "x", "notes": "#pc"},
        {"bucket": "Savings", "amount": 500, "category": "Rollover sweep", "notes": "#rollover"},
    ]
    assert tag_totals(txs) == [{"tag": "pc", "count": 2, "amount": 150},
                               {"tag": "game", "count": 1, "amount": 50}]


# ------------------------------------------------------ analytics data layer
def test_asset_allocation_shares_and_empty_accounts():
    accounts = [{"bank_name": "GoTyme", "goal_name": "Go Save", "current_balance": 3000},
                {"bank_name": "Maya", "goal_name": "", "current_balance": 1000},
                {"bank_name": "Empty", "goal_name": "", "current_balance": 0}]
    rows = asset_allocation(accounts)
    assert [(r["account"], r["balance"], r["share"]) for r in rows] == [
        ("GoTyme - Go Save", 3000, 75.0), ("Maya", 1000, 25.0)]
    assert asset_allocation([]) == []


def test_daily_spend_series_is_zero_filled_and_ignores_out_of_scope_rows():
    txs = [
        {"bucket": "Needs", "amount": 100, "date": "2026-10-02", "category": "a"},
        {"bucket": "Needs", "amount": 30, "date": "2026-10-02", "category": "b"},
        {"bucket": "Wants", "amount": 50, "date": "2026-10-02", "category": "c"},
        {"bucket": "Needs", "amount": 999, "date": "2026-10-15", "category": "outside the cycle"},
        {"bucket": "Savings", "amount": 700, "date": "2026-10-03", "category": "Save"},
        {"bucket": "Savings", "amount": 800, "date": "2026-10-04", "category": "Rollover sweep"},
    ]
    rows = daily_spend_series(OCT_CYCLE, txs)
    assert len(rows) == 10 * 2 and rows[0] == {"date": "2026-10-01", "bucket": "Needs", "amount": 0}
    day2 = {r["bucket"]: r["amount"] for r in rows if r["date"] == "2026-10-02"}
    assert day2 == {"Needs": 130, "Wants": 50}
    assert sum(r["amount"] for r in rows) == 180
    assert len(daily_spend_series(OCT_CYCLE, txs, ("Needs",))) == 10
    with_savings = daily_spend_series(OCT_CYCLE, txs, ("Savings",))
    assert sum(r["amount"] for r in with_savings) == 700  # the sweep is not "spending"
    with pytest.raises(ValidationError):
        daily_spend_series(OCT_CYCLE, txs, ("Needs", "Income"))


def test_pacing_summary_under_over_and_edges():
    base = [{"bucket": "Needs", "amount": 5000, "date": "2026-10-02", "category": "r"},
            {"bucket": "Wants", "amount": 1500, "date": "2026-10-04", "category": "g"},
            {"bucket": "Needs", "amount": 400, "date": "2026-11-01", "category": "outside"}]
    p = pacing_summary(OCT_CYCLE, base, date(2026, 10, 5))
    assert (p["days_total"], p["days_elapsed"]) == (10, 5)
    assert p["target_total"] == 16000 and p["target_per_day"] == 1600 and p["target_to_date"] == 8000
    assert p["spent"] == 6500 and p["variance"] == 1500 and p["on_track"] is True
    assert p["projected_total"] == 13000 and p["outside_range"] == 1
    over = pacing_summary(OCT_CYCLE, base + [{"bucket": "Wants", "amount": 2500, "date": "2026-10-05",
                                              "category": "x"}], date(2026, 10, 5))
    assert over["spent"] == 9000 and over["variance"] == -1000 and over["on_track"] is False
    before = pacing_summary(OCT_CYCLE, [], date(2026, 9, 20))
    assert before["days_elapsed"] == 0 and before["target_to_date"] == 0 and before["projected_total"] == 0
    assert pacing_summary(OCT_CYCLE, [], date(2026, 12, 1))["days_elapsed"] == 10  # capped at the cycle length
    assert pacing_summary(OCT_CYCLE, [], date(2026, 10, 5), ("Savings", "Debt"))["target_total"] == 4000


# ------------------------------------------------------------ rollover sweep
def _spend_for_sweep(service, cycle, account):
    service.add_transaction(cycle["id"], "2026-10-02", 9000.50, "Rent", "Needs", "Cash")
    service.add_transaction(cycle["id"], "2026-10-03", 2500.25, "Games", "Wants", "Maya")
    service.add_transaction(cycle["id"], "2026-10-04", 2000, "Save", "Savings", "Cash",
                            target_account_id=account["id"])


def test_unspent_capacity_calculation_and_overspent_buckets():
    cycle = {"net_salary": 30000, **calculate_split(30000)}
    txs = [{"bucket": "Needs", "amount": 9000.50, "category": "a"}, {"bucket": "Wants", "amount": 2500.25, "category": "b"},
           {"bucket": "Savings", "amount": 2000, "category": "c"}, {"bucket": "Debt", "amount": 1000, "category": "d"},
           {"bucket": "Income", "amount": 500, "category": "e"}]
    info = unspent_capacity(cycle, txs)
    assert info["needs"] == pytest.approx(5999.50) and info["wants"] == pytest.approx(6499.75)
    assert info["total"] == info["sweepable"] == pytest.approx(12499.25) and info["already_swept"] == 0
    assert unspent_capacity(cycle, [])["total"] == 24000  # Savings/Debt/Income never change it
    over = unspent_capacity(cycle, [{"bucket": "Needs", "amount": 16000, "category": "a"},
                                    {"bucket": "Wants", "amount": 7000, "category": "b"}])
    assert (over["needs"], over["wants"], over["total"]) == (0, 2000, 2000)  # overspent Needs is not netted


def test_sweep_creates_one_grouped_savings_transaction_and_updates_balance():
    with tempfile.TemporaryDirectory() as tmp:
        service, db = make_service(tmp)
        cycle, account, _ = setup_world(service)
        _spend_for_sweep(service, cycle, account)  # account balance is now 3,000
        result = sweep_unspent_to_savings(db, cycle["id"], account["id"], today=date(2026, 10, 31))
        assert result["amount"] == pytest.approx(12499.25)
        assert (result["needs"], result["wants"]) == (pytest.approx(5999.50), pytest.approx(6499.75))
        tx = result["transaction"]
        assert (tx["bucket"], tx["category"], tx["date"]) == ("Savings", "Rollover sweep", "2026-10-31")
        assert tx["target_account_id"] == account["id"] and "#rollover" in tx["notes"]
        assert db.get(SAVINGS_ACCOUNTS, account["id"])["current_balance"] == pytest.approx(3000 + 12499.25)
        assert result["account"]["current_balance"] == pytest.approx(15499.25)
        assert len([t for t in service.list_transactions(cycle["id"]) if t["category"] == "Rollover sweep"]) == 1


def test_sweep_is_not_counted_against_the_20_percent_cap():
    with tempfile.TemporaryDirectory() as tmp:
        service, db = make_service(tmp)
        cycle, account, _ = setup_world(service)
        _spend_for_sweep(service, cycle, account)
        service.sweep_unspent_to_savings(cycle["id"], account["id"], date(2026, 10, 31))
        summary = service.cycle_summary(db.get("income_cycles", cycle["id"]))
        assert summary["swept"] == pytest.approx(12499.25)
        assert summary["lines"]["Savings & Debt"]["spent"] == pytest.approx(2000)  # only the planned saving
        assert summary["spent"]["Savings"] == pytest.approx(14499.25)
        assert summary["total_allocated"] == pytest.approx(26000)


def test_sweep_cannot_move_the_same_money_twice_and_can_be_undone():
    with tempfile.TemporaryDirectory() as tmp:
        service, db = make_service(tmp)
        cycle, account, _ = setup_world(service)
        service.add_transaction(cycle["id"], "2026-10-02", 9000.50, "Rent", "Needs", "Cash")
        first = sweep_unspent_to_savings(db, cycle["id"], account["id"], today=date(2026, 10, 31))
        assert first["amount"] == pytest.approx(14999.50)
        with pytest.raises(ValidationError):
            sweep_unspent_to_savings(db, cycle["id"], account["id"])  # nothing new to sweep
        service.add_transaction(cycle["id"], "2026-10-05", 1000, "Fare", "Needs", "Cash")
        with pytest.raises(ValidationError):  # spending after a sweep never claws money back
            sweep_unspent_to_savings(db, cycle["id"], account["id"])
        service.delete_transaction(first["transaction"]["id"])
        assert db.get(SAVINGS_ACCOUNTS, account["id"])["current_balance"] == 1000
        again = sweep_unspent_to_savings(db, cycle["id"], account["id"], today=date(2026, 10, 31))
        assert again["amount"] == pytest.approx(24000 - 9000.50 - 1000)


def test_second_sweep_only_moves_newly_freed_capacity():
    with tempfile.TemporaryDirectory() as tmp:
        service, db = make_service(tmp)
        cycle, account, _ = setup_world(service)
        rent = service.add_transaction(cycle["id"], "2026-10-02", 9000.50, "Rent", "Needs", "Cash")
        sweep_unspent_to_savings(db, cycle["id"], account["id"], today=date(2026, 10, 31))
        service.delete_transaction(rent["id"])  # frees 9,000.50 of Needs capacity
        second = sweep_unspent_to_savings(db, cycle["id"], account["id"], today=date(2026, 10, 31))
        assert second["amount"] == pytest.approx(9000.50) and second["already_swept"] == pytest.approx(14999.50)
        assert "already swept" in second["transaction"]["notes"]
        assert db.get(SAVINGS_ACCOUNTS, account["id"])["current_balance"] == pytest.approx(25000)


def test_sweep_validation_changes_nothing():
    with tempfile.TemporaryDirectory() as tmp:
        service, db = make_service(tmp)
        cycle, account, _ = setup_world(service)
        service.add_transaction(cycle["id"], "2026-10-02", 16000, "Big", "Needs", "Cash")
        service.add_transaction(cycle["id"], "2026-10-02", 9500, "Big", "Wants", "Cash")  # both overspent
        before = len(service.list_transactions(cycle["id"]))
        for cid, target in (("no-cycle", account["id"]), (cycle["id"], None), (cycle["id"], "no-account"),
                            (cycle["id"], account["id"])):  # the last one has nothing to sweep
            with pytest.raises(ValidationError):
                sweep_unspent_to_savings(db, cid, target)
        assert len(service.list_transactions(cycle["id"])) == before
        assert db.get(SAVINGS_ACCOUNTS, account["id"])["current_balance"] == 1000


def test_sweep_date_is_clamped_into_the_cycle():
    for today, expected in ((date(2026, 9, 1), "2026-10-01"), (date(2026, 12, 5), "2026-10-31"),
                            (date(2026, 10, 15), "2026-10-15")):
        with tempfile.TemporaryDirectory() as tmp:
            service, db = make_service(tmp)
            cycle, account, _ = setup_world(service)
            assert sweep_unspent_to_savings(db, cycle["id"], account["id"], today)["transaction"]["date"] == expected


# ------------------------------------------------------- split transactions
def test_split_transaction_creates_linked_documents_that_share_notes_and_timestamp():
    with tempfile.TemporaryDirectory() as tmp:
        service, db = make_service(tmp)
        cycle, _, _ = setup_world(service)
        result = service.log_split_transaction(
            cycle["id"], "2026-10-05", "Maya",
            [{"category": "Groceries", "bucket": "Needs", "amount": 2000},
             {"category": "Steam Game", "bucket": "Wants", "amount": 1500}],
            total_amount=3500, notes="Mall run #errands")
        gid = result["split_id"]
        stored = sorted(service.list_transactions(cycle["id"]), key=lambda t: t["notes"])
        assert result["total"] == 3500 and len(stored) == 2
        assert [t["notes"] for t in stored] == [f"Mall run #errands | split {gid} 1/2",
                                                f"Mall run #errands | split {gid} 2/2"]
        assert len({t["created_at"] for t in stored}) == 1
        assert {(t["category"], t["bucket"], t["amount"]) for t in stored} == {
            ("Groceries", "Needs", 2000), ("Steam Game", "Wants", 1500)}
        assert collect_tags(stored) == {"errands": 2}
        spent = service.cycle_summary(db.get("income_cycles", cycle["id"]))["spent"]
        assert spent["Needs"] == 2000 and spent["Wants"] == 1500


def test_split_total_must_match_and_failure_writes_nothing():
    with tempfile.TemporaryDirectory() as tmp:
        service, db = make_service(tmp)
        cycle, _, _ = setup_world(service)
        parts = [{"category": "A", "bucket": "Needs", "amount": 2000}, {"category": "B", "bucket": "Wants", "amount": 1400}]
        with pytest.raises(ValidationError):
            service.log_split_transaction(cycle["id"], "2026-10-05", "Cash", parts, total_amount=3500)
        assert db.query(TRANSACTIONS) == []
        ok = service.log_split_transaction(cycle["id"], "2026-10-05", "Cash", parts)  # no total: use the sum
        assert ok["total"] == 3400


def test_split_routes_savings_and_debt_in_one_atomic_batch():
    with tempfile.TemporaryDirectory() as tmp:
        service, db = make_service(tmp)
        cycle, account, debt = setup_world(service)
        db.logger.clear()
        service.log_split_transaction(cycle["id"], "2026-10-05", "GoTyme", [
            {"category": "Save", "bucket": "Savings", "amount": 500, "target_account_id": account["id"]},
            {"category": "Save more", "bucket": "Savings", "amount": 250, "target_account_id": account["id"]},
            {"category": "Card", "bucket": "Debt", "amount": 300, "target_account_id": debt["id"]},
            {"category": "Lunch", "bucket": "Needs", "amount": 100}], total_amount=1150)
        assert db.get(SAVINGS_ACCOUNTS, account["id"])["current_balance"] == 1750  # both rows accumulate
        assert db.get(DEBTS, debt["id"])["remaining_balance"] == 7700
        batches = [e for e in db.logger.recent() if e["operation"].startswith("BATCH")]
        assert len(batches) == 1 and batches[0]["result_count"] == 7  # 4 documents + 3 balance updates


def test_split_is_all_or_nothing():
    with tempfile.TemporaryDirectory() as tmp:
        service, db = make_service(tmp)
        cycle, account, debt = setup_world(service)
        with pytest.raises(ValidationError) as err:
            service.log_split_transaction(cycle["id"], "2026-10-05", "Cash", [
                {"category": "Save", "bucket": "Savings", "amount": 500, "target_account_id": account["id"]},
                {"category": "Lunch", "bucket": "Needs", "amount": 100},
                {"category": "Card", "bucket": "Debt", "amount": 9000, "target_account_id": debt["id"]}])
        assert "Row 3" in str(err.value) and "exceeds" in str(err.value)
        assert db.query(TRANSACTIONS) == []
        assert db.get(SAVINGS_ACCOUNTS, account["id"])["current_balance"] == 1000
        assert db.get(DEBTS, debt["id"])["remaining_balance"] == 8000


def test_split_validation_errors_name_the_row():
    with tempfile.TemporaryDirectory() as tmp:
        service, db = make_service(tmp)
        cycle, account, _ = setup_world(service)
        good = {"category": "A", "bucket": "Needs", "amount": 10}
        cases = [
            ([good], "at least two"),
            ([good, {"category": "B", "bucket": "Nonsense", "amount": 10}], "Row 2"),
            ([{"category": "B", "bucket": "Savings", "amount": 10}, good], "Row 1"),
            ([good, {"category": "B", "bucket": "Savings", "amount": 10, "target_account_id": "ghost"}], "Row 2"),
            ([good, {"category": "B", "bucket": "Wants", "amount": -5}], "Row 2"),
            ([good, {"category": "", "bucket": "Wants", "amount": 5}], "Row 2"),
            ([good] * 51, "at most 50"),
        ]
        for parts, fragment in cases:
            with pytest.raises(ValidationError) as err:
                service.log_split_transaction(cycle["id"], "2026-10-05", "Cash", parts)
            assert fragment in str(err.value), (fragment, str(err.value))
        for bad_total in (0, "abc", -1):
            with pytest.raises(ValidationError):
                service.log_split_transaction(cycle["id"], "2026-10-05", "Cash", [good, good], total_amount=bad_total)
        with pytest.raises(ValidationError):
            service.log_split_transaction("no-cycle", "2026-10-05", "Cash", [good, good])
        assert db.query(TRANSACTIONS) == []
        parts = [SplitPart(category="A", bucket="Needs", amount=10), SplitPart(category="B", bucket="Wants", amount=5)]
        assert service.log_split_transaction(cycle["id"], "2026-10-05", "Cash", parts, 15)["total"] == 15


# --------------------------------------------------------------- CSV parsing
def test_parse_amount_handles_bank_formats():
    good = {"1,250.00": 1250, "-1250": -1250, "(1,250.50)": -1250.5, "₱1,250.00": 1250, "PHP 99.5": 99.5,
            "1250 DR": -1250, "1250 CR": 1250, "500-": -500, "+75": 75, " 42 ": 42}
    for text, expected in good.items():
        assert parse_amount(text) == pytest.approx(expected), text
    for bad in ("", "abc", "1,2,x", None, "nan", "inf"):
        with pytest.raises(ValidationError):
            parse_amount(bad)


def test_parse_date_text_formats_and_day_first():
    for text in ("2026-10-26", "10/26/2026", "26/10/2026", "Oct 26, 2026", "26 Oct 2026", "10/26/26",
                 "2026-10-26 14:03:00", "2026-10-26T14:03:00", "10/26/2026 14:03"):
        assert parse_date_text(text) == date(2026, 10, 26), text
    assert parse_date_text("03/04/2026") == date(2026, 3, 4)  # month first by default
    assert parse_date_text("03/04/2026", dayfirst=True) == date(2026, 4, 3)
    for bad in ("", "not a date", "13/13/2026", None):
        with pytest.raises(ValidationError):
            parse_date_text(bad)


STANDARD_CSV = """Date,Amount,Description
2026-10-27,-1250.00,MERALCO PAYMENT
2026-10-28,-549.00,Netflix
2026-10-30,25000.00,Salary credit

2026-10-31,-300.00,Random store
"""


def test_parse_bank_csv_standard_rows_with_bucket_suggestions():
    parsed = parse_bank_csv(STANDARD_CSV)
    rows = parsed["rows"]
    assert parsed["warnings"] == [] and len(rows) == 4  # the blank line is skipped
    assert [r["amount"] for r in rows] == [1250, 549, 25000, 300]  # amounts are stored positive
    assert [r["date"] for r in rows][0] == date(2026, 10, 27)
    assert [r["bucket"] for r in rows] == ["Needs", "Wants", "Income", "Wants"]
    assert rows[0]["description"] == "MERALCO PAYMENT" and rows[0]["category"] == "MERALCO PAYMENT"
    assert all(r["payment_channel"] == "" for r in rows)


def test_parse_bank_csv_accepts_bytes_bom_crlf_and_other_layouts():
    raw = ("\ufeff" + STANDARD_CSV.replace("\n", "\r\n")).encode("utf-8")
    assert len(parse_bank_csv(raw)["rows"]) == 4
    all_positive = parse_bank_csv("Date,Amount,Description\n2026-10-01,5000,Payroll\n2026-10-02,300,Mystery\n")
    assert [r["bucket"] for r in all_positive["rows"]] == ["Income", "Wants"]  # no negatives: keywords decide
    split_cols = parse_bank_csv("Transaction Date,Details,Debit,Credit\n10/05/2026,Rent,\"12,000.00\",\n"
                                "10/06/2026,Refund,,500.00\n")
    assert [(r["amount"], r["bucket"]) for r in split_cols["rows"]] == [(12000, "Needs"), (500, "Income")]
    day_first = parse_bank_csv("Date,Amount,Description\n03/04/2026,-10,x\n", dayfirst=True)
    assert day_first["rows"][0]["date"] == date(2026, 4, 3)


def test_parse_bank_csv_reports_bad_rows_and_missing_columns():
    parsed = parse_bank_csv("Date,Amount,Description\n2026-10-01,-10,ok\nnot a date,-20,bad date\n2026-10-03,abc,bad amount\n")
    assert [(r["date"] is None, r["amount"] is None) for r in parsed["rows"]] == [(False, False), (True, False), (False, True)]
    assert len(parsed["warnings"]) == 2 and "Line 3" in parsed["warnings"][0] and "Line 4" in parsed["warnings"][1]
    for text, fragment in (("Amount,Description\n1,x\n", "Date column"), ("Date,Amount\n2026-10-01,5\n", "Description column"),
                           ("Date,Description\n2026-10-01,x\n", "Amount"), ("", "empty"),
                           ("Date,Amount,Description\n", "No transaction rows")):
        with pytest.raises(ValidationError) as err:
            parse_bank_csv(text)
        assert fragment in str(err.value), (fragment, str(err.value))


def test_csv_export_round_trips_and_neutralises_formulas():
    with tempfile.TemporaryDirectory() as tmp:
        service, db = make_service(tmp)
        cycle, account, _ = setup_world(service)
        service.add_transaction(cycle["id"], "2026-10-05", 2000, "Groceries", "Needs", "Maya", notes="weekly #food")
        service.add_transaction(cycle["id"], "2026-10-02", 500, "-promo", "Wants", "GoTyme", notes="=SUM(A1)")
        service.add_transaction(cycle["id"], "2026-10-03", 750.5, "Save", "Savings", "Cash", target_account_id=account["id"])
        txs = service.list_transactions(cycle["id"])
        text = transactions_to_csv(txs, {account["id"]: "GoTyme"})
        lines = text.strip().split("\n")
        assert lines[0] == "Date,Amount,Description,Bucket,Category,Payment Channel,Target,ID"
        assert [l.split(",")[0] for l in lines[1:]] == ["2026-10-02", "2026-10-03", "2026-10-05"]  # oldest first
        assert "'=SUM(A1)" in text and "'-promo" in text and ",750.50," in text and ",GoTyme," in text
        back = parse_bank_csv(text)["rows"]
        assert [(r["date"].isoformat(), r["amount"], r["bucket"], r["category"], r["payment_channel"]) for r in back] == [
            ("2026-10-02", 500, "Wants", "-promo", "GoTyme"), ("2026-10-03", 750.5, "Savings", "Save", "Cash"),
            ("2026-10-05", 2000, "Needs", "Groceries", "Maya")]
        assert back[0]["description"] == "=SUM(A1)"
        assert transactions_to_csv([]).strip() == "Date,Amount,Description,Bucket,Category,Payment Channel,Target,ID"


def test_find_possible_duplicates_matches_date_and_amount():
    existing = [{"date": "2026-10-27", "amount": 549.0}]
    rows = [{"date": date(2026, 10, 27), "amount": 549}, {"date": date(2026, 10, 28), "amount": 549},
            {"date": date(2026, 10, 27), "amount": None}, {"date": None, "amount": 549}]
    assert find_possible_duplicates(rows, existing) == [True, False, False, False]


# -------------------------------------------------------------- CSV import
def _row(**kw):
    base = {"date": date(2026, 10, 5), "amount": 100, "category": "Misc", "bucket": "Needs",
            "payment_channel": "", "target_account_id": None, "notes": "from bank"}
    return {**base, **kw}


def test_import_transactions_commits_one_batch_with_correct_balances():
    with tempfile.TemporaryDirectory() as tmp:
        service, db = make_service(tmp)
        cycle, account, debt = setup_world(service)
        db.logger.clear()
        result = service.import_transactions(cycle["id"], [
            _row(amount=500), _row(amount=200, bucket="Savings", target_account_id=account["id"]),
            _row(amount=300, bucket="Savings", target_account_id=account["id"]),
            _row(amount=400, bucket="Debt", target_account_id=debt["id"], payment_channel="GoTyme"),
            _row(amount=1000, bucket="Income", target_account_id="ignored")], default_channel="Maya")
        assert result["imported"] == 5 and result["total"] == 2400
        assert db.get(SAVINGS_ACCOUNTS, account["id"])["current_balance"] == 1500
        assert db.get(DEBTS, debt["id"])["remaining_balance"] == 7600
        stored = service.list_transactions(cycle["id"])
        assert {t["payment_channel"] for t in stored} == {"Maya", "GoTyme"} and all(t["notes"] == "from bank" for t in stored)
        assert [t for t in stored if t["bucket"] == "Income"][0]["target_account_id"] is None
        assert len([e for e in db.logger.recent() if e["operation"].startswith("BATCH")]) == 1


def test_import_transactions_chunks_batches_without_splitting_a_transaction():
    with tempfile.TemporaryDirectory() as tmp:
        service, db = make_service(tmp)
        cycle, account, _ = setup_world(service)
        rows = [_row(amount=100, bucket="Savings", target_account_id=account["id"]) for _ in range(12)]
        db.logger.clear()
        result = service.import_transactions(cycle["id"], rows, chunk_ops=10)  # 2 ops per row -> 5 rows per batch
        batches = [e for e in db.logger.recent() if e["operation"].startswith("BATCH")]
        assert result["imported"] == 12 and len(batches) == 3 and max(b["result_count"] for b in batches) <= 10
        assert db.get(SAVINGS_ACCOUNTS, account["id"])["current_balance"] == 2200
        assert len(db.query(TRANSACTIONS)) == 12


def test_import_transactions_rejects_the_whole_file_if_any_row_is_bad():
    with tempfile.TemporaryDirectory() as tmp:
        service, db = make_service(tmp)
        cycle, account, debt = setup_world(service)
        with pytest.raises(ValidationError) as err:
            service.import_transactions(cycle["id"], [
                _row(), _row(amount=0), _row(date=None), _row(bucket="Savings"),
                _row(amount=9000, bucket="Debt", target_account_id=debt["id"]), _row(category="")])
        message = str(err.value)
        assert all(f"Row {n}" in message for n in (2, 3, 4, 5, 6)) and "Row 1" not in message
        assert db.query(TRANSACTIONS) == []
        assert db.get(SAVINGS_ACCOUNTS, account["id"])["current_balance"] == 1000
        assert db.get(DEBTS, debt["id"])["remaining_balance"] == 8000
        with pytest.raises(ValidationError) as many:
            service.import_transactions(cycle["id"], [_row(amount=0)] * 12)
        assert "+2 more" in str(many.value)
        for bad_rows, cid in (([], cycle["id"]), ([_row()], "no-cycle")):
            with pytest.raises(ValidationError):
                service.import_transactions(cid, bad_rows)


def test_csv_import_end_to_end_flags_a_second_import_as_duplicates():
    with tempfile.TemporaryDirectory() as tmp:
        service, db = make_service(tmp)
        cycle, _, _ = setup_world(service)
        rows = parse_bank_csv(STANDARD_CSV)["rows"]
        assert find_possible_duplicates(rows, service.list_transactions(cycle["id"])) == [False] * 4
        service.import_transactions(cycle["id"], [{**r, "notes": r["description"]} for r in rows], "GoTyme")
        assert find_possible_duplicates(rows, service.list_transactions(cycle["id"])) == [True] * 4
        assert collect_tags(service.list_transactions(cycle["id"])) == {}


def test_ui_state_prefixes_cover_the_v1_2_widgets():
    state = {"split_total_0": 1, "split_gen": 2, "imp_file_0": 3, "imp_gen": 1, "ana_tags": [], "sweep_account": "x",
             "tx_split": True, "keep_me": 1}
    assert sorted(clear_ui_state(state)) == ["ana_tags", "imp_file_0", "imp_gen", "split_gen", "split_total_0",
                                             "sweep_account", "tx_split"]
    assert state == {"keep_me": 1}
