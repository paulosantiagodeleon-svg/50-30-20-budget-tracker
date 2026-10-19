"""50/30/20 Budget, Debt & Savings Tracker - Streamlit UI entry point.

Run with:  streamlit run app.py
"""
from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

from src.firebase_client import COLLECTIONS, QUERY_LOG, get_database
from src.models import LOG_BUCKETS, PAYMENT_CHANNELS, ValidationError
from src.services import FREQUENCIES, BudgetService, calculate_split, debt_progress, default_end_date

st.set_page_config(page_title="50/30/20 Budget Tracker", page_icon="💰", layout="wide")

PAGES = ("📊 Dashboard", "🧾 Transactions", "🏦 Savings", "💳 Debts", "🛠️ Developer")
CATEGORY_HINTS = {
    "Needs": "Rent, Groceries, Utilities, Transportation, Insurance",
    "Wants": "Dining out, Shopping, Subscriptions, Entertainment",
    "Savings": "Monthly savings, Emergency fund, Goal top-up",
    "Debt": "Credit card payment, Loan payment",
    "Income": "Freelance, Bonus, Refund",
}


@st.cache_resource(show_spinner=False)
def get_backend():
    return get_database()


db, backend_note = get_backend()
service = BudgetService(db)


def peso(amount: float) -> str:
    return f"₱{amount:,.2f}"


def flash(message: str) -> None:
    """Show a toast after the next rerun."""
    st.session_state["_flash"] = message


def run_action(action, success: str) -> bool:
    """Run a service call, surfacing validation problems as friendly errors."""
    try:
        action()
    except (ValidationError, KeyError) as exc:
        st.error(str(exc).strip("'\""))
        return False
    flash(success)
    st.rerun()
    return True


if "_flash" in st.session_state:
    st.toast(st.session_state.pop("_flash"))


# ------------------------------------------------------------------ sidebar
st.sidebar.title("💰 50/30/20 Tracker")
if db.is_live:
    st.sidebar.success("🟢 Live Cloud Firestore")
else:
    st.sidebar.info("🟡 Local mock database")
st.sidebar.caption(backend_note)

page = st.sidebar.radio("Navigate", PAGES, label_visibility="collapsed")
st.sidebar.divider()

active_cycle = service.get_active_cycle()
if active_cycle:
    st.sidebar.markdown(f"**Active cycle**  \n{active_cycle['cycle_name']}  \n"
                        f"{active_cycle['start_date']} → {active_cycle['end_date']}")

if st.sidebar.button("🌱 Seed Sample PHP Data"):
    counts = service.seed_sample_data()
    flash(f"Seeded {counts['transactions']} transactions, {counts['savings_accounts']} savings "
          f"accounts and {counts['debts']} debts. Sample cycle is now active.")
    st.rerun()
st.sidebar.caption("Sample docs use ids starting with `seed-`; re-seeding resets only those.")

if db.supports_reset:
    with st.sidebar.expander("Danger zone"):
        confirm = st.checkbox("I want to wipe the local mock database")
        if st.button("Reset local mock data", disabled=not confirm):
            db.reset()
            flash("Local mock database cleared.")
            st.rerun()


# --------------------------------------------------------------- dashboard
def render_new_cycle_form(expanded: bool) -> None:
    with st.expander("➕ Start a new pay cycle", expanded=expanded):
        c1, c2, c3 = st.columns(3)
        name = c1.text_input("Cycle name", value=f"{date.today():%B %Y}")
        frequency = c2.radio("Pay frequency", FREQUENCIES, horizontal=True)
        salary = c3.number_input("Net salary per cycle (₱)", min_value=0.0, step=500.0, format="%.2f")
        start = c1.date_input("Start date", value=date.today().replace(day=1))
        end = default_end_date(start, frequency)
        c2.metric("End date", f"{end:%b %d, %Y}")
        if salary > 0:
            caps = calculate_split(salary)
            c3.caption(f"Needs {peso(caps['needs_cap'])} · Wants {peso(caps['wants_cap'])} · "
                       f"Savings & Debt {peso(caps['savings_debt_cap'])}")
        if st.button("Create & activate cycle", type="primary"):
            run_action(lambda: service.create_cycle(name, salary, start, end, frequency),
                       f"Cycle '{name}' created.")


def page_dashboard() -> None:
    st.header("Dashboard")
    cycle = active_cycle
    if cycle is None:
        st.info("No active pay cycle yet. Create one below, or click **Seed Sample PHP Data** in the sidebar.")
        render_new_cycle_form(expanded=True)
        return

    summary = service.cycle_summary(cycle)
    st.subheader(f"{cycle['cycle_name']}  ·  {cycle['start_date']} → {cycle['end_date']}")
    top = st.columns(3)
    top[0].metric("Net salary", peso(cycle["net_salary"]))
    top[1].metric("Allocated so far", peso(summary["total_allocated"]))
    top[2].metric("Unallocated", peso(summary["unallocated"]))

    labels = {"Needs": "50% Needs", "Wants": "30% Wants", "Savings & Debt": "20% Savings & Debt"}
    cols = st.columns(3)
    for col, (key, line) in zip(cols, summary["lines"].items()):
        with col:
            st.metric(labels[key], f"{peso(line['spent'])} / {peso(line['cap'])}",
                      delta=(f"-{peso(-line['remaining'])} over cap" if line["over"]
                             else f"{peso(line['remaining'])} left"))
            st.progress(min(line["pct"], 1.0), text=f"{line['pct'] * 100:.0f}% of cap used")
            if line["over"]:
                st.warning(f"Over the {key} cap by {peso(-line['remaining'])}.")
    if summary["other_income"]:
        st.caption(f"Extra income logged this cycle: {peso(summary['other_income'])}")

    st.subheader("Recent transactions")
    recent = service.list_transactions(cycle_id=cycle["id"])[:8]
    if recent:
        st.dataframe(pd.DataFrame(recent)[["date", "bucket", "category", "amount", "payment_channel"]],
                     hide_index=True)
    else:
        st.caption("Nothing logged yet - head to the Transactions page.")

    cycles = service.list_cycles()
    if len(cycles) > 1:
        with st.expander("Switch pay cycle"):
            options = {c["id"]: f"{c['cycle_name']} ({c['start_date']})" for c in cycles}
            choice = st.selectbox("Cycle", list(options), format_func=options.get,
                                  index=list(options).index(cycle["id"]))
            if st.button("Make active") and choice != cycle["id"]:
                run_action(lambda: service.activate_cycle(choice), "Cycle switched.")
    render_new_cycle_form(expanded=False)


# ------------------------------------------------------------ transactions
def page_transactions() -> None:
    st.header("Transactions")
    cycle = active_cycle
    if cycle is None:
        st.info("Create a pay cycle on the Dashboard first.")
        return

    accounts = service.list_savings_accounts()
    debts = [d for d in service.list_debts() if d["status"] == "active"]

    st.subheader("Log a transaction")
    bucket = st.radio("Bucket", LOG_BUCKETS, horizontal=True)
    st.caption(f"Category ideas: {CATEGORY_HINTS[bucket]}")
    with st.form("tx_form", clear_on_submit=True):
        c1, c2 = st.columns(2)
        tx_date = c1.date_input("Date", value=date.today())
        amount = c2.number_input("Amount (₱)", min_value=0.0, step=50.0, format="%.2f")
        category = c1.text_input("Category")
        channel = c2.selectbox("Payment channel", PAYMENT_CHANNELS)
        target = None
        if bucket == "Savings":
            labels = {a["id"]: f"{a['bank_name']} - {a['goal_name'] or 'Savings'} ({peso(a['current_balance'])})"
                      for a in accounts}
            target = st.selectbox("Deposit into", list(labels), format_func=labels.get) if labels else None
            if not labels:
                st.warning("Add a savings account on the Savings page first.")
        elif bucket == "Debt":
            labels = {d["id"]: f"{d['name']} (remaining {peso(d['remaining_balance'])})" for d in debts}
            target = st.selectbox("Pay toward", list(labels), format_func=labels.get) if labels else None
            if not labels:
                st.warning("No active debts. Add one on the Debts page first.")
        notes = st.text_input("Notes (optional)")
        submitted = st.form_submit_button("Add transaction", type="primary")
    if submitted:
        if not (cycle["start_date"] <= tx_date.isoformat() <= cycle["end_date"]):
            st.warning("That date is outside the active pay cycle, so it is saved but won't appear on the "
                       "current cycle's view of the same range.")
        run_action(lambda: service.add_transaction(cycle["id"], tx_date, amount, category, bucket,
                                                   channel, target, notes),
                   f"Logged {peso(amount)} under {bucket}.")

    st.subheader("History")
    f1, f2 = st.columns(2)
    bucket_filter = f1.multiselect("Filter by bucket", LOG_BUCKETS)
    channel_filter = f2.multiselect("Filter by payment channel", PAYMENT_CHANNELS)
    txs = service.list_transactions(cycle_id=cycle["id"])
    txs = [t for t in txs
           if (not bucket_filter or t["bucket"] in bucket_filter)
           and (not channel_filter or t["payment_channel"] in channel_filter)]
    if not txs:
        st.caption("No transactions match.")
        return
    names = {a["id"]: a["bank_name"] for a in accounts}
    names.update({d["id"]: d["name"] for d in service.list_debts()})
    df = pd.DataFrame(txs)
    df["target"] = df["target_account_id"].map(lambda i: names.get(i, "") if i else "")
    st.dataframe(df[["date", "bucket", "category", "amount", "payment_channel", "target", "notes"]],
                 hide_index=True)
    st.caption(f"{len(txs)} transaction(s) · total {peso(sum(t['amount'] for t in txs))}")

    st.markdown("**Delete a transaction** (savings and debt balances are reversed automatically)")
    labels = {t["id"]: f"{t['date']} · {t['bucket']} · {t['category']} · {peso(t['amount'])}" for t in txs}
    victim = st.selectbox("Transaction", list(labels), format_func=labels.get, label_visibility="collapsed")
    if st.button("🗑️ Delete selected"):
        run_action(lambda: service.delete_transaction(victim), "Transaction deleted.")


# ----------------------------------------------------------------- savings
def page_savings() -> None:
    st.header("Savings & Digital Bank Routing")
    accounts = service.list_savings_accounts()
    st.metric("Total savings", peso(sum(a["current_balance"] for a in accounts)))
    if accounts:
        cols = st.columns(min(len(accounts), 4))
        for i, acct in enumerate(accounts):
            with cols[i % len(cols)].container(border=True):
                st.markdown(f"**{acct['bank_name']}**")
                st.caption(acct["goal_name"] or "Savings")
                st.subheader(peso(acct["current_balance"]))
                st.caption(f"{acct['interest_rate']:.2f}% p.a.")
    else:
        st.info("No savings accounts yet - add one below.")
    st.caption("Log a transaction in the **Savings** bucket to deposit into an account automatically.")

    with st.form("savings_form", clear_on_submit=True):
        st.subheader("Add a savings account")
        c1, c2 = st.columns(2)
        bank = c1.text_input("Bank / wallet", placeholder="GoTyme, MariBank, Maya, Emergency Fund...")
        goal = c2.text_input("Goal name", placeholder="Go Save, Travel fund...")
        balance = c1.number_input("Starting balance (₱)", min_value=0.0, step=100.0, format="%.2f")
        rate = c2.number_input("Interest rate (% p.a.)", min_value=0.0, max_value=100.0, step=0.25, format="%.2f")
        if st.form_submit_button("Add account", type="primary"):
            run_action(lambda: service.add_savings_account(bank, goal, balance, rate),
                       f"Added {bank}.")


# ------------------------------------------------------------------- debts
def page_debts() -> None:
    st.header("Debt Payoff Tracker")
    debts = service.list_debts()
    c1, c2 = st.columns(2)
    c1.metric("Total remaining", peso(sum(d["remaining_balance"] for d in debts)))
    c2.metric("Minimum payments / cycle", peso(sum(d["minimum_payment"] for d in debts if d["status"] == "active")))

    for debt in debts:
        progress = debt_progress(debt["total_amount"], debt["remaining_balance"])
        with st.container(border=True):
            h1, h2 = st.columns([3, 1])
            h1.markdown(f"**{debt['name']}**  ·  {'✅ Paid off' if debt['status'] == 'paid' else 'Active'}")
            h2.caption(f"Due {debt['due_date']}")
            st.progress(progress, text=f"{progress * 100:.0f}% paid · {peso(debt['remaining_balance'])} "
                                       f"remaining of {peso(debt['total_amount'])}")
            st.caption(f"Minimum payment: {peso(debt['minimum_payment'])}")
    if not debts:
        st.info("No debts tracked yet - add one below.")
    st.caption("Log a transaction in the **Debt** bucket to pay down a liability automatically.")

    with st.form("debt_form", clear_on_submit=True):
        st.subheader("Add a liability")
        d1, d2 = st.columns(2)
        name = d1.text_input("Debt name", placeholder="Credit card, Salary loan...")
        due = d2.date_input("Next due date", value=date.today())
        total = d1.number_input("Total amount (₱)", min_value=0.0, step=500.0, format="%.2f")
        remaining = d2.number_input("Remaining balance (₱)", min_value=0.0, step=500.0, format="%.2f",
                                    help="Leave at 0 only if already paid; must not exceed the total.")
        minimum = d1.number_input("Minimum payment (₱)", min_value=0.0, step=100.0, format="%.2f")
        if st.form_submit_button("Add debt", type="primary"):
            run_action(lambda: service.add_debt(name, total, minimum, due, remaining_balance=remaining),
                       f"Added {name}.")


# --------------------------------------------------------------- developer
def page_developer() -> None:
    st.header("Firestore Query & Payload Inspector")
    st.caption(f"Backend: **{db.backend_name}** - {backend_note}")
    log_tab, docs_tab = st.tabs(["📜 Query log", "🗄️ Live documents"])

    with log_tab:
        c1, c2 = st.columns([3, 1])
        limit = c1.slider("Entries to show", 5, 100, 25)
        if c2.button("Clear log"):
            QUERY_LOG.clear()
            st.rerun()
        entries = QUERY_LOG.recent(limit)
        if not entries:
            st.info("No operations logged yet. Use the app and they will appear here.")
        for i, entry in enumerate(entries):
            count = "" if entry["result_count"] is None else f" · {entry['result_count']} doc(s)"
            with st.expander(f"{entry['time']} · {entry['operation']} · {entry['collection']}{count}",
                             expanded=i < 2):
                st.code(entry["query"], language="python")
                if entry["payload"] is not None:
                    st.json(entry["payload"], expanded=False)

    with docs_tab:
        st.button("🔄 Refresh")
        snapshot = db.snapshot()
        for name in COLLECTIONS:
            docs = snapshot.get(name, [])
            st.subheader(f"{name}  ({len(docs)})")
            st.json(docs, expanded=False)


{"📊 Dashboard": page_dashboard, "🧾 Transactions": page_transactions, "🏦 Savings": page_savings,
 "💳 Debts": page_debts, "🛠️ Developer": page_developer}[page]()
