"""50/30/20 Budget, Debt & Savings Tracker - Streamlit UI entry point.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import re
from datetime import date

import pandas as pd
import streamlit as st

from src.firebase_client import QUERY_LOG, get_database
from src.models import BUCKETS, LOG_BUCKETS, PAYMENT_CHANNELS, SPEND_BUCKETS, ValidationError
from src.services import (
    ALL_COLLECTIONS,
    FREQUENCIES,
    BudgetService,
    asset_allocation,
    calculate_interest_projection,
    calculate_split,
    clear_ui_state,
    collect_tags,
    daily_spend_series,
    debt_progress,
    default_end_date,
    filter_by_tags,
    find_possible_duplicates,
    next_due_date,
    pacing_summary,
    parse_bank_csv,
    portfolio_interest_projection,
    tag_totals,
    top_up_cycle_salary,
    transactions_to_csv,
    unspent_capacity,
)

st.set_page_config(page_title="50/30/20 Budget Tracker", page_icon="💰", layout="wide")

PAGES = ("📊 Dashboard", "🧾 Transactions", "🔁 Recurring Bills", "📈 Analytics", "🏦 Savings", "💳 Debts",
         "🛠️ Developer")
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


def notify(level: str, message: str) -> None:
    """Queue an st.error / st.warning to display after the next rerun."""
    st.session_state.setdefault("_notices", []).append((level, message))


if "_flash" in st.session_state:
    st.toast(st.session_state.pop("_flash"))
for _level, _message in st.session_state.pop("_notices", []):
    getattr(st, _level)(_message)


WIPE_PHRASE = "WIPE"


def _wipe_everything() -> None:
    """Button callback. Streamlit reruns the script right after a callback finishes (an explicit
    st.rerun() inside a callback is a no-op), so the app comes back as a true empty slate."""
    if st.session_state.get("wipe_confirm_text", "").strip() != WIPE_PHRASE:
        notify("error", f"Type {WIPE_PHRASE} to confirm.")
        return
    try:
        counts = service.wipe_all_data()
    except Exception as exc:  # noqa: BLE001 - surface any backend failure instead of crashing the app
        notify("error", f"Wipe failed: {exc}. Nothing was reset in the UI; click again to retry.")
        return
    clear_ui_state(st.session_state)
    flash(f"All data wiped ({sum(counts.values())} documents deleted). Fresh slate!")


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
          f"accounts, {counts['debts']} debts and {counts['recurring_templates']} recurring bills. "
          "Sample cycle is now active.")
    st.rerun()
st.sidebar.caption("Sample docs use ids starting with `seed-`; re-seeding resets only those.")

with st.sidebar.expander("⚠️ Danger Zone"):
    where = "your LIVE Cloud Firestore database" if db.is_live else "the local mock database"
    st.warning(f"Permanently deletes every record in {where}: pay cycles, transactions, debts, savings "
               "accounts and recurring bills. This cannot be undone.")
    typed = st.text_input(f"Type {WIPE_PHRASE} and press Enter to confirm", key="wipe_confirm_text")
    st.button("🗑️ Wipe All Data (Fresh Slate)", key="wipe_btn", type="primary",
              disabled=typed.strip() != WIPE_PHRASE, on_click=_wipe_everything)


# --------------------------------------------------------------- dashboard
def _apply_top_up(cycle_id: str) -> None:
    """Button callback: runs before the rerun, so the inputs can be cleared safely."""
    amount = st.session_state.get("topup_amount", 0.0)
    try:
        top_up_cycle_salary(db, cycle_id, amount, st.session_state.get("topup_note", ""))
    except ValidationError as exc:
        notify("error", str(exc))
        return
    st.session_state["topup_amount"] = 0.0
    st.session_state["topup_note"] = ""
    flash(f"Added {peso(amount)} to the active cycle. Caps were re-split 50/30/20.")


def render_top_up_form(cycle: dict) -> None:
    with st.expander("➕ Add 15th/30th Paycheck or Top-Up Salary"):
        c1, c2 = st.columns(2)
        amount = c1.number_input("Additional net salary (₱)", min_value=0.0, step=500.0, format="%.2f",
                                 key="topup_amount")
        c2.text_input("Note (optional)", placeholder="30th paycheck, bonus...", key="topup_note")
        if amount > 0:
            new_net = round(cycle["net_salary"] + amount, 2)
            caps = calculate_split(new_net)
            st.caption(f"New net salary {peso(new_net)} → Needs {peso(caps['needs_cap'])} · "
                       f"Wants {peso(caps['wants_cap'])} · Savings & Debt {peso(caps['savings_debt_cap'])}")
        st.button("Add to active cycle", type="primary", key="topup_btn",
                  on_click=_apply_top_up, args=(cycle["id"],))
        history = cycle.get("top_ups") or []
        if history:
            st.markdown("**Top-up history**")
            st.dataframe(pd.DataFrame(history).reindex(columns=["at", "amount", "note", "net_salary_after"]),
                         hide_index=True)


def _sync_cycle_end() -> None:
    """Quick-fill: when the start date or preset changes, reset the end date to the preset default."""
    st.session_state["new_cycle_end"] = default_end_date(
        st.session_state["new_cycle_start"], st.session_state["new_cycle_freq"])


def render_new_cycle_form(expanded: bool) -> None:
    with st.expander("➕ Start a new pay cycle", expanded=expanded):
        today = date.today()
        st.session_state.setdefault("new_cycle_name", f"{today:%B %Y}")
        st.session_state.setdefault("new_cycle_freq", FREQUENCIES[0])
        st.session_state.setdefault("new_cycle_start", today.replace(day=1))
        st.session_state.setdefault("new_cycle_end", default_end_date(
            st.session_state["new_cycle_start"], st.session_state["new_cycle_freq"]))

        c1, c2, c3 = st.columns(3)
        name = c1.text_input("Cycle name", key="new_cycle_name")
        salary = c3.number_input("Net salary per cycle (₱)", min_value=0.0, step=500.0, format="%.2f",
                                 key="new_cycle_salary")
        start = c1.date_input("Cycle start date", key="new_cycle_start", on_change=_sync_cycle_end)
        c2.radio("Quick-fill end date", FREQUENCIES, horizontal=True, key="new_cycle_freq",
                 on_change=_sync_cycle_end, help="Only pre-fills the end date; you can still pick any date.")
        end = c2.date_input("Cycle end date", key="new_cycle_end",
                            help="Any date on or after the start, e.g. Oct 26 to Nov 30 for an irregular cutoff.")
        invalid = end < start
        if invalid:
            c2.error("The end date cannot be before the start date.")
        else:
            c2.caption(f"{(end - start).days + 1}-day cycle")
        if salary > 0:
            caps = calculate_split(salary)
            c3.caption(f"Needs {peso(caps['needs_cap'])} · Wants {peso(caps['wants_cap'])} · "
                       f"Savings & Debt {peso(caps['savings_debt_cap'])}")
        if st.button("Create & activate cycle", type="primary", disabled=invalid):
            run_action(lambda: service.create_cycle(name, salary, start, end, st.session_state["new_cycle_freq"]),
                       f"Cycle '{name}' created.")


def render_edit_cycle_dates(cycle: dict) -> None:
    with st.expander("✏️ Edit cycle dates"):
        c1, c2 = st.columns(2)
        start = c1.date_input("Cycle start date", value=date.fromisoformat(cycle["start_date"]),
                              key=f"cycle_edit_start_{cycle['id']}")
        end = c2.date_input("Cycle end date", value=date.fromisoformat(cycle["end_date"]),
                            key=f"cycle_edit_end_{cycle['id']}")
        if end < start:
            st.error("The end date cannot be before the start date.")
        st.caption("Transactions belong to the cycle, not its dates, so existing entries are unaffected.")
        if st.button("Save cycle dates", key=f"cycle_edit_btn_{cycle['id']}", disabled=end < start):
            run_action(lambda: service.update_cycle_dates(cycle["id"], start, end), "Cycle dates updated.")


def render_sweep(cycle: dict) -> None:
    with st.expander("🧹 Sweep Unspent to Savings"):
        info = unspent_capacity(cycle, service.list_transactions(cycle_id=cycle["id"]))
        m1, m2, m3 = st.columns(3)
        m1.metric("Unspent Needs", peso(info["needs"]))
        m2.metric("Unspent Wants", peso(info["wants"]))
        m3.metric("Sweepable now", peso(info["sweepable"]))
        if info["already_swept"]:
            st.caption(f"Already swept this cycle: {peso(info['already_swept'])}.")
        st.caption("Moves the unspent Needs + Wants budget into ONE Savings transaction. A bucket that is over "
                   "its cap counts as 0. Sweeps are not counted against the 20% cap. To undo, delete the "
                   "'Rollover sweep' transaction and the savings balance is reversed.")
        accounts = service.list_savings_accounts()
        if not accounts:
            st.info("Add a savings account on the Savings page first.")
            return
        labels = {a["id"]: f"{a['bank_name']} - {a['goal_name'] or 'Savings'} ({peso(a['current_balance'])})"
                  for a in accounts}
        target = st.selectbox("Sweep into", list(labels), format_func=labels.get, key="sweep_account")
        confirmed = True
        if date.today().isoformat() < cycle["end_date"]:
            confirmed = st.checkbox(f"This cycle ends {cycle['end_date']}. Sweep anyway.", key="sweep_early")
        if st.button("🧹 Sweep Unspent to Savings", type="primary", key="sweep_btn",
                     disabled=info["sweepable"] <= 0 or not confirmed):
            run_action(lambda: service.sweep_unspent_to_savings(cycle["id"], target),
                       f"Swept {peso(info['sweepable'])} into savings.")


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
    if summary["swept"]:
        st.caption(f"Rolled over to savings: {peso(summary['swept'])} (not counted against the 20% cap).")
    render_top_up_form(cycle)
    render_edit_cycle_dates(cycle)
    render_sweep(cycle)

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
def _cell(value):
    """A data_editor cell as a plain value (NaN/NaT/None -> None)."""
    return None if value is None or pd.isna(value) else value


def target_options(accounts: list[dict], debts: list[dict]) -> dict[str, str]:
    """Label -> id for every savings account and active debt a row can point at."""
    options: dict[str, str] = {}
    for kind, docs in (("Savings", accounts), ("Debt", debts)):
        for doc in docs:
            name = f"{doc['bank_name']} - {doc['goal_name'] or 'Savings'}" if kind == "Savings" else doc["name"]
            label = f"{kind} · {name}"
            options[label if label not in options else f"{label} ({doc['id'][:4]})"] = doc["id"]
    return options


def resolve_target(label, bucket, options: dict[str, str]):
    """Id behind a Target label, but only if its kind matches the row's bucket."""
    label = _cell(label)
    if not label or label not in options or not str(label).startswith(f"{bucket} ·"):
        return None
    return options[label]


def render_single_tx_form(cycle: dict, accounts: list[dict], debts: list[dict]) -> None:
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


def render_split_form(cycle: dict, accounts: list[dict], debts: list[dict]) -> None:
    gen = st.session_state.setdefault("split_gen", 0)  # bumped after a save so the inputs start fresh
    options = target_options(accounts, debts)
    c1, c2 = st.columns(2)
    total = c1.number_input("Total payment amount (₱)", min_value=0.0, step=50.0, format="%.2f",
                            key=f"split_total_{gen}")
    tx_date = c2.date_input("Date", value=date.today(), key=f"split_date_{gen}")
    channel = c1.selectbox("Payment channel", PAYMENT_CHANNELS, key=f"split_channel_{gen}")
    notes = c2.text_input("Notes (optional) - hashtags like #vacation work", key=f"split_notes_{gen}")
    st.caption("Add one row per category. Savings and Debt rows also need a Target. "
               "Use the + at the bottom of the table for more rows.")
    base = pd.DataFrame({"Category": ["", ""], "Bucket": ["Needs", "Wants"], "Amount": [0.0, 0.0],
                         "Target": [None, None]})
    edited = st.data_editor(
        base, key=f"split_editor_{gen}", num_rows="dynamic", hide_index=True,
        column_config={
            "Category": st.column_config.TextColumn("Category", required=True),
            "Bucket": st.column_config.SelectboxColumn("Bucket", options=list(LOG_BUCKETS), required=True),
            "Amount": st.column_config.NumberColumn("Amount (₱)", min_value=0.0, step=50.0, format="%.2f"),
            "Target": st.column_config.SelectboxColumn("Target (Savings/Debt rows)", options=list(options)),
        })
    parts = []
    for _, row in edited.iterrows():
        category = str(_cell(row["Category"]) or "").strip()
        amount = float(_cell(row["Amount"]) or 0.0)
        if not category and amount == 0:
            continue  # ignore untouched rows
        bucket = _cell(row["Bucket"])
        parts.append({"category": category, "bucket": bucket, "amount": amount,
                      "target_account_id": resolve_target(row["Target"], bucket, options)})
    allocated = round(sum(p["amount"] for p in parts), 2)
    remaining = round(total - allocated, 2)
    if total <= 0:
        st.info("Enter the total payment amount.")
    elif remaining == 0:
        st.success(f"Fully allocated: {peso(allocated)} of {peso(total)}.")
    else:
        st.warning(f"Allocated {peso(allocated)} of {peso(total)} · "
                   f"{'remaining' if remaining > 0 else 'over by'} {peso(abs(remaining))}")
    if not (cycle["start_date"] <= tx_date.isoformat() <= cycle["end_date"]):
        st.warning("That date is outside the active pay cycle.")

    def save() -> None:
        service.log_split_transaction(cycle["id"], tx_date, channel, parts, total_amount=total, notes=notes)
        st.session_state["split_gen"] = gen + 1

    if st.button("Add split transaction", type="primary", key=f"split_btn_{gen}", disabled=not parts):
        run_action(save, f"Logged {peso(total)} split across {len(parts)} rows.")


def render_import_export(cycle: dict, accounts: list[dict]) -> None:
    with st.expander("📤 Export / 📥 Import CSV"):
        debts = [d for d in service.list_debts() if d["status"] == "active"]
        everything = service.list_transactions(cycle_id=cycle["id"])
        names = {a["id"]: a["bank_name"] for a in accounts}
        names.update({d["id"]: d["name"] for d in service.list_debts()})

        st.markdown("**Export this cycle**")
        slug = re.sub(r"[^A-Za-z0-9]+", "_", cycle["cycle_name"]).strip("_") or "cycle"
        st.download_button("⬇️ Download transactions (CSV)", key="tx_export", mime="text/csv",
                           data=transactions_to_csv(everything, names).encode("utf-8-sig"),
                           file_name=f"transactions_{slug}.csv", disabled=not everything)
        st.divider()

        st.markdown("**Import a bank CSV** (needs Date, Amount and Description columns)")
        gen = st.session_state.setdefault("imp_gen", 0)  # bumped after an import to reset the uploader
        c1, c2 = st.columns(2)
        dayfirst = c1.checkbox("Dates are day-first (DD/MM/YYYY)", key=f"imp_dayfirst_{gen}")
        default_channel = c2.selectbox("Default payment channel", PAYMENT_CHANNELS,
                                       index=PAYMENT_CHANNELS.index("Other"), key=f"imp_channel_{gen}")
        upload = st.file_uploader("Choose a CSV file", type=["csv"], key=f"imp_file_{gen}")
        if upload is None:
            return
        data = upload.getvalue()
        signature = (upload.name, len(data), dayfirst, default_channel)
        if st.session_state.get(f"imp_sig_{gen}") != signature:  # (re)parse only when the file/options change
            try:
                parsed = parse_bank_csv(data, dayfirst=dayfirst)
            except ValidationError as exc:
                st.error(str(exc))
                return
            dupes = find_possible_duplicates(parsed["rows"], everything)
            st.session_state[f"imp_df_{gen}"] = pd.DataFrame({
                "Import": [not dup and r["date"] is not None and r["amount"] is not None
                           for r, dup in zip(parsed["rows"], dupes)],
                "Date": pd.to_datetime(pd.Series([r["date"] for r in parsed["rows"]]), errors="coerce"),
                "Amount": [r["amount"] for r in parsed["rows"]],
                "Description": [r["description"] for r in parsed["rows"]],
                "Category": [r["category"] for r in parsed["rows"]],
                "Bucket": [r["bucket"] for r in parsed["rows"]],
                "Channel": [r["payment_channel"] or default_channel for r in parsed["rows"]],
                "Target": [None] * len(parsed["rows"]),
                "Possible duplicate": dupes,
            })
            st.session_state[f"imp_warn_{gen}"] = parsed["warnings"]
            st.session_state[f"imp_sig_{gen}"] = signature
            st.session_state[f"imp_ver_{gen}"] = st.session_state.get(f"imp_ver_{gen}", 0) + 1

        for warning in st.session_state.get(f"imp_warn_{gen}", []):
            st.warning(warning)
        st.caption("Map each row to a bucket below. Rows matching an existing date + amount are flagged and "
                   "start unticked. Savings and Debt rows need a Target. Bucket guesses are only suggestions.")
        options = target_options(accounts, debts)
        edited = st.data_editor(
            st.session_state[f"imp_df_{gen}"], key=f"imp_editor_{gen}_{st.session_state[f'imp_ver_{gen}']}",
            hide_index=True, num_rows="fixed", disabled=["Possible duplicate"],
            column_config={
                "Import": st.column_config.CheckboxColumn("Import"),
                "Date": st.column_config.DateColumn("Date", format="YYYY-MM-DD"),
                "Amount": st.column_config.NumberColumn("Amount (₱)", min_value=0.0, format="%.2f"),
                "Bucket": st.column_config.SelectboxColumn("Bucket", options=list(LOG_BUCKETS), required=True),
                "Channel": st.column_config.SelectboxColumn("Channel", options=list(PAYMENT_CHANNELS), required=True),
                "Target": st.column_config.SelectboxColumn("Target (Savings/Debt)", options=list(options)),
            })
        rows = []
        for _, r in edited.iterrows():
            if not r["Import"]:
                continue
            when = _cell(r["Date"])
            when = when.date() if hasattr(when, "date") else when
            description = str(_cell(r["Description"]) or "").strip()
            bucket = _cell(r["Bucket"])
            rows.append({"date": when, "amount": _cell(r["Amount"]), "bucket": bucket,
                         "category": str(_cell(r["Category"]) or "").strip() or description[:40],
                         "payment_channel": _cell(r["Channel"]), "notes": description,
                         "target_account_id": resolve_target(r["Target"], bucket, options)})
        total = sum(row["amount"] or 0 for row in rows)
        st.caption(f"{len(rows)} of {len(edited)} rows selected · total {peso(total)}")

        def commit() -> None:
            service.import_transactions(cycle["id"], rows, default_channel)
            st.session_state["imp_gen"] = gen + 1

        if st.button("Import selected rows", type="primary", key=f"imp_btn_{gen}", disabled=not rows):
            run_action(commit, f"Imported {len(rows)} transaction(s) totalling {peso(total)}.")


def page_transactions() -> None:
    st.header("Transactions")
    cycle = active_cycle
    if cycle is None:
        st.info("Create a pay cycle on the Dashboard first.")
        return

    accounts = service.list_savings_accounts()
    debts = [d for d in service.list_debts() if d["status"] == "active"]

    st.subheader("Log a transaction")
    if st.toggle("Split transaction", key="tx_split",
                 help="Divide one payment across several categories and buckets."):
        render_split_form(cycle, accounts, debts)
    else:
        render_single_tx_form(cycle, accounts, debts)
    render_import_export(cycle, accounts)

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


# --------------------------------------------------------- recurring bills
def _post_selected_bills(cycle_id: str) -> None:
    """Button callback: post ticked templates, then untick the ones that were posted."""
    chosen = [t["id"] for t in service.list_recurring_templates()
              if st.session_state.get(f"rec_chk_{t['id']}")]
    if not chosen:
        notify("warning", "Tick at least one bill first.")
        return
    try:
        result = service.post_recurring_templates(cycle_id, chosen, st.session_state["rec_post_date"])
    except ValidationError as exc:
        notify("error", str(exc))
        return
    failed_ids = {f["template_id"] for f in result["failed"]}
    for tid in chosen:
        if tid not in failed_ids:
            st.session_state[f"rec_chk_{tid}"] = False
    for failure in result["failed"]:
        notify("warning", f"{failure['name']} was not posted: {failure['error']}")
    if result["posted"]:
        total = sum(tx["amount"] for tx in result["posted"])
        flash(f"Posted {len(result['posted'])} bill(s) totalling {peso(total)} to the active cycle.")


def page_recurring() -> None:
    st.header("Recurring Bills & Subscriptions")
    cycle = active_cycle
    templates = service.list_recurring_templates()
    accounts = service.list_savings_accounts()
    all_debts = service.list_debts()
    names = {a["id"]: a["bank_name"] for a in accounts}
    names.update({d["id"]: d["name"] for d in all_debts})
    today = date.today()

    if cycle is None:
        st.info("Create a pay cycle on the Dashboard to post bills. You can still manage templates here.")
    st.subheader("Your bills")
    if not templates:
        st.info("No recurring bills yet. Add one below, or click **Seed Sample PHP Data** in the sidebar.")
    selected_total, selected_count = 0.0, 0
    for tpl in templates:
        due = next_due_date(tpl["due_day"], today)
        days = (due - today).days
        c0, c1, c2 = st.columns([0.6, 6, 1])
        ticked = c0.checkbox("Select", key=f"rec_chk_{tpl['id']}", label_visibility="collapsed")
        if ticked:
            selected_total += tpl["amount"]
            selected_count += 1
        c1.markdown(f"**{tpl['name']}**  ·  {peso(tpl['amount'])}  ·  {tpl['bucket']} / {tpl['category']}  ·  "
                    f"via {tpl['payment_channel']}")
        when = "today" if days == 0 else f"in {days} day(s)"
        target = f" · → {names.get(tpl.get('target_account_id'), 'unknown target')}" if tpl.get("target_account_id") else ""
        c1.caption(f"{'🔔 ' if days <= 7 else ''}Due day {tpl['due_day']} · next {due:%b %d} ({when}){target}")
        if c2.button("🗑️", key=f"rec_del_{tpl['id']}", help="Delete this template"):
            run_action(lambda tid=tpl["id"]: service.delete_recurring_template(tid), f"Deleted {tpl['name']}.")

    if templates and cycle is not None:
        p1, p2 = st.columns([1, 2])
        post_date = p1.date_input("Posting date", value=today, key="rec_post_date")
        if not (cycle["start_date"] <= post_date.isoformat() <= cycle["end_date"]):
            p1.warning("That date is outside the active pay cycle.")
        p2.caption(f"{selected_count} selected · {peso(selected_total)}")
        p2.button("Post Selected Bills to Active Cycle", type="primary",
                  on_click=_post_selected_bills, args=(cycle["id"],))
        st.caption("Each bill is logged like a manual transaction, so savings and debt balances update too. "
                   "Posting the same bill twice logs it twice.")

    st.subheader("Add a recurring bill")
    bucket = st.radio("Bucket", LOG_BUCKETS, horizontal=True, key="rec_bucket")
    with st.form("rec_form", clear_on_submit=True):
        c1, c2 = st.columns(2)
        name = c1.text_input("Bill name", placeholder="Meralco Electricity, Netflix, GoTyme Auto-Save...")
        amount = c2.number_input("Amount (₱)", min_value=0.0, step=50.0, format="%.2f")
        category = c1.text_input("Category", placeholder=CATEGORY_HINTS[bucket].split(",")[0])
        channel = c2.selectbox("Payment channel", PAYMENT_CHANNELS)
        due_day = c1.number_input("Due day of month", min_value=1, max_value=31, value=1, step=1)
        target = None
        if bucket == "Savings":
            labels = {a["id"]: f"{a['bank_name']} - {a['goal_name'] or 'Savings'}" for a in accounts}
            target = c2.selectbox("Deposit into", list(labels), format_func=labels.get) if labels else None
            if not labels:
                st.warning("Add a savings account on the Savings page first.")
        elif bucket == "Debt":
            labels = {d["id"]: d["name"] for d in all_debts if d["status"] == "active"}
            target = c2.selectbox("Pay toward", list(labels), format_func=labels.get) if labels else None
            if not labels:
                st.warning("No active debts. Add one on the Debts page first.")
        submitted = st.form_submit_button("Add recurring bill", type="primary")
    if submitted:
        run_action(lambda: service.create_recurring_template(name, amount, category, bucket, channel,
                                                             due_day, target),
                   f"Added recurring bill {name}.")


# --------------------------------------------------------------- analytics
def page_analytics() -> None:
    st.header("Analytics")
    try:
        import plotly.express as px
    except ImportError:
        st.error("The Analytics page needs Plotly. Install it (PowerShell) and restart the app:")
        st.code("pip install plotly", language="powershell")
        return

    st.subheader("Savings asset allocation")
    allocation = asset_allocation(service.list_savings_accounts())
    if allocation:
        fig = px.pie(pd.DataFrame(allocation), names="account", values="balance", hole=0.55)
        fig.update_traces(textinfo="percent+label",
                          hovertemplate="%{label}<br>₱%{value:,.2f} (%{percent})<extra></extra>")
        fig.update_layout(margin={"t": 10, "b": 10})
        st.plotly_chart(fig)
        st.caption(f"Total across accounts: {peso(sum(a['balance'] for a in allocation))}. "
                   "This chart shows balances, so the hashtag filter below does not apply to it.")
    else:
        st.info("No savings balances yet. Add an account on the Savings page.")

    st.subheader("Daily spend pacing")
    cycle = active_cycle
    if cycle is None:
        st.info("Create a pay cycle on the Dashboard first.")
        return
    txs = service.list_transactions(cycle_id=cycle["id"])
    tags = collect_tags(txs)
    st.session_state["ana_tags"] = [t for t in st.session_state.get("ana_tags", []) if t in tags]
    f1, f2, f3 = st.columns([3, 1, 2])
    selected = f1.multiselect("Filter by hashtag (from transaction notes)", list(tags),
                              format_func=lambda t: f"#{t} ({tags[t]})", key="ana_tags")
    match_all = f2.checkbox("Match all", key="ana_match_all", disabled=len(selected) < 2)
    buckets = f3.multiselect("Buckets", list(BUCKETS), default=list(SPEND_BUCKETS), key="ana_buckets")
    if not tags:
        st.caption("Tip: put hashtags in a transaction's notes, e.g. `GPU #pc-build`, to track them here.")
    if not buckets:
        st.info("Pick at least one bucket.")
        return

    filtered = filter_by_tags(txs, selected, match_all)
    if selected:
        joiner = " and " if match_all else " or "
        st.caption(f"Showing {len(filtered)} of {len(txs)} transactions tagged "
                   f"{joiner.join('#' + t for t in selected)}.")
    pace = pacing_summary(cycle, filtered, buckets=buckets)
    m1, m2, m3 = st.columns(3)
    if selected:
        m1.metric("Spent (filtered)", peso(pace["spent"]))
    else:
        variance = pace["variance"]
        m1.metric("Spent so far", peso(pace["spent"]),
                  delta=(f"{peso(variance)} under pace" if pace["on_track"] else f"-{peso(-variance)} over pace"))
        m2.metric("Even-pace target to date", peso(pace["target_to_date"]))
    m3.metric("Projected cycle total", peso(pace["projected_total"]),
              help="Spend so far extended at the same daily rate through the end of the cycle.")
    if pace["outside_range"]:
        st.caption(f"{pace['outside_range']} transaction(s) dated outside this cycle are not shown.")

    series = pd.DataFrame(daily_spend_series(cycle, filtered, buckets))
    fig = px.bar(series, x="date", y="amount", color="bucket", barmode="stack",
                 labels={"date": "Day", "amount": "Spent", "bucket": "Bucket"})
    if not selected:
        fig.add_hline(y=pace["target_per_day"], line_dash="dash", annotation_position="top left",
                      annotation_text=f"Even pace {peso(pace['target_per_day'])}/day")
    fig.update_layout(yaxis_tickprefix="₱", yaxis_tickformat=",.0f", margin={"t": 30, "b": 10})
    st.plotly_chart(fig)

    totals = tag_totals(txs)
    if totals:
        st.subheader("Spend by hashtag")
        st.dataframe(pd.DataFrame([{"Hashtag": f"#{r['tag']}", "Transactions": r["count"], "Total": r["amount"]}
                                   for r in totals]), hide_index=True)
        st.caption("A transaction with several hashtags counts toward each. Income and sweeps are excluded.")


# ----------------------------------------------------------------- savings
def page_savings() -> None:
    st.header("Savings & Digital Bank Routing")
    accounts = service.list_savings_accounts()
    st.metric("Total savings", peso(sum(a["current_balance"] for a in accounts)))
    if accounts:
        proj = portfolio_interest_projection(accounts)
        m1, m2, m3 = st.columns(3)
        m1.metric("Net interest / day", peso(proj["daily_net"]))
        m2.metric("Net interest / month", peso(proj["monthly_net"]))
        m3.metric("Net interest / year", peso(proj["annual_net"]))
        st.caption("Net = after the 20% PH final withholding tax. Estimates use a flat annual rate with no "
                   "compounding, so tiered or promo rates will differ.")
        cols = st.columns(min(len(accounts), 4))
        for i, acct in enumerate(accounts):
            ap = calculate_interest_projection(acct["current_balance"], acct["interest_rate"])
            with cols[i % len(cols)].container(border=True):
                st.markdown(f"**{acct['bank_name']}**")
                st.caption(acct["goal_name"] or "Savings")
                st.subheader(peso(acct["current_balance"]))
                st.caption(f"{acct['interest_rate']:.2f}% p.a.")
                st.caption(f"≈ {peso(ap['daily_net'])}/day · {peso(ap['monthly_net'])}/mo net")
                st.caption(f"≈ {peso(ap['annual_net'])}/yr net ({peso(ap['annual_gross'])} gross)")
        rows = []
        for acct in accounts:
            ap = calculate_interest_projection(acct["current_balance"], acct["interest_rate"])
            rows.append({"Bank": acct["bank_name"], "Goal": acct["goal_name"], "Balance": acct["current_balance"],
                         "Rate %": acct["interest_rate"], "Daily net": round(ap["daily_net"], 2),
                         "Monthly net": round(ap["monthly_net"], 2), "Annual net": round(ap["annual_net"], 2),
                         "Annual gross": round(ap["annual_gross"], 2)})
        st.markdown("**Projected interest by account**")
        st.dataframe(pd.DataFrame(rows), hide_index=True)
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
        for name in ALL_COLLECTIONS:
            docs = snapshot.get(name, [])
            st.subheader(f"{name}  ({len(docs)})")
            st.json(docs, expanded=False)


{"📊 Dashboard": page_dashboard, "🧾 Transactions": page_transactions,
 "🔁 Recurring Bills": page_recurring, "📈 Analytics": page_analytics, "🏦 Savings": page_savings,
 "💳 Debts": page_debts, "🛠️ Developer": page_developer}[page]()
