"""
advanced_features.py
---------------------
Everything added on top of the original DNN demand-prediction system,
without touching the original core (predict.py / feature_engineering.py /
train_model.py keep working exactly as before).

Design principle: the DNN (or its statistical fallback) still does the core
STATISTICAL forecasting from historical borrow patterns — nothing here
retrains or replaces it. These features add a transparent, RULE-BASED
CONTEXT LAYER (syllabus + academic calendar) on top of the DNN's raw
output, plus a set of planning/analytics tools that consume the (adjusted)
predictions. Every adjustment is explainable in plain English — see
explain_demand().

Sections:
    A. Context-adjusted predictions (syllabus + academic calendar)
    B. Explainable AI (perturbation-based feature attribution)
    C. Early Shortage Prediction
    D. Smart Acquisition Planner
    E. Overstock Detection
    F. Budget Optimizer
    G. Student Interest Forecasting
    H. Personalized Reading Path
    I. Anomaly Detection
    J. Department-wise Semester Forecasting
    K. Proactive Alerts (aggregator)
    L. Extra What-If scenarios (new course, enrollment change)
    M. Prediction confidence (for the trend dashboard)
"""

import datetime
import math
import os

import numpy as np
import pandas as pd

import predict as P
import academic_context as AC
from feature_engineering import get_training_table, FEATURE_COLUMNS

TODAY = datetime.date(2026, 8, 15)
LIBRARY_ANNUAL_BUDGET_DEFAULT = 250000  # matches data/generate_data.py; override via API param


# ==========================================================================
# A. CONTEXT-ADJUSTED PREDICTIONS (Syllabus-Aware + Academic Calendar)
# ==========================================================================

_BOOST_MAP_CACHE = {"key": None, "map": None, "status": None}


def _build_boost_map(reference_date=None):
    reference_date = reference_date or TODAY
    cache_key = str(reference_date)
    if _BOOST_MAP_CACHE["key"] == cache_key:
        return _BOOST_MAP_CACHE["map"], _BOOST_MAP_CACHE["status"]

    subs = AC.upcoming_subjects(weeks_ahead=6, reference_date=reference_date)
    status = AC.current_academic_status(reference_date)

    boost_map = {}
    for _, row in subs.iterrows():
        key = (row["department"], row["category"])
        if status["is_exam_now"]:
            boost_map[key] = (1.35, f"'{row['category']}' is an active syllabus subject and semester exams are on now")
        elif status["is_internal_test_now"]:
            boost_map[key] = (1.20, f"'{row['category']}' is an active syllabus subject with an internal test underway")
        else:
            boost_map[key] = (1.10, f"'{row['category']}' is part of the current/upcoming semester syllabus")

    _BOOST_MAP_CACHE.update({"key": cache_key, "map": boost_map, "status": status})
    return boost_map, status


def get_enhanced_predictions(reference_date=None) -> pd.DataFrame:
    """
    The DNN's predict_all_books() output, PLUS a syllabus/calendar-aware
    adjustment layer. Adds:
        boost_multiplier, boost_reason      -> why the number moved
        adjusted_predicted_demand            -> DNN output * boost
        adjusted_demand_score                -> re-scaled 0-100 score
        prediction_confidence, confidence_label  -> how stable recent demand has been
    Original DNN columns (predicted_demand, demand_score) are left untouched.
    """
    base = P.predict_all_books()
    boost_map, status = _build_boost_map(reference_date)

    def lookup(row):
        return boost_map.get((row["department"], row["category"]), (1.0, None))

    boosts = base.apply(lookup, axis=1)
    base = base.copy()
    base["boost_multiplier"] = boosts.apply(lambda t: t[0])
    base["boost_reason"] = boosts.apply(lambda t: t[1])
    base["adjusted_predicted_demand"] = np.round(base["predicted_demand"] * base["boost_multiplier"], 2)
    base["adjusted_demand_score"] = base["adjusted_predicted_demand"].apply(P.PREDICTOR.demand_score)

    conf = _prediction_confidence_all()
    base = base.merge(conf, on="book_id", how="left")

    return base


# ==========================================================================
# M. PREDICTION CONFIDENCE  (for the Demand Trend Dashboard)
# ==========================================================================

def _prediction_confidence_all() -> pd.DataFrame:
    """
    Confidence heuristic: books with a STABLE recent borrowing pattern
    (low week-to-week variance in lag_1..lag_4) get a high-confidence
    forecast; books with erratic recent history get a lower-confidence
    label. This is a standard coefficient-of-variation based proxy —
    simple, explainable, and doesn't require an ensemble of models.
    """
    latest = P.get_latest_feature_rows()
    lags = latest[["lag_1", "lag_2", "lag_3", "lag_4"]]
    mean = lags.mean(axis=1)
    std = lags.std(axis=1).fillna(0)
    cv = std / (mean.replace(0, np.nan))
    cv = cv.fillna(0)
    confidence = (100 / (1 + cv)).clip(0, 100)

    out = pd.DataFrame({
        "book_id": latest["book_id"].values,
        "prediction_confidence": confidence.round(1).values,
    })
    out["confidence_label"] = pd.cut(
        out["prediction_confidence"], bins=[-1, 40, 70, 101],
        labels=["Low", "Medium", "High"]
    ).astype(str)
    return out


# ==========================================================================
# B. EXPLAINABLE AI  (perturbation / occlusion-based feature attribution)
# ==========================================================================

_FEATURE_GROUPS = {
    "Recent borrowing trend": ["lag_1", "lag_2", "lag_3", "lag_4", "rolling_avg_4"],
    "Exam period": ["is_exam_period"],
    "Vacation period": ["is_vacation"],
    "Seasonal calendar position": ["month_sin", "month_cos", "week_sin", "week_cos"],
    "Department-wide pressure": ["dept_week_demand"],
    "Long-run popularity": ["base_popularity"],
}

_FEATURE_MEANS_CACHE = {"means": None}


def _feature_means():
    if _FEATURE_MEANS_CACHE["means"] is None:
        hist = get_training_table()
        _FEATURE_MEANS_CACHE["means"] = hist[FEATURE_COLUMNS].mean()
    return _FEATURE_MEANS_CACHE["means"]


def explain_demand(book_id: int) -> dict:
    """
    Answers: WHY is this book's demand predicted as high/low?
    Technique: occlusion-based sensitivity analysis — for each factor
    group, neutralise it (reset to the historical average) and see how
    much the prediction changes. The bigger the drop, the bigger that
    factor's contribution to the current prediction. This works whether
    the underlying predictor is the trained DNN or the statistical
    fallback (both are exposed through the same predict_batch() call),
    so explanations remain available in every run mode.
    """
    latest = P.get_latest_feature_rows()
    row = latest[latest["book_id"] == book_id]
    if row.empty:
        return None
    row = row.copy()

    baseline_pred = float(P.PREDICTOR.predict_batch(row)[0])
    means = _feature_means()

    contributions = []
    for group_name, cols in _FEATURE_GROUPS.items():
        modified = row.copy()
        for c in cols:
            if c in modified.columns:
                modified[c] = means[c]
        modified_pred = float(P.PREDICTOR.predict_batch(modified)[0])
        delta = baseline_pred - modified_pred
        contributions.append({"factor": group_name, "impact": round(delta, 3)})

    contributions.sort(key=lambda x: abs(x["impact"]), reverse=True)
    total_abs = sum(abs(c["impact"]) for c in contributions) or 1.0
    for c in contributions:
        c["contribution_pct"] = round(100 * c["impact"] / total_abs, 1)
        c["direction"] = "increases" if c["impact"] > 0 else ("decreases" if c["impact"] < 0 else "no effect")

    # context layer (syllabus / calendar) explanation, on top of the DNN explanation
    enhanced = get_enhanced_predictions()
    erow = enhanced[enhanced["book_id"] == book_id]
    context_reason = None
    boost_multiplier = 1.0
    if not erow.empty:
        context_reason = erow.iloc[0]["boost_reason"]
        boost_multiplier = float(erow.iloc[0]["boost_multiplier"])

    status = AC.current_academic_status()
    plain_summary = []
    top_positive = [c for c in contributions if c["impact"] > 0][:2]
    if top_positive:
        plain_summary.append(
            "Driven mainly by: " + ", ".join(f"{c['factor'].lower()}" for c in top_positive)
        )
    if context_reason:
        plain_summary.append(f"Context boost applied: {context_reason} (x{boost_multiplier})")
    if status["is_exam_now"]:
        plain_summary.append("Semester exams are currently underway university-wide.")
    elif status["is_internal_test_now"]:
        plain_summary.append("An internal test is currently underway university-wide.")

    return {
        "book_id": book_id,
        "baseline_predicted_demand": round(baseline_pred, 2),
        "context_adjusted_demand": round(baseline_pred * boost_multiplier, 2),
        "factor_breakdown": contributions,
        "context_reason": context_reason,
        "plain_summary": " | ".join(plain_summary) if plain_summary else "Driven by steady historical demand.",
    }


# ==========================================================================
# C. EARLY SHORTAGE PREDICTION
# ==========================================================================

def early_shortage_prediction(weeks_ahead: int = 3) -> pd.DataFrame:
    """
    Projects forward: at the current predicted weekly borrow rate, how
    many weeks until available_copies hits zero? Flags books that will
    run out WITHIN `weeks_ahead` weeks — i.e. before they'd normally
    trigger a "low availability" alert (which only fires once stock is
    already critically low). This is a forward-looking risk radar.
    """
    df = get_enhanced_predictions()
    df = df[df["available_copies"] > 0].copy()  # already-zero books are a current alert, not a "prediction"

    weekly_rate = df["adjusted_predicted_demand"].clip(lower=0.05)
    df["weeks_to_shortage"] = np.round(df["available_copies"] / weekly_rate, 1)

    at_risk = df[df["weeks_to_shortage"] <= weeks_ahead].copy()
    at_risk["risk_level"] = np.select(
        [at_risk["weeks_to_shortage"] <= 1, at_risk["weeks_to_shortage"] <= 2],
        ["Critical", "High"],
        default="Moderate",
    )
    cols = ["book_id", "title", "category", "department", "available_copies",
            "adjusted_predicted_demand", "weeks_to_shortage", "risk_level"]
    return at_risk.sort_values("weeks_to_shortage")[cols].reset_index(drop=True)


# ==========================================================================
# D. SMART ACQUISITION PLANNER
# ==========================================================================

def smart_acquisition_planner(horizon_weeks: int = 4) -> pd.DataFrame:
    """
    Recommends EXACTLY how many extra copies of which books to buy over
    the given horizon (default: next month), using the syllabus/calendar
    -adjusted demand (not just raw history), and prices it out using each
    book's unit_cost.

    The "worth buying" cutoff is the 65th percentile of the CURRENT
    adjusted-demand-score distribution (rather than a fixed number) so the
    planner adapts automatically whether the whole catalog is in a quiet
    off-peak week or a high-pressure exam week.
    """
    df = get_enhanced_predictions()
    books = P.get_books_df()[["book_id", "unit_cost", "difficulty_level"]]
    df = df.merge(books, on="book_id", how="left")

    projected_need = np.ceil(df["adjusted_predicted_demand"] * horizon_weeks * 0.55)
    df["recommended_extra_copies"] = np.clip(projected_need - df["total_copies"], 0, None).astype(int)

    score_cutoff = df["adjusted_demand_score"].quantile(0.65)
    df.loc[df["adjusted_demand_score"] < score_cutoff, "recommended_extra_copies"] = 0
    df["recommended_extra_copies"] = df["recommended_extra_copies"].clip(upper=8)
    # ensure genuinely high-demand-but-thin-stock books always get at least 1 copy recommended
    thin_stock = (df["adjusted_demand_score"] >= df["adjusted_demand_score"].quantile(0.85)) & (df["total_copies"] <= 3)
    df.loc[thin_stock & (df["recommended_extra_copies"] == 0), "recommended_extra_copies"] = 2

    df["estimated_cost"] = df["recommended_extra_copies"] * df["unit_cost"]
    df["priority_score"] = round(df["adjusted_demand_score"] * (1 + df["recommended_extra_copies"] / 5), 1)

    result = df[df["recommended_extra_copies"] > 0].copy()
    cols = ["book_id", "title", "category", "department", "adjusted_predicted_demand",
            "adjusted_demand_score", "recommended_extra_copies", "unit_cost",
            "estimated_cost", "priority_score"]
    return result.sort_values("priority_score", ascending=False)[cols].reset_index(drop=True)


# ==========================================================================
# E. OVERSTOCK DETECTION
# ==========================================================================

def overstock_detection(min_copies: int = 3, demand_score_percentile: float = 0.30) -> pd.DataFrame:
    """
    Mirror image of the acquisition planner: books that are barely
    borrowed but occupy a lot of shelf copies. Flags them for weeding /
    reduced reordering, and estimates the acquisition budget that could
    be freed up (feeds into the Budget Optimizer).

    Uses the bottom `demand_score_percentile` of the CURRENT adjusted
    demand-score distribution (adaptive, like the acquisition planner)
    rather than a fixed score cutoff.
    """
    df = get_enhanced_predictions()
    books = P.get_books_df()[["book_id", "unit_cost"]]
    df = df.merge(books, on="book_id", how="left")

    score_cutoff = df["adjusted_demand_score"].quantile(demand_score_percentile)
    df = df[(df["total_copies"] >= min_copies) & (df["adjusted_demand_score"] <= score_cutoff)].copy()
    ideal_copies = np.ceil(df["adjusted_predicted_demand"] * 3).clip(lower=1)
    df["excess_copies"] = np.clip(df["total_copies"] - ideal_copies, 0, None).astype(int)
    df = df[df["excess_copies"] >= 2]  # only meaningfully overstocked books
    df["potential_savings"] = df["excess_copies"] * df["unit_cost"]

    cols = ["book_id", "title", "category", "department", "total_copies",
            "adjusted_demand_score", "excess_copies", "potential_savings"]
    return df.sort_values("potential_savings", ascending=False)[cols].reset_index(drop=True)


# ==========================================================================
# F. BUDGET OPTIMIZER
# ==========================================================================

def budget_optimizer(total_budget: float = None) -> dict:
    """
    Greedy knapsack-style allocation: fund the highest priority_score
    acquisitions first until the budget runs out. Also shows how much of
    the budget could effectively be extended by reinvesting a share of
    the savings identified by Overstock Detection (transparent, simple
    heuristic — not a full financial optimizer, but a practical planning
    aid a librarian could actually use).
    """
    total_budget = total_budget if total_budget is not None else LIBRARY_ANNUAL_BUDGET_DEFAULT

    plan = smart_acquisition_planner()
    overstock = overstock_detection()
    raw_savings = float(overstock["potential_savings"].sum()) if len(overstock) else 0.0
    # reinvest at most 40% of the base budget's worth of savings — keeps the
    # optimizer a realistic planning aid rather than letting an unusually
    # large overstock estimate dwarf the real budget
    reinvested_savings = min(raw_savings * 0.5, total_budget * 0.4)
    effective_budget = total_budget + reinvested_savings

    plan = plan.sort_values("priority_score", ascending=False).copy()
    allocations = []
    unfunded = []
    remaining = effective_budget

    for _, row in plan.iterrows():
        cost = float(row["estimated_cost"])
        if cost <= 0:
            continue
        if cost <= remaining:
            allocations.append({
                "book_id": int(row["book_id"]), "title": row["title"], "category": row["category"],
                "copies_funded": int(row["recommended_extra_copies"]),
                "amount_spent": cost, "priority_score": row["priority_score"],
            })
            remaining -= cost
        else:
            # try to partially fund (as many whole copies as the remaining budget allows)
            unit_cost = cost / max(row["recommended_extra_copies"], 1)
            affordable_copies = int(remaining // unit_cost) if unit_cost > 0 else 0
            if affordable_copies > 0:
                spent = affordable_copies * unit_cost
                allocations.append({
                    "book_id": int(row["book_id"]), "title": row["title"], "category": row["category"],
                    "copies_funded": affordable_copies,
                    "amount_spent": round(spent, 2), "priority_score": row["priority_score"],
                })
                remaining -= spent
            unfunded.append({
                "book_id": int(row["book_id"]), "title": row["title"], "category": row["category"],
                "copies_needed": int(row["recommended_extra_copies"]) - affordable_copies,
                "priority_score": row["priority_score"],
            })

    return {
        "total_budget": total_budget,
        "reinvested_overstock_savings": round(reinvested_savings, 2),
        "effective_budget": round(effective_budget, 2),
        "total_spent": round(effective_budget - remaining, 2),
        "remaining_budget": round(remaining, 2),
        "allocations": allocations,
        "unfunded": unfunded,
    }


# ==========================================================================
# G. STUDENT INTEREST FORECASTING
# ==========================================================================

def student_interest_forecast(student_id: int, recent_days: int = 90) -> list:
    """
    Splits a student's borrowing history into a RECENT window vs the
    window before it, and compares category frequency to detect emerging
    (rising) interests vs fading ones — a lightweight trend-detection
    technique (no separate model needed, just windowed counting).
    """
    tx = P.get_transactions_df()
    books = P.get_books_df()
    student_tx = tx[tx["student_id"] == student_id].merge(books, on="book_id", how="left")
    if student_tx.empty:
        return []

    cutoff_recent = pd.Timestamp(TODAY) - pd.Timedelta(days=recent_days)
    cutoff_earlier = cutoff_recent - pd.Timedelta(days=recent_days)

    recent = student_tx[student_tx["borrow_date"] >= cutoff_recent]
    earlier = student_tx[(student_tx["borrow_date"] >= cutoff_earlier) & (student_tx["borrow_date"] < cutoff_recent)]

    recent_counts = recent["category"].value_counts()
    earlier_counts = earlier["category"].value_counts()

    all_cats = set(recent_counts.index) | set(earlier_counts.index)
    results = []
    for cat in all_cats:
        r = int(recent_counts.get(cat, 0))
        e = int(earlier_counts.get(cat, 0))
        if r == 0 and e == 0:
            continue
        is_new = e == 0 and r > 0
        if is_new:
            trend = "New interest"
        elif r > e:
            trend = "Rising"
        elif r < e:
            trend = "Fading"
        else:
            trend = "Stable"
        results.append({
            "category": cat, "recent_borrows": r, "earlier_borrows": e,
            "trend": trend, "is_new_interest": is_new,
        })

    results.sort(key=lambda x: (x["trend"] == "Rising" or x["is_new_interest"], x["recent_borrows"]), reverse=True)
    return results


# ==========================================================================
# H. PERSONALIZED READING PATH
# ==========================================================================

_DIFFICULTY_ORDER = {"Beginner": 0, "Intermediate": 1, "Advanced": 2}


def personalized_reading_path(student_id: int, category: str = None, path_length: int = 5) -> dict:
    """
    Builds an ORDERED sequence of books (Beginner -> Intermediate ->
    Advanced) in the student's top-interest category (or a chosen one),
    skipping books already borrowed, and using demand/availability as a
    tiebreaker within each difficulty tier. This is a genuine "path"
    (progression), not just a ranked list.
    """
    pattern = P.student_reading_pattern(student_id)
    if category is None:
        if pattern["top_categories"]:
            category = pattern["top_categories"][0]["category"]
        else:
            return {"student_id": student_id, "category": None, "path": []}

    tx = P.get_transactions_df()
    already_borrowed = set(tx[tx["student_id"] == student_id]["book_id"].tolist())

    enhanced = get_enhanced_predictions()
    books = P.get_books_df()[["book_id", "difficulty_level"]]
    pool = enhanced.merge(books, on="book_id", how="left")
    pool = pool[(pool["category"] == category) & (~pool["book_id"].isin(already_borrowed))].copy()

    pool["difficulty_rank"] = pool["difficulty_level"].map(_DIFFICULTY_ORDER).fillna(1)
    pool = pool.sort_values(["difficulty_rank", "adjusted_demand_score"], ascending=[True, False])

    path = pool.head(path_length).copy()
    path["step"] = range(1, len(path) + 1)
    cols = ["step", "book_id", "title", "difficulty_level", "available_copies", "adjusted_demand_score"]
    return {
        "student_id": student_id,
        "category": category,
        "path": path[cols].to_dict("records"),
    }


# ==========================================================================
# I. ANOMALY DETECTION
# ==========================================================================

def anomaly_detection(z_threshold: float = 2.5, min_history_weeks: int = 8) -> pd.DataFrame:
    """
    Statistical (z-score / control-chart) anomaly detection: for each book,
    compares its most recent completed week's actual borrow count to the
    mean/std of its own prior weekly history. Flags books whose latest
    week is an unusual spike (possible syllabus change, hoarding, exam
    leak, or a data-entry issue) or an unusual drop (book missing/damaged,
    withdrawn from syllabus, etc).
    """
    hist = get_training_table()
    books = P.get_books_df()[["book_id", "title", "category", "department"]]

    results = []
    for book_id, g in hist.groupby("book_id"):
        g = g.sort_values("week")
        if len(g) < min_history_weeks + 1:
            continue
        prior = g.iloc[:-1]["demand"]
        latest_week_demand = g.iloc[-1]["demand"]
        mean, std = prior.mean(), prior.std()
        if std == 0 or np.isnan(std):
            continue
        z = (latest_week_demand - mean) / std
        if abs(z) >= z_threshold:
            results.append({
                "book_id": book_id, "latest_week_demand": int(latest_week_demand),
                "historical_avg": round(float(mean), 2), "z_score": round(float(z), 2),
                "anomaly_type": "Unusual Spike" if z > 0 else "Unusual Drop",
            })

    if not results:
        return pd.DataFrame(columns=["book_id", "title", "category", "department",
                                      "latest_week_demand", "historical_avg", "z_score", "anomaly_type"])

    out = pd.DataFrame(results).merge(books, on="book_id", how="left")
    cols = ["book_id", "title", "category", "department", "latest_week_demand",
            "historical_avg", "z_score", "anomaly_type"]
    return out.sort_values("z_score", key=abs, ascending=False)[cols].reset_index(drop=True)


# ==========================================================================
# J. DEPARTMENT-WISE SEMESTER FORECASTING
# ==========================================================================

def department_semester_forecast() -> list:
    df = get_enhanced_predictions()
    subs = AC.load_subjects()
    status = AC.current_academic_status()

    by_dept = df.groupby("department").agg(
        total_adjusted_demand=("adjusted_predicted_demand", "sum"),
        avg_adjusted_score=("adjusted_demand_score", "mean"),
        n_books=("book_id", "count"),
    ).reset_index()

    dept_semester = subs.groupby("department")["semester_type"].agg(lambda s: s.mode().iloc[0] if not s.mode().empty else None)

    next_event = status["upcoming_events"][0] if status["upcoming_events"] else None

    records = []
    for _, row in by_dept.iterrows():
        records.append({
            "department": row["department"],
            "total_adjusted_demand": round(float(row["total_adjusted_demand"]), 1),
            "avg_adjusted_score": round(float(row["avg_adjusted_score"]), 1),
            "n_books": int(row["n_books"]),
            "primary_semester_type": dept_semester.get(row["department"]),
            "next_calendar_event": next_event["event_type"] if next_event else None,
            "next_event_days_away": next_event["days_away"] if next_event else None,
        })
    records.sort(key=lambda r: r["total_adjusted_demand"], reverse=True)
    return records


# ==========================================================================
# K. PROACTIVE ALERTS  (aggregator across everything above)
# ==========================================================================

def proactive_alerts() -> list:
    alerts = []

    core = P.get_alerts()
    for _, r in core.iterrows():
        alerts.append({
            "book_id": int(r["book_id"]), "title": r["title"], "category": r["category"],
            "alert_type": "High Demand + Low Stock", "severity": r["severity"],
            "message": f"Demand score {r['demand_score']} but only {r['available_copies']}/{r['total_copies']} copies available.",
        })

    shortage = early_shortage_prediction(weeks_ahead=2)
    for _, r in shortage.iterrows():
        alerts.append({
            "book_id": int(r["book_id"]), "title": r["title"], "category": r["category"],
            "alert_type": "Upcoming Shortage", "severity": r["risk_level"],
            "message": f"Projected to run out in ~{r['weeks_to_shortage']} weeks at current demand.",
        })

    anomalies = anomaly_detection()
    for _, r in anomalies.iterrows():
        sev = "Critical" if abs(r["z_score"]) >= 3.5 else "High"
        alerts.append({
            "book_id": int(r["book_id"]), "title": r["title"], "category": r["category"],
            "alert_type": r["anomaly_type"], "severity": sev,
            "message": f"Last week's demand ({r['latest_week_demand']}) is unusual vs its own history (avg {r['historical_avg']}, z={r['z_score']}).",
        })

    overstock = overstock_detection().head(5)
    for _, r in overstock.iterrows():
        alerts.append({
            "book_id": int(r["book_id"]), "title": r["title"], "category": r["category"],
            "alert_type": "Overstock / Low Usage", "severity": "Info",
            "message": f"{r['excess_copies']} likely-unneeded copies (potential savings: {r['potential_savings']}).",
        })

    severity_rank = {"Critical": 0, "High": 1, "Moderate": 2, "Info": 3}
    alerts.sort(key=lambda a: severity_rank.get(a["severity"], 4))
    return alerts


# ==========================================================================
# L. EXTRA WHAT-IF SCENARIOS
# ==========================================================================

def what_if_new_course(department: str, category: str, intensity: str = "internal_test") -> pd.DataFrame:
    """
    Simulates the effect of introducing/activating a new course tied to
    `category` in `department` (e.g. "what if we launch an elective in
    Deep Learning this semester?") by applying the same syllabus-boost
    logic used for real syllabus subjects.
    """
    boost = {"normal": 1.10, "internal_test": 1.20, "exam": 1.35}.get(intensity, 1.10)
    df = P.predict_all_books()
    df = df[(df["department"] == department) & (df["category"] == category)].copy()
    df["simulated_demand"] = np.round(df["predicted_demand"] * boost, 2)
    df["simulated_demand_score"] = df["simulated_demand"].apply(P.PREDICTOR.demand_score)
    df["scenario"] = f"New/expanded course in '{category}' ({department})"
    cols = ["book_id", "title", "category", "available_copies", "predicted_demand",
            "simulated_demand", "simulated_demand_score", "scenario"]
    return df.sort_values("simulated_demand", ascending=False)[cols].reset_index(drop=True)


def what_if_enrollment_change(department: str, pct_change: float) -> pd.DataFrame:
    """
    Simulates a department-wide enrollment increase/decrease (e.g. "what
    if CS intake grows 20% next year?") by scaling the department-week
    demand pressure feature and re-running the SAME trained predictor.
    """
    latest = P.get_latest_feature_rows().copy()
    latest = latest[latest["department"] == department]
    if latest.empty:
        return pd.DataFrame()

    scale = 1 + (pct_change / 100.0)
    modified = latest.copy()
    modified["dept_week_demand"] = modified["dept_week_demand"] * scale
    # recent trend also nudged proportionally - more students -> more borrows across the board
    for c in ["lag_1", "lag_2", "lag_3", "lag_4", "rolling_avg_4"]:
        modified[c] = modified[c] * scale

    baseline_preds = P.PREDICTOR.predict_batch(latest)
    scenario_preds = P.PREDICTOR.predict_batch(modified)

    books = P.get_books_df()[["book_id", "title"]]
    out = latest[["book_id", "category"]].merge(books, on="book_id", how="left")
    out["baseline_demand"] = np.round(baseline_preds, 2)
    out["simulated_demand"] = np.round(scenario_preds, 2)
    out["simulated_demand_score"] = [P.PREDICTOR.demand_score(p) for p in scenario_preds]
    out["scenario"] = f"{department} enrollment {'+' if pct_change >= 0 else ''}{pct_change}%"
    return out.sort_values("simulated_demand", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    print("=== Enhanced predictions (head) ===")
    print(get_enhanced_predictions().head(5)[["title", "predicted_demand", "adjusted_predicted_demand", "boost_reason"]])

    print("\n=== Explain demand (book_id=1) ===")
    print(explain_demand(1))

    print("\n=== Early shortage prediction ===")
    print(early_shortage_prediction().head(5))

    print("\n=== Smart acquisition planner ===")
    print(smart_acquisition_planner().head(5))

    print("\n=== Overstock detection ===")
    print(overstock_detection().head(5))

    print("\n=== Budget optimizer ===")
    res = budget_optimizer()
    print({k: v for k, v in res.items() if k not in ("allocations", "unfunded")})
    print("allocations:", len(res["allocations"]), "unfunded:", len(res["unfunded"]))

    print("\n=== Student interest forecast (student 50) ===")
    print(student_interest_forecast(50))

    print("\n=== Personalized reading path (student 50) ===")
    print(personalized_reading_path(50))

    print("\n=== Anomaly detection ===")
    print(anomaly_detection().head(5))

    print("\n=== Department semester forecast ===")
    print(department_semester_forecast())

    print("\n=== Proactive alerts ===")
    print(proactive_alerts()[:5])
