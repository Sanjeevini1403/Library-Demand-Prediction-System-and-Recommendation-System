"""
academic_context.py
--------------------
Reads the `subjects` (syllabus map) and `academic_calendar` tables and
exposes helpers that answer the questions:

    - "What semester/exam window are we in right now?"
    - "What's coming up in the next N weeks (internal test, exam, holiday)?"
    - "Which subjects/categories will be actively taught soon, for a given
       department?" (Syllabus-Aware Prediction)

These are used as an EXPLAINABLE, RULE-BASED ADJUSTMENT LAYER on top of the
DNN's raw output — the DNN still does the core statistical forecasting
(from historical borrow patterns), and this layer adds domain knowledge
(syllabus + calendar) the DNN doesn't directly see, in a fully transparent
way (every boost/adjustment can be printed as a human-readable reason —
see model/advanced_features.py: explain_demand()).
"""

import os
import sqlite3
import datetime

import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "..", "data", "library.db")

TODAY = datetime.date(2026, 8, 15)  # kept in sync with data/generate_data.py


def get_conn():
    return sqlite3.connect(DB_PATH)


def load_subjects() -> pd.DataFrame:
    with get_conn() as conn:
        return pd.read_sql("SELECT * FROM subjects", conn)


def load_calendar() -> pd.DataFrame:
    with get_conn() as conn:
        df = pd.read_sql("SELECT * FROM academic_calendar", conn)
    df["start_date"] = pd.to_datetime(df["start_date"]).dt.date
    df["end_date"] = pd.to_datetime(df["end_date"]).dt.date
    return df


# --------------------------------------------------------------------------
# Academic Calendar Prediction
# --------------------------------------------------------------------------

def current_academic_status(reference_date: datetime.date = None) -> dict:
    """What's happening RIGHT NOW academically, and what's coming up next."""
    reference_date = reference_date or TODAY
    cal = load_calendar()

    current = cal[(cal["start_date"] <= reference_date) & (cal["end_date"] >= reference_date)]
    upcoming = cal[cal["start_date"] > reference_date].sort_values("start_date")

    current_events = current[["event_type", "start_date", "end_date", "semester_type"]].copy()
    current_events["start_date"] = current_events["start_date"].astype(str)
    current_events["end_date"] = current_events["end_date"].astype(str)

    next_events = upcoming.head(4)[["event_type", "start_date", "end_date", "semester_type"]].copy()
    next_events["days_away"] = upcoming.head(4)["start_date"].apply(lambda d: (d - reference_date).days)
    next_events["start_date"] = next_events["start_date"].astype(str)
    next_events["end_date"] = next_events["end_date"].astype(str)

    is_exam_now = "Semester Exam" in current_events["event_type"].values
    is_internal_test_now = current_events["event_type"].str.startswith("Internal Test").any()
    is_vacation_now = current_events["semester_type"].eq("Vacation").any()

    return {
        "reference_date": str(reference_date),
        "current_events": current_events.to_dict("records"),
        "upcoming_events": next_events.to_dict("records"),
        "is_exam_now": bool(is_exam_now),
        "is_internal_test_now": bool(is_internal_test_now),
        "is_vacation_now": bool(is_vacation_now),
    }


def days_to_next_event(event_type_prefix: str, reference_date: datetime.date = None) -> int:
    reference_date = reference_date or TODAY
    cal = load_calendar()
    future = cal[(cal["start_date"] > reference_date) &
                 (cal["event_type"].str.startswith(event_type_prefix))]
    if future.empty:
        return None
    nxt = future.sort_values("start_date").iloc[0]
    return (nxt["start_date"] - reference_date).days


# --------------------------------------------------------------------------
# Syllabus-Aware Prediction
# --------------------------------------------------------------------------

def upcoming_subjects(weeks_ahead: int = 6, department: str = None,
                       reference_date: datetime.date = None) -> pd.DataFrame:
    """
    Subjects that are either being taught RIGHT NOW or will start within
    `weeks_ahead` weeks (i.e. their semester_type's exam or start window
    is approaching) — these are the categories a syllabus-aware system
    should start boosting demand for.
    """
    reference_date = reference_date or TODAY
    subjects = load_subjects()
    if department:
        subjects = subjects[subjects["department"] == department]

    cal = load_calendar()
    horizon = reference_date + datetime.timedelta(weeks=weeks_ahead)

    # semester types that are "active or about to start" within the horizon
    relevant = cal[(cal["event_type"] == "Semester Start") &
                    (cal["start_date"] >= reference_date - datetime.timedelta(days=150)) &
                    (cal["start_date"] <= horizon)]
    # also include the CURRENT semester (already running)
    current_sem = cal[(cal["start_date"] <= reference_date) & (cal["end_date"] >= reference_date)]

    relevant_types = set(relevant["semester_type"]).union(set(current_sem["semester_type"]))
    relevant_types.discard("Vacation")

    result = subjects[subjects["semester_type"].isin(relevant_types)].copy()
    return result.reset_index(drop=True)


def syllabus_boost_for_category(category: str, department: str = None,
                                 reference_date: datetime.date = None) -> tuple:
    """
    Returns (boost_multiplier, reason_string) for a given book category.
    boost_multiplier of 1.0 = no adjustment.
    """
    subs = upcoming_subjects(weeks_ahead=6, department=department, reference_date=reference_date)
    match = subs[subs["category"] == category]
    if match.empty:
        return 1.0, None

    status = current_academic_status(reference_date)
    if status["is_exam_now"]:
        return 1.35, f"'{category}' is an active syllabus subject and semester exams are on now"
    if status["is_internal_test_now"]:
        return 1.20, f"'{category}' is an active syllabus subject with an internal test underway"
    return 1.10, f"'{category}' is part of the current/upcoming semester syllabus"


if __name__ == "__main__":
    print(current_academic_status())
    print(upcoming_subjects().head(10))
