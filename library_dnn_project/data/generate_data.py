"""
generate_data.py
-----------------
Creates a REALISTIC SYNTHETIC library dataset (no real dataset needed).

Generates:
    - books table        (catalog with category/department, copies, popularity)
    - students table      (department, year of study)
    - borrow_transactions  (2 academic years of borrowing history with
                             semester / exam-season / holiday seasonality)

Everything is written to an SQLite database (library.db) AND exported as
CSV files (for quick inspection / use in notebooks).

Run:
    python data/generate_data.py
"""

import sqlite3
import random
import os
from datetime import date, timedelta

import numpy as np
import pandas as pd

random.seed(42)
np.random.seed(42)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "library.db")

# --------------------------------------------------------------------------
# 1. CONFIG
# --------------------------------------------------------------------------

DEPARTMENTS = [
    "Computer Science", "Electronics & Comm.", "Mechanical",
    "Civil", "Electrical", "AI & Data Science", "Biotechnology", "MBA"
]

CATEGORIES = {
    "Computer Science": ["Programming", "Data Structures", "AI/ML", "Databases", "Networks"],
    "Electronics & Comm.": ["Circuits", "Signal Processing", "Communication Systems", "VLSI"],
    "Mechanical": ["Thermodynamics", "Design", "Manufacturing", "Fluid Mechanics"],
    "Civil": ["Structural Engg", "Surveying", "Construction", "Geotechnical"],
    "Electrical": ["Power Systems", "Control Systems", "Machines"],
    "AI & Data Science": ["AI/ML", "Data Structures", "Statistics", "Deep Learning"],
    "Biotechnology": ["Genetics", "Microbiology", "Bioprocess"],
    "MBA": ["Finance", "Marketing", "HR", "Operations"],
    "General": ["Fiction", "Competitive Exams", "Soft Skills", "General Science"]
}

ALL_CATEGORIES = sorted({c for cats in CATEGORIES.values() for c in cats})

TITLES_POOL = [
    "Introduction to {c}", "Fundamentals of {c}", "Advanced {c}", "{c} Made Easy",
    "A Practical Guide to {c}", "{c}: Theory and Practice", "Concepts of {c}",
    "Mastering {c}", "{c} for Engineers", "Applied {c}"
]

# difficulty progression - used by the Personalized Reading Path feature
DIFFICULTY_BY_TEMPLATE = {
    "Introduction to {c}": "Beginner",
    "Fundamentals of {c}": "Beginner",
    "{c} Made Easy": "Beginner",
    "A Practical Guide to {c}": "Intermediate",
    "Concepts of {c}": "Intermediate",
    "{c}: Theory and Practice": "Intermediate",
    "{c} for Engineers": "Intermediate",
    "Applied {c}": "Advanced",
    "Advanced {c}": "Advanced",
    "Mastering {c}": "Advanced",
}
DIFFICULTY_ORDER = {"Beginner": 0, "Intermediate": 1, "Advanced": 2}

NUM_BOOKS = 220
NUM_STUDENTS = 600
START_DATE = date(2024, 6, 1)      # academic year start
END_DATE = date(2026, 5, 31)       # ~2 academic years of history
TODAY = date(2026, 8, 15)          # "current date" the system predicts from
CALENDAR_END_DATE = date(2027, 5, 31)  # academic calendar is generated further into
                                        # the FUTURE than the transaction history, so
                                        # calendar-aware / what-if features have real
                                        # upcoming events to reason about

LIBRARY_ANNUAL_BUDGET = 250000  # (currency-agnostic units) used by the Budget Optimizer


# --------------------------------------------------------------------------
# 2. SEMESTER / EXAM CALENDAR
#    (odd semester: Jul-Nov, even semester: Dec-Apr, May-Jun = vacation)
# --------------------------------------------------------------------------

def semester_info(d: date):
    """Return (semester_label, is_exam_period, is_vacation) for a date."""
    m = d.month
    if m in (6,):
        return "Vacation", False, True
    if m in (7, 8, 9, 10):
        return "Odd-Semester", False, False
    if m == 11:
        return "Odd-Semester", True, False           # odd sem exams
    if m in (12, 1, 2, 3):
        return "Even-Semester", False, False
    if m == 4:
        return "Even-Semester", True, False           # even sem exams
    return "Vacation", False, True                     # May


# --------------------------------------------------------------------------
# 3. BOOKS
# --------------------------------------------------------------------------

def make_books():
    rows = []
    book_id = 1
    for cat in ALL_CATEGORIES:
        n = random.randint(20, 30)
        for _ in range(n):
            title_template = random.choice(TITLES_POOL)
            title = title_template.format(c=cat) + f" Vol.{random.randint(1,3)}"
            total_copies = random.choice([2, 3, 4, 5, 6, 8, 10])
            # base_popularity drives how "in demand" a book naturally is (0-1)
            base_popularity = np.clip(np.random.beta(2, 5), 0.02, 0.98)
            publish_year = random.randint(2008, 2025)
            difficulty = DIFFICULTY_BY_TEMPLATE[title_template]
            # advanced/specialist books cost more to acquire
            base_cost = {"Beginner": 350, "Intermediate": 550, "Advanced": 850}[difficulty]
            unit_cost = int(base_cost + random.randint(-80, 250))
            rows.append({
                "book_id": book_id,
                "title": title,
                "author": f"Author {random.randint(1, 150)}",
                "category": cat,
                "department": next(dep for dep, cats in CATEGORIES.items() if cat in cats),
                "total_copies": total_copies,
                "available_copies": total_copies,  # updated after transactions
                "publish_year": publish_year,
                "base_popularity": round(float(base_popularity), 3),
                "difficulty_level": difficulty,       # Beginner/Intermediate/Advanced -> reading path ordering
                "unit_cost": unit_cost,                # -> acquisition planner / budget optimizer
            })
            book_id += 1
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 4. STUDENTS
# --------------------------------------------------------------------------

def make_students():
    rows = []
    for sid in range(1, NUM_STUDENTS + 1):
        dept = random.choice(DEPARTMENTS)
        rows.append({
            "student_id": sid,
            "department": dept,
            "year_of_study": random.choice([1, 2, 3, 4]),
            # each student has a personal "reading affinity" across a few categories
            "preferred_categories": ",".join(
                random.sample(CATEGORIES.get(dept, ALL_CATEGORIES),
                               k=min(2, len(CATEGORIES.get(dept, ALL_CATEGORIES))))
                + random.sample(ALL_CATEGORIES, k=1)
            ),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 4B. SUBJECTS / SYLLABUS  (drives Syllabus-Aware Prediction)
#     Each department teaches a rotating set of subjects every Odd/Even
#     semester. Each subject maps to a book category, so we know exactly
#     which categories become "syllabus-relevant" as a semester approaches.
# --------------------------------------------------------------------------

def make_subjects():
    rows = []
    subject_id = 1
    for dep, cats in CATEGORIES.items():
        for i, cat in enumerate(cats):
            # alternate subjects between Odd/Even semester so every department
            # has a realistic mix of "this semester" and "next semester" courses
            semester_type = "Odd-Semester" if i % 2 == 0 else "Even-Semester"
            rows.append({
                "subject_id": subject_id,
                "department": dep,
                "semester_type": semester_type,
                "subject_name": f"{cat} (Core Course)",
                "category": cat,
                # week (0-indexed, within the ~5 month semester) the topic
                # becomes exam-relevant / syllabus-active
                "syllabus_active_week": random.choice([1, 2, 4, 6, 8]),
            })
            subject_id += 1
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 4C. ACADEMIC CALENDAR  (drives Academic Calendar Prediction / What-If)
#     Explicit, named events (Internal Tests, Semester Exams, Holidays)
#     generated across past + current + FUTURE academic years, consistent
#     with semester_info() above, but with real event names/date ranges
#     the rest of the system can reason about ("2 weeks to Internal Test 2").
# --------------------------------------------------------------------------

def make_academic_calendar():
    rows = []
    event_id = 1
    year = START_DATE.year
    while date(year, 6, 1) <= CALENDAR_END_DATE:
        # ---- Odd semester: Jul - Nov ----
        rows.append(("Semester Start", date(year, 7, 1), date(year, 7, 7), "Odd-Semester"))
        rows.append(("Internal Test 1", date(year, 8, 18), date(year, 8, 23), "Odd-Semester"))
        rows.append(("Internal Test 2", date(year, 10, 6), date(year, 10, 11), "Odd-Semester"))
        rows.append(("Semester Exam", date(year, 11, 1), date(year, 11, 25), "Odd-Semester"))

        # ---- Even semester: Dec - Apr (spills into next calendar year) ----
        rows.append(("Semester Start", date(year, 12, 1), date(year, 12, 7), "Even-Semester"))
        rows.append(("Internal Test 1", date(year + 1, 1, 12), date(year + 1, 1, 17), "Even-Semester"))
        rows.append(("Internal Test 2", date(year + 1, 3, 2), date(year + 1, 3, 7), "Even-Semester"))
        rows.append(("Semester Exam", date(year + 1, 4, 1), date(year + 1, 4, 25), "Even-Semester"))

        # ---- Vacation / Holidays ----
        rows.append(("Holiday - Summer Vacation", date(year + 1, 5, 1), date(year + 1, 5, 31), "Vacation"))
        rows.append(("Holiday - Term Break", date(year, 11, 26), date(year, 11, 30), "Odd-Semester"))

        year += 1

    df = pd.DataFrame(rows, columns=["event_type", "start_date", "end_date", "semester_type"])
    df = df[(df["start_date"] <= CALENDAR_END_DATE)].copy()
    df.insert(0, "event_id", range(1, len(df) + 1))
    df["department"] = "All"  # all events apply university-wide in this model
    df["start_date"] = df["start_date"].apply(lambda d: d.isoformat())
    df["end_date"] = df["end_date"].apply(lambda d: d.isoformat())
    return df.sort_values("start_date").reset_index(drop=True)


# --------------------------------------------------------------------------
# 5. BORROW TRANSACTIONS  (the seasonal, DNN-worthy signal lives here)
# --------------------------------------------------------------------------

def make_transactions(books_df: pd.DataFrame, students_df: pd.DataFrame):
    rows = []
    tid = 1
    dept_pref_books = {}
    for dep in DEPARTMENTS:
        cats = CATEGORIES.get(dep, ALL_CATEGORIES)
        dept_pref_books[dep] = books_df[books_df["category"].isin(cats)]["book_id"].tolist()

    student_list = students_df.to_dict("records")
    d = START_DATE
    while d <= END_DATE:
        sem_label, is_exam, is_vac = semester_info(d)

        # baseline daily borrow "budget" - varies by weekday, exam boost, vacation drop
        weekday = d.weekday()
        if weekday >= 5:            # weekend -> fewer borrows
            daily_scale = 0.35
        else:
            daily_scale = 1.0
        if is_exam:
            daily_scale *= 2.4       # exam-season spike (esp. reference/competitive books)
        if is_vac:
            daily_scale *= 0.15      # library barely used on vacation

        n_transactions_today = int(np.random.poisson(lam=55 * daily_scale))

        for _ in range(n_transactions_today):
            student = random.choice(student_list)
            dept = student["department"]
            preferred_cats = student["preferred_categories"].split(",")

            # 70% chance: borrow from own department/preferred category (realistic pattern)
            if random.random() < 0.7 and dept_pref_books.get(dept):
                candidate_pool = books_df[books_df["category"].isin(preferred_cats)]
                if candidate_pool.empty:
                    candidate_pool = books_df
            else:
                candidate_pool = books_df

            # weight by base_popularity so some books are hot, some are cold
            weights = candidate_pool["base_popularity"].values
            if weights.sum() == 0:
                continue
            weights = weights / weights.sum()
            book_row = candidate_pool.sample(n=1, weights=weights).iloc[0]

            borrow_date = d
            loan_days = random.choice([7, 14, 14, 14, 21])
            return_date = borrow_date + timedelta(days=loan_days)
            if return_date > TODAY:
                return_date = None  # still on loan

            rows.append({
                "transaction_id": tid,
                "student_id": student["student_id"],
                "book_id": int(book_row["book_id"]),
                "borrow_date": borrow_date.isoformat(),
                "return_date": return_date.isoformat() if return_date else None,
                "semester": sem_label,
                "is_exam_period": int(is_exam),
            })
            tid += 1

        d += timedelta(days=1)

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 6. AVAILABLE COPIES (currently on loan vs total)
# --------------------------------------------------------------------------

def recompute_availability(books_df, tx_df):
    on_loan = tx_df[tx_df["return_date"].isna()].groupby("book_id").size()
    books_df = books_df.copy()
    books_df["currently_on_loan"] = books_df["book_id"].map(on_loan).fillna(0).astype(int)
    books_df["available_copies"] = (books_df["total_copies"] - books_df["currently_on_loan"]).clip(lower=0)
    return books_df


# --------------------------------------------------------------------------
# 7. WRITE TO SQLITE + CSV
# --------------------------------------------------------------------------

def save_all(books_df, students_df, tx_df, subjects_df, calendar_df):
    conn = sqlite3.connect(DB_PATH)
    books_df.to_sql("books", conn, if_exists="replace", index=False)
    students_df.to_sql("students", conn, if_exists="replace", index=False)
    tx_df.to_sql("borrow_transactions", conn, if_exists="replace", index=False)
    subjects_df.to_sql("subjects", conn, if_exists="replace", index=False)
    calendar_df.to_sql("academic_calendar", conn, if_exists="replace", index=False)

    conn.execute("CREATE INDEX IF NOT EXISTS idx_tx_book ON borrow_transactions(book_id);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_tx_date ON borrow_transactions(borrow_date);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_tx_student ON borrow_transactions(student_id);")
    conn.commit()
    conn.close()

    books_df.to_csv(os.path.join(BASE_DIR, "books.csv"), index=False)
    students_df.to_csv(os.path.join(BASE_DIR, "students.csv"), index=False)
    tx_df.to_csv(os.path.join(BASE_DIR, "borrow_transactions.csv"), index=False)
    subjects_df.to_csv(os.path.join(BASE_DIR, "subjects.csv"), index=False)
    calendar_df.to_csv(os.path.join(BASE_DIR, "academic_calendar.csv"), index=False)


def main():
    print("Generating books...")
    books_df = make_books()
    print(f"  {len(books_df)} books across {books_df['category'].nunique()} categories")

    print("Generating students...")
    students_df = make_students()
    print(f"  {len(students_df)} students across {students_df['department'].nunique()} departments")

    print("Generating subjects/syllabus map...")
    subjects_df = make_subjects()
    print(f"  {len(subjects_df)} subjects across {subjects_df['department'].nunique()} departments")

    print("Generating academic calendar (past + current + future events)...")
    calendar_df = make_academic_calendar()
    print(f"  {len(calendar_df)} calendar events, up to {calendar_df['end_date'].max()}")

    print("Generating borrow transactions (this simulates ~2 academic years)...")
    tx_df = make_transactions(books_df, students_df)
    print(f"  {len(tx_df)} borrow transactions generated")

    books_df = recompute_availability(books_df, tx_df)

    print("Saving to SQLite (library.db) and CSV...")
    save_all(books_df, students_df, tx_df, subjects_df, calendar_df)
    print("Done. Files written to:", BASE_DIR)


if __name__ == "__main__":
    main()
