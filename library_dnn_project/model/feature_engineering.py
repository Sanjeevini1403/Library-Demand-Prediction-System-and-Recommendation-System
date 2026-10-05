"""
feature_engineering.py
-----------------------
Turns raw borrow_transactions into a WEEKLY per-book demand table that the
DNN is trained on.

For every (book, week) we build:
    lag_1, lag_2, lag_3, lag_4   -> borrow counts of the previous 4 weeks
    rolling_avg_4                -> rolling mean of previous 4 weeks
    is_exam_period                -> 1 if this week falls in an exam window
    is_vacation                   -> 1 if this week falls in vacation
    month_sin / month_cos         -> cyclical month encoding
    week_of_year_sin/cos          -> cyclical week encoding
    category_encoded               -> integer-encoded category
    department_encoded             -> integer-encoded department
    total_copies                   -> catalog size (supply signal)
    base_popularity                -> long-run popularity prior
    dept_week_demand               -> total borrows across the book's dept that week
                                       (captures department-wide exam pressure)
Target:
    next_week_demand -> borrow count in the FOLLOWING week (regression target)

This is exactly the kind of tabular, multi-factor, non-linear forecasting
problem a Dense Neural Network (DNN) is well suited for: many heterogeneous
signals (temporal lags + categorical + cyclical + popularity) need to be
combined non-linearly, which classical single-formula heuristics cannot
capture well, and dataset size (thousands of book-weeks) is enough to train
a small DNN without overfitting.
"""

import os
import sqlite3

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "..", "data", "library.db")


def load_raw():
    conn = sqlite3.connect(DB_PATH)
    tx = pd.read_sql("SELECT * FROM borrow_transactions", conn)
    books = pd.read_sql("SELECT * FROM books", conn)
    conn.close()
    tx["borrow_date"] = pd.to_datetime(tx["borrow_date"])
    return tx, books


def build_weekly_demand(tx: pd.DataFrame, books: pd.DataFrame) -> pd.DataFrame:
    tx = tx.copy()
    tx["week"] = tx["borrow_date"].dt.to_period("W-SUN").apply(lambda p: p.start_time)

    # full grid of book x week so books with zero borrows in a week are included
    all_weeks = pd.date_range(tx["week"].min(), tx["week"].max(), freq="7D")
    all_books = books["book_id"].unique()
    grid = pd.MultiIndex.from_product([all_books, all_weeks], names=["book_id", "week"]).to_frame(index=False)

    weekly_counts = tx.groupby(["book_id", "week"]).size().reset_index(name="demand")
    df = grid.merge(weekly_counts, on=["book_id", "week"], how="left")
    df["demand"] = df["demand"].fillna(0).astype(int)

    df = df.merge(books[["book_id", "category", "department", "total_copies", "base_popularity"]],
                   on="book_id", how="left")

    # exam / vacation flags per week (majority day-of-week rule reused from generator logic)
    def week_flags(week_start):
        month = week_start.month
        if month == 6 or month == 5:
            return 0, 1
        if month in (11, 4):
            return 1, 0
        return 0, 0

    flags = df["week"].apply(week_flags)
    df["is_exam_period"] = flags.apply(lambda x: x[0])
    df["is_vacation"] = flags.apply(lambda x: x[1])

    df["month"] = df["week"].dt.month
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    woy = df["week"].dt.isocalendar().week.astype(int)
    df["week_sin"] = np.sin(2 * np.pi * woy / 52)
    df["week_cos"] = np.cos(2 * np.pi * woy / 52)

    df = df.sort_values(["book_id", "week"]).reset_index(drop=True)

    # lag features (per book)
    for lag in (1, 2, 3, 4):
        df[f"lag_{lag}"] = df.groupby("book_id")["demand"].shift(lag)
    df["rolling_avg_4"] = df.groupby("book_id")["demand"].transform(
        lambda s: s.shift(1).rolling(4, min_periods=1).mean()
    )

    # department-week aggregate demand (captures dept-wide exam pressure)
    dept_week = df.groupby(["department", "week"])["demand"].sum().reset_index(name="dept_week_demand")
    df = df.merge(dept_week, on=["department", "week"], how="left")

    # target = NEXT week's demand
    df["next_week_demand"] = df.groupby("book_id")["demand"].shift(-1)

    # categorical encodings
    df["category_encoded"] = df["category"].astype("category").cat.codes
    df["department_encoded"] = df["department"].astype("category").cat.codes

    # drop rows without enough lag history or without a target (start/end of series)
    df = df.dropna(subset=["lag_4", "next_week_demand"]).reset_index(drop=True)

    return df


FEATURE_COLUMNS = [
    "lag_1", "lag_2", "lag_3", "lag_4", "rolling_avg_4",
    "is_exam_period", "is_vacation",
    "month_sin", "month_cos", "week_sin", "week_cos",
    "category_encoded", "department_encoded",
    "total_copies", "base_popularity", "dept_week_demand",
]
TARGET_COLUMN = "next_week_demand"


def get_training_table():
    tx, books = load_raw()
    df = build_weekly_demand(tx, books)
    return df


if __name__ == "__main__":
    df = get_training_table()
    print(df.shape, "rows x cols")
    print(df[FEATURE_COLUMNS + [TARGET_COLUMN]].describe().T)
    out = os.path.join(BASE_DIR, "..", "data", "weekly_training_table.csv")
    df.to_csv(out, index=False)
    print("Saved:", out)
