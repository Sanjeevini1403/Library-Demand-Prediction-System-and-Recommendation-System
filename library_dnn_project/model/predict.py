"""
predict.py
----------
Inference / business-logic layer used by the Flask app.

Responsibilities:
    1. Load the trained DNN (demand_model.keras) + scaler and expose
       predict_next_week_demand(book_id).
       -> If the model hasn't been trained yet (artifacts missing) OR
          TensorFlow isn't installed on this machine, it automatically
          falls back to a transparent statistical predictor (weighted
          recent-average + exam boost) so the rest of the system
          (dashboard, recommendations, alerts) keeps working. This is
          logged clearly so it's obvious which mode is active.
    2. Compute a 0-100 "Demand Score" from the raw predicted count.
    3. High-demand / low-availability alert generation.
    4. AI-based extra-copy / inventory suggestions.
    5. Personalized recommendations = student interest (reading pattern)
       + current availability + predicted future demand combined
       (this is the project's core novelty).
    6. "Unavailable book -> similar alternative" recommendation.
    7. Category-wise / department-wise demand analytics.
    8. What-if simulation (e.g. "what happens to demand if exams start now").
"""

import json
import os
import pickle
import sqlite3
import datetime

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")
DB_PATH = os.path.join(BASE_DIR, "..", "data", "library.db")

from feature_engineering import FEATURE_COLUMNS, get_training_table

TODAY = datetime.date(2026, 8, 15)


# --------------------------------------------------------------------------
# Model loading (DNN if available, else a documented fallback)
# --------------------------------------------------------------------------

class DemandPredictor:
    def __init__(self):
        self.mode = "fallback"
        self.model = None
        self.scaler = None
        self.feature_columns = FEATURE_COLUMNS
        self.max_observed = 6.0
        self.p95 = 3.0
        self._compute_reference_stats()
        self._try_load_dnn()

    def _compute_reference_stats(self):
        """Always compute demand-score scaling reference from historical data
        (used directly in fallback mode, and as a sane default before a
        trained DNN's own reference file is loaded)."""
        try:
            hist = get_training_table()
            self.max_observed = float(hist["demand"].max())
            self.p95 = float(hist["demand"].quantile(0.95))
            if self.p95 <= 0:
                self.p95 = max(1.0, hist["demand"].mean())
        except Exception as e:
            print(f"[predict.py] Could not compute reference stats ({e}); using defaults.")

    def _try_load_dnn(self):
        model_path = os.path.join(ARTIFACT_DIR, "demand_model.keras")
        scaler_path = os.path.join(ARTIFACT_DIR, "scaler.pkl")
        cols_path = os.path.join(ARTIFACT_DIR, "feature_columns.json")
        ref_path = os.path.join(ARTIFACT_DIR, "demand_score_reference.json")

        if not (os.path.exists(model_path) and os.path.exists(scaler_path)):
            print("[predict.py] No trained DNN artifacts found in model/artifacts/. "
                  "Run `python model/train_model.py` first. "
                  "Using statistical fallback predictor for now.")
            return
        try:
            import tensorflow as tf  # noqa
            from tensorflow import keras
            self.model = keras.models.load_model(model_path)
            with open(scaler_path, "rb") as f:
                self.scaler = pickle.load(f)
            if os.path.exists(cols_path):
                with open(cols_path) as f:
                    self.feature_columns = json.load(f)
            if os.path.exists(ref_path):
                with open(ref_path) as f:
                    ref = json.load(f)
                    self.max_observed = ref.get("max_observed_weekly_demand", 6.0)
                    self.p95 = ref.get("p95_weekly_demand", 3.0)
            self.mode = "dnn"
            print("[predict.py] Loaded trained DNN model. mode=dnn")
        except Exception as e:
            print(f"[predict.py] Could not load DNN ({e}). Using statistical fallback. mode=fallback")

    def predict_batch(self, feature_df: pd.DataFrame) -> np.ndarray:
        """feature_df must contain self.feature_columns."""
        if self.mode == "dnn":
            X = feature_df[self.feature_columns].values.astype("float32")
            X_s = self.scaler.transform(X)
            preds = self.model.predict(X_s, verbose=0).flatten()
            return np.clip(preds, 0, None)
        else:
            # Transparent fallback heuristic: weighted recent trend + exam boost.
            # weights favour the most recent week, then adds an exam multiplier
            # and a small pull toward the book's long run popularity.
            preds = (
                0.45 * feature_df["lag_1"] +
                0.25 * feature_df["lag_2"] +
                0.15 * feature_df["lag_3"] +
                0.15 * feature_df["lag_4"]
            )
            preds = preds * np.where(feature_df["is_exam_period"] == 1, 1.6, 1.0)
            preds = preds * np.where(feature_df["is_vacation"] == 1, 0.3, 1.0)
            preds = preds + feature_df["base_popularity"] * 1.5
            return np.clip(preds.values, 0, None)

    def demand_score(self, predicted_demand: float) -> int:
        """Scale a raw predicted weekly-borrow-count into an interpretable 0-100 score."""
        ref = max(self.p95, 1.0) * 1.4  # anything near/above ~1.4x the 95th percentile -> near 100
        score = 100 * (1 - np.exp(-predicted_demand / ref))
        return int(round(np.clip(score, 0, 100)))


PREDICTOR = DemandPredictor()


# --------------------------------------------------------------------------
# DB helpers
# --------------------------------------------------------------------------

def get_conn():
    return sqlite3.connect(DB_PATH)


def get_books_df():
    with get_conn() as conn:
        return pd.read_sql("SELECT * FROM books", conn)


def get_students_df():
    with get_conn() as conn:
        return pd.read_sql("SELECT * FROM students", conn)


def get_transactions_df():
    with get_conn() as conn:
        df = pd.read_sql("SELECT * FROM borrow_transactions", conn)
    df["borrow_date"] = pd.to_datetime(df["borrow_date"])
    return df


# --------------------------------------------------------------------------
# Build "latest feature row" per book (what we feed the model to predict
# NEXT week's demand from where things stand right now)
# --------------------------------------------------------------------------

_LATEST_FEATURES_CACHE = {"df": None}


def get_latest_feature_rows(force_refresh: bool = False) -> pd.DataFrame:
    if _LATEST_FEATURES_CACHE["df"] is not None and not force_refresh:
        return _LATEST_FEATURES_CACHE["df"]

    full = get_training_table()  # has demand + next_week_demand for history
    # Rebuild the "current" row per book: the most recent completed week's
    # lag stack, projected one step ahead (i.e. use each book's LAST row of
    # the training table as the jumping-off point for predicting the week
    # that comes after all observed history).
    latest = full.sort_values("week").groupby("book_id").tail(1).copy()

    # shift the lag window forward by one: this week's demand becomes lag_1
    # for the prediction of the week after next.
    latest["lag_4"] = latest["lag_3"]
    latest["lag_3"] = latest["lag_2"]
    latest["lag_2"] = latest["lag_1"]
    latest["lag_1"] = latest["demand"]
    latest["rolling_avg_4"] = latest[["lag_1", "lag_2", "lag_3", "lag_4"]].mean(axis=1)

    # advance calendar features by one week
    next_week = latest["week"] + pd.Timedelta(days=7)
    month = next_week.dt.month
    latest["is_exam_period"] = month.isin([11, 4]).astype(int)
    latest["is_vacation"] = month.isin([5, 6]).astype(int)
    latest["month_sin"] = np.sin(2 * np.pi * month / 12)
    latest["month_cos"] = np.cos(2 * np.pi * month / 12)
    woy = next_week.dt.isocalendar().week.astype(int)
    latest["week_sin"] = np.sin(2 * np.pi * woy / 52)
    latest["week_cos"] = np.cos(2 * np.pi * woy / 52)
    latest["target_week"] = next_week

    _LATEST_FEATURES_CACHE["df"] = latest
    return latest


# --------------------------------------------------------------------------
# 1. Book demand prediction + demand score
# --------------------------------------------------------------------------

def predict_all_books() -> pd.DataFrame:
    latest = get_latest_feature_rows()
    preds = PREDICTOR.predict_batch(latest)
    out = latest[["book_id", "category", "department", "total_copies", "target_week"]].copy()
    out["target_week"] = out["target_week"].dt.strftime("%Y-%m-%d")
    out["predicted_demand"] = np.round(preds, 2)
    out["demand_score"] = [PREDICTOR.demand_score(p) for p in preds]

    books = get_books_df()[["book_id", "title", "author", "available_copies"]]
    out = out.merge(books, on="book_id", how="left")
    return out.sort_values("predicted_demand", ascending=False).reset_index(drop=True)


def predict_book(book_id: int) -> dict:
    df = predict_all_books()
    row = df[df["book_id"] == book_id]
    if row.empty:
        return None
    return row.iloc[0].to_dict()


# --------------------------------------------------------------------------
# 2. High-demand & low-availability alerts
# --------------------------------------------------------------------------

def get_alerts(demand_score_threshold: int = 60, availability_ratio_threshold: float = 0.25) -> pd.DataFrame:
    df = predict_all_books()
    df["availability_ratio"] = df["available_copies"] / df["total_copies"].replace(0, 1)
    alerts = df[(df["demand_score"] >= demand_score_threshold) &
                (df["availability_ratio"] <= availability_ratio_threshold)].copy()
    alerts["severity"] = np.select(
        [alerts["demand_score"] >= 85, alerts["demand_score"] >= 70],
        ["Critical", "High"],
        default="Moderate",
    )
    return alerts.sort_values("demand_score", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------
# 3. AI-based inventory / extra-copy suggestions
# --------------------------------------------------------------------------

def inventory_suggestions(min_extra: int = 1) -> pd.DataFrame:
    df = predict_all_books()
    df["availability_ratio"] = df["available_copies"] / df["total_copies"].replace(0, 1)

    # simple, explainable rule engine driven by the DNN's demand score:
    # suggest enough extra copies so that predicted demand can be served
    # without exhausting the shelf (target: available copies >= 60% of predicted demand)
    target_copies = np.ceil(df["predicted_demand"] * 0.6)
    df["suggested_extra_copies"] = np.clip(target_copies - df["total_copies"], 0, None).astype(int)
    df.loc[df["demand_score"] < 50, "suggested_extra_copies"] = 0

    result = df[df["suggested_extra_copies"] >= min_extra].copy()
    result["suggested_extra_copies"] = result["suggested_extra_copies"].clip(upper=10)
    return result.sort_values("suggested_extra_copies", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------
# 4 & 5. Student reading-pattern analysis + personalized recommendation
#          (Core novelty: interest + availability + predicted demand combined)
# --------------------------------------------------------------------------

def student_reading_pattern(student_id: int) -> dict:
    tx = get_transactions_df()
    books = get_books_df()
    student_tx = tx[tx["student_id"] == student_id].merge(books, on="book_id", how="left")

    if student_tx.empty:
        return {"student_id": student_id, "total_borrows": 0, "top_categories": [],
                "top_departments": [], "recent_books": []}

    top_categories = (student_tx["category"].value_counts()
                       .head(5).reset_index())
    top_categories.columns = ["category", "count"]

    recent = student_tx.sort_values("borrow_date", ascending=False).head(5)

    return {
        "student_id": int(student_id),
        "total_borrows": int(len(student_tx)),
        "top_categories": top_categories.to_dict("records"),
        "recent_books": recent[["title", "category", "borrow_date"]].assign(
            borrow_date=lambda d: d["borrow_date"].dt.strftime("%Y-%m-%d")
        ).to_dict("records"),
    }


def recommend_for_student(student_id: int, top_n: int = 8) -> pd.DataFrame:
    """
    NOVELTY of this project: combine three signals into one ranked list
        interest_score      -> how much this student reads this category/dept
        availability_score  -> can they actually borrow it right now
        demand_score (DNN)   -> is this book trending, worth recommending before
                                 it runs out (or a genuinely good steady pick)
    final_score = 0.45*interest + 0.20*availability + 0.35*predicted_demand_score
    """
    pattern = student_reading_pattern(student_id)
    preferred_categories = [c["category"] for c in pattern["top_categories"]]

    tx = get_transactions_df()
    already_borrowed = set(tx[tx["student_id"] == student_id]["book_id"].tolist())

    books_df = predict_all_books()
    books_df = books_df[~books_df["book_id"].isin(already_borrowed)].copy()

    if preferred_categories:
        books_df["interest_score"] = books_df["category"].apply(
            lambda c: 100 if c == preferred_categories[0]
            else (70 if c in preferred_categories[:3] else 25)
        )
    else:
        # cold-start: no history yet -> fall back to overall popularity/demand
        books_df["interest_score"] = 50

    books_df["availability_score"] = np.where(books_df["available_copies"] > 0, 100, 10)

    books_df["final_score"] = (
        0.45 * books_df["interest_score"] +
        0.20 * books_df["availability_score"] +
        0.35 * books_df["demand_score"]
    )

    cols = ["book_id", "title", "author", "category", "department",
            "available_copies", "demand_score", "interest_score",
            "availability_score", "final_score"]
    return books_df.sort_values("final_score", ascending=False)[cols].head(top_n).reset_index(drop=True)


# --------------------------------------------------------------------------
# 6. Unavailable book -> similar alternative recommendation
# --------------------------------------------------------------------------

def find_alternatives(book_id: int, top_n: int = 5) -> pd.DataFrame:
    books = get_books_df()
    target = books[books["book_id"] == book_id]
    if target.empty:
        return pd.DataFrame()
    target = target.iloc[0]

    preds = predict_all_books()

    same_category = preds[(preds["category"] == target["category"]) &
                           (preds["book_id"] != book_id)].copy()
    same_category["match_reason"] = "Same category: " + target["category"]

    if len(same_category) < top_n:
        same_dept = preds[(preds["department"] == target["department"]) &
                           (preds["book_id"] != book_id) &
                           (~preds["book_id"].isin(same_category["book_id"]))].copy()
        same_dept["match_reason"] = "Same department: " + target["department"]
        same_category = pd.concat([same_category, same_dept], ignore_index=True)

    # prefer alternatives that are BOTH available and reasonably in-demand (well reviewed / popular)
    same_category["alt_score"] = (
        (same_category["available_copies"] > 0).astype(int) * 50 +
        same_category["demand_score"] * 0.5
    )
    return same_category.sort_values("alt_score", ascending=False).head(top_n).reset_index(drop=True)


# --------------------------------------------------------------------------
# 7. Category-wise & department-wise demand analytics
# --------------------------------------------------------------------------

def category_department_analytics() -> dict:
    preds = predict_all_books()
    by_category = (preds.groupby("category")
                   .agg(avg_demand_score=("demand_score", "mean"),
                        total_predicted_demand=("predicted_demand", "sum"),
                        n_books=("book_id", "count"))
                   .reset_index().sort_values("total_predicted_demand", ascending=False))
    by_department = (preds.groupby("department")
                     .agg(avg_demand_score=("demand_score", "mean"),
                          total_predicted_demand=("predicted_demand", "sum"),
                          n_books=("book_id", "count"))
                     .reset_index().sort_values("total_predicted_demand", ascending=False))
    return {
        "by_category": by_category.round(2).to_dict("records"),
        "by_department": by_department.round(2).to_dict("records"),
    }


# --------------------------------------------------------------------------
# 8. Past vs future demand analytics (for dashboard trend chart)
# --------------------------------------------------------------------------

def past_vs_future_trend(book_id: int = None, weeks_back: int = 16):
    full = get_training_table()
    if book_id is not None:
        full = full[full["book_id"] == book_id]
    trend = full.groupby("week")["demand"].sum().reset_index().sort_values("week").tail(weeks_back)
    trend["week"] = trend["week"].dt.strftime("%Y-%m-%d")

    if book_id is not None:
        future = predict_book(book_id)
        future_point = {"week": str(future["target_week"])[:10], "demand": future["predicted_demand"]} if future else None
    else:
        preds = predict_all_books()
        future_point = {"week": "next_week (forecast)", "demand": round(float(preds["predicted_demand"].sum()), 1)}

    return {"history": trend.to_dict("records"), "forecast": future_point}


# --------------------------------------------------------------------------
# 9. What-if prediction (exam season / future demand simulation)
# --------------------------------------------------------------------------

def what_if_simulation(scenario: str = "exam_season", category: str = None) -> pd.DataFrame:
    """
    scenario:
        "exam_season" -> force is_exam_period=1, is_vacation=0
        "vacation"    -> force is_vacation=1, is_exam_period=0
        "normal"      -> force both 0
    Re-runs the SAME trained model with the calendar flags flipped, so the
    output demonstrates the DNN's learned response to seasonality, isolated
    from the current real calendar date.
    """
    latest = get_latest_feature_rows().copy()
    if category:
        latest = latest[latest["category"] == category]

    if scenario == "exam_season":
        latest["is_exam_period"] = 1
        latest["is_vacation"] = 0
    elif scenario == "vacation":
        latest["is_exam_period"] = 0
        latest["is_vacation"] = 1
    else:
        latest["is_exam_period"] = 0
        latest["is_vacation"] = 0

    preds = PREDICTOR.predict_batch(latest)
    out = latest[["book_id", "category", "department", "total_copies", "available_copies"]].copy() \
        if "available_copies" in latest.columns else latest[["book_id", "category", "department", "total_copies"]].copy()
    books = get_books_df()[["book_id", "title", "available_copies"]]
    out = latest[["book_id", "category", "department"]].merge(books, on="book_id", how="left")
    out["simulated_demand"] = np.round(preds, 2)
    out["simulated_demand_score"] = [PREDICTOR.demand_score(p) for p in preds]
    return out.sort_values("simulated_demand", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------
# BONUS 10. Low-usage / "dead stock" detector (weeding & donation suggestions)
#            -> the natural mirror image of inventory_suggestions(): instead
#               of buying MORE copies of hot books, flag books nobody wants
#               so the library can free up shelf space / budget.
# --------------------------------------------------------------------------

def low_usage_books(demand_score_threshold: int = 15, min_copies: int = 2) -> pd.DataFrame:
    df = predict_all_books()
    dead = df[(df["demand_score"] <= demand_score_threshold) &
              (df["total_copies"] >= min_copies)].copy()
    dead["suggestion"] = "Consider reducing copies / relocating to archive"
    return dead.sort_values("demand_score").reset_index(drop=True)


if __name__ == "__main__":
    print("Predictor mode:", PREDICTOR.mode)
    df = predict_all_books()
    print(df.head(10))
    print("\nTop alerts:")
    print(get_alerts().head(5))
