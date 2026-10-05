"""
app.py
------
Flask backend for the Library Book Demand Prediction & Intelligent
Recommendation System.

Serves:
    - the dashboard UI (templates/index.html)
    - a REST JSON API used by static/js/dashboard.js

Run:
    python app.py
Then open:
    http://127.0.0.1:5000
"""

import sys
import os
import io
import csv
import datetime
import functools

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "model"))

from flask import (
    Flask, jsonify, render_template, request, redirect, url_for,
    session, Response, send_from_directory
)

import predict as P  # model/predict.py
import academic_context as AC
import advanced_features as ADV
from feature_engineering import FEATURE_COLUMNS

app = Flask(__name__)
app.secret_key = os.environ.get("LIBRARY_APP_SECRET", "college-project-dev-secret-key")

# --------------------------------------------------------------------------
# Librarian login (simple session-based auth)
# --------------------------------------------------------------------------

LIBRARIAN_USERNAME = os.environ.get("LIBRARIAN_USERNAME", "admin")
LIBRARIAN_PASSWORD = os.environ.get("LIBRARIAN_PASSWORD", "admin123")


@app.before_request
def require_login():
    open_endpoints = {"login", "static", "model_artifact_file"}
    if request.endpoint is None or request.endpoint in open_endpoints:
        return
    if request.endpoint.startswith("static"):
        return
    if not session.get("logged_in"):
        if request.path.startswith("/api/"):
            return jsonify({"error": "unauthorized", "login_required": True}), 401
        return redirect(url_for("login", next=request.path))


@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        u = request.form.get("username", "")
        p = request.form.get("password", "")
        if u == LIBRARIAN_USERNAME and p == LIBRARIAN_PASSWORD:
            session["logged_in"] = True
            session["username"] = u
            return redirect(request.args.get("next") or url_for("index"))
        error = "Invalid username or password."
    return render_template("login.html", error=error)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# --------------------------------------------------------------------------
# Dashboard page
# --------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html", predictor_mode=P.PREDICTOR.mode,
                            username=session.get("username", "Librarian"))


# --------------------------------------------------------------------------
# API: summary (for top cards on dashboard)
# --------------------------------------------------------------------------

@app.route("/api/summary")
def api_summary():
    df = P.predict_all_books()
    alerts = P.get_alerts()
    inv = P.inventory_suggestions()
    books = P.get_books_df()
    students = P.get_students_df()
    status = AC.current_academic_status()
    proactive = ADV.proactive_alerts()

    return jsonify({
        "predictor_mode": P.PREDICTOR.mode,
        "total_books": int(len(books)),
        "total_students": int(len(students)),
        "avg_demand_score": round(float(df["demand_score"].mean()), 1),
        "high_demand_books": int((df["demand_score"] >= 60).sum()),
        "active_alerts": int(len(alerts)),
        "extra_copies_suggested": int(inv["suggested_extra_copies"].sum()) if len(inv) else 0,
        "proactive_alerts_count": len(proactive),
        "is_exam_now": status["is_exam_now"],
        "is_internal_test_now": status["is_internal_test_now"],
        "next_event": status["upcoming_events"][0] if status["upcoming_events"] else None,
    })


# --------------------------------------------------------------------------
# API 1: DNN-based book demand prediction (all books, or one book)
# --------------------------------------------------------------------------

@app.route("/api/predict")
def api_predict_all():
    limit = int(request.args.get("limit", 50))
    category = request.args.get("category")
    department = request.args.get("department")

    df = P.predict_all_books()
    if category:
        df = df[df["category"] == category]
    if department:
        df = df[df["department"] == department]

    df = df.head(limit)
    return jsonify(df.to_dict("records"))


@app.route("/api/predict/<int:book_id>")
def api_predict_one(book_id):
    result = P.predict_book(book_id)
    if result is None:
        return jsonify({"error": "book not found"}), 404
    return jsonify(result)


# --------------------------------------------------------------------------
# API 2: High-demand & low-availability alerts
# --------------------------------------------------------------------------

@app.route("/api/alerts")
def api_alerts():
    threshold = int(request.args.get("threshold", 60))
    avail_ratio = float(request.args.get("availability_ratio", 0.25))
    df = P.get_alerts(demand_score_threshold=threshold, availability_ratio_threshold=avail_ratio)
    return jsonify(df.to_dict("records"))


# --------------------------------------------------------------------------
# API 3: AI-based inventory / extra-copy suggestions
# --------------------------------------------------------------------------

@app.route("/api/inventory-suggestions")
def api_inventory():
    df = P.inventory_suggestions()
    return jsonify(df.to_dict("records"))


# Bonus: low-usage / "dead stock" books -> weeding & donation suggestions
@app.route("/api/low-usage-books")
def api_low_usage():
    df = P.low_usage_books()
    return jsonify(df.to_dict("records"))


# --------------------------------------------------------------------------
# API 4: Student reading-pattern analysis
# --------------------------------------------------------------------------

@app.route("/api/student/<int:student_id>/pattern")
def api_student_pattern(student_id):
    return jsonify(P.student_reading_pattern(student_id))


# --------------------------------------------------------------------------
# API 5: Personalized recommendation (core novelty)
# --------------------------------------------------------------------------

@app.route("/api/student/<int:student_id>/recommend")
def api_recommend(student_id):
    top_n = int(request.args.get("top_n", 8))
    df = P.recommend_for_student(student_id, top_n=top_n)
    return jsonify(df.to_dict("records"))


# --------------------------------------------------------------------------
# API 6: Unavailable book -> similar alternative
# --------------------------------------------------------------------------

@app.route("/api/book/<int:book_id>/alternatives")
def api_alternatives(book_id):
    top_n = int(request.args.get("top_n", 5))
    df = P.find_alternatives(book_id, top_n=top_n)
    return jsonify(df.to_dict("records"))


# --------------------------------------------------------------------------
# API 7: Category-wise & department-wise demand analytics
# --------------------------------------------------------------------------

@app.route("/api/analytics")
def api_analytics():
    return jsonify(P.category_department_analytics())


# --------------------------------------------------------------------------
# API 8: Past vs future demand trend (dashboard chart)
# --------------------------------------------------------------------------

@app.route("/api/trend")
def api_trend():
    book_id = request.args.get("book_id", type=int)
    weeks_back = int(request.args.get("weeks_back", 16))
    return jsonify(P.past_vs_future_trend(book_id=book_id, weeks_back=weeks_back))


# --------------------------------------------------------------------------
# API 9: What-if prediction (exam season / vacation / normal simulation)
# --------------------------------------------------------------------------

@app.route("/api/whatif")
def api_whatif():
    scenario = request.args.get("scenario", "exam_season")
    category = request.args.get("category")
    limit = int(request.args.get("limit", 20))
    df = P.what_if_simulation(scenario=scenario, category=category)
    return jsonify(df.head(limit).to_dict("records"))


# --------------------------------------------------------------------------
# Misc: lists used to populate dashboard dropdowns
# --------------------------------------------------------------------------

@app.route("/api/meta")
def api_meta():
    books = P.get_books_df()
    students = P.get_students_df()
    return jsonify({
        "categories": sorted(books["category"].unique().tolist()),
        "departments": sorted(books["department"].unique().tolist()),
        "sample_student_ids": sorted(students["student_id"].sample(
            min(10, len(students)), random_state=1).tolist()),
        "sample_book_ids": sorted(books["book_id"].sample(
            min(10, len(books)), random_state=1).tolist()),
    })


@app.route("/api/books/search")
def api_book_search():
    q = request.args.get("q", "").strip().lower()
    books = P.get_books_df()
    if q:
        books = books[books["title"].str.lower().str.contains(q) |
                      books["author"].str.lower().str.contains(q)]
    cols = ["book_id", "title", "author", "category", "department", "available_copies", "total_copies"]
    return jsonify(books[cols].head(20).to_dict("records"))


# ---- Books catalog page (search + filter + live demand score) ----

@app.route("/api/books/catalog")
def api_books_catalog():
    q = request.args.get("q", "").strip().lower()
    category = request.args.get("category")
    department = request.args.get("department")
    limit = int(request.args.get("limit", 100))

    df = P.predict_all_books()
    if q:
        df = df[df["title"].str.lower().str.contains(q) |
                df["author"].str.lower().str.contains(q)]
    if category:
        df = df[df["category"] == category]
    if department:
        df = df[df["department"] == department]

    cols = ["book_id", "title", "author", "category", "department",
            "available_copies", "total_copies", "demand_score"]
    return jsonify(df[cols].head(limit).to_dict("records"))


# ==========================================================================
# NEW ADVANCED-FEATURE ENDPOINTS
# (core endpoints above are untouched - these are purely additive)
# ==========================================================================

# ---- Syllabus-Aware Prediction + Academic Calendar Prediction ----

@app.route("/api/academic-status")
def api_academic_status():
    return jsonify(AC.current_academic_status())


@app.route("/api/upcoming-subjects")
def api_upcoming_subjects():
    department = request.args.get("department")
    weeks_ahead = int(request.args.get("weeks_ahead", 6))
    df = AC.upcoming_subjects(weeks_ahead=weeks_ahead, department=department)
    return jsonify(df.to_dict("records"))


@app.route("/api/predict/enhanced")
def api_predict_enhanced():
    limit = int(request.args.get("limit", 50))
    category = request.args.get("category")
    department = request.args.get("department")
    df = ADV.get_enhanced_predictions()
    if category:
        df = df[df["category"] == category]
    if department:
        df = df[df["department"] == department]
    df = df.sort_values("adjusted_predicted_demand", ascending=False).head(limit)
    return jsonify(df.to_dict("records"))


# ---- Explainable AI ----

@app.route("/api/explain/<int:book_id>")
def api_explain(book_id):
    result = ADV.explain_demand(book_id)
    if result is None:
        return jsonify({"error": "book not found"}), 404
    return jsonify(result)


# ---- Early Shortage Prediction ----

@app.route("/api/shortage-risk")
def api_shortage_risk():
    weeks_ahead = int(request.args.get("weeks_ahead", 3))
    df = ADV.early_shortage_prediction(weeks_ahead=weeks_ahead)
    return jsonify(df.to_dict("records"))


# ---- Smart Acquisition Planner ----

@app.route("/api/acquisition-plan")
def api_acquisition_plan():
    horizon_weeks = int(request.args.get("horizon_weeks", 4))
    df = ADV.smart_acquisition_planner(horizon_weeks=horizon_weeks)
    return jsonify(df.to_dict("records"))


# ---- Overstock Detection ----

@app.route("/api/overstock")
def api_overstock():
    limit = int(request.args.get("limit", 30))
    df = ADV.overstock_detection()
    return jsonify(df.head(limit).to_dict("records"))


# ---- Budget Optimizer ----

@app.route("/api/budget-optimizer")
def api_budget_optimizer():
    budget = request.args.get("budget", type=float)
    return jsonify(ADV.budget_optimizer(total_budget=budget))


# ---- Student Interest Forecasting ----

@app.route("/api/student/<int:student_id>/interest-forecast")
def api_interest_forecast(student_id):
    return jsonify(ADV.student_interest_forecast(student_id))


# ---- Personalized Reading Path ----

@app.route("/api/student/<int:student_id>/reading-path")
def api_reading_path(student_id):
    category = request.args.get("category")
    path_length = int(request.args.get("path_length", 5))
    return jsonify(ADV.personalized_reading_path(student_id, category=category, path_length=path_length))


# ---- Anomaly Detection ----

@app.route("/api/anomalies")
def api_anomalies():
    z = float(request.args.get("z_threshold", 2.5))
    df = ADV.anomaly_detection(z_threshold=z)
    return jsonify(df.to_dict("records"))


# ---- Department-wise Semester Forecasting ----

@app.route("/api/department-forecast")
def api_department_forecast():
    return jsonify(ADV.department_semester_forecast())


# ---- Proactive Alerts (aggregator) ----

@app.route("/api/proactive-alerts")
def api_proactive_alerts():
    return jsonify(ADV.proactive_alerts())


# ---- Extra What-If Scenarios ----

@app.route("/api/whatif/new-course")
def api_whatif_new_course():
    department = request.args.get("department")
    category = request.args.get("category")
    intensity = request.args.get("intensity", "internal_test")
    if not department or not category:
        return jsonify({"error": "department and category are required"}), 400
    df = ADV.what_if_new_course(department, category, intensity=intensity)
    return jsonify(df.to_dict("records"))


@app.route("/api/whatif/enrollment-change")
def api_whatif_enrollment():
    department = request.args.get("department")
    pct_change = request.args.get("pct_change", type=float, default=20.0)
    if not department:
        return jsonify({"error": "department is required"}), 400
    df = ADV.what_if_enrollment_change(department, pct_change)
    return jsonify(df.to_dict("records"))


# ==========================================================================
# CIRCULATION DESK — issue / return books (turns the analytics dashboard
# into a working circulation system; updates books.available_copies and
# inserts real rows into borrow_transactions)
# ==========================================================================

def _current_semester_context():
    status = AC.current_academic_status()
    if status["current_events"]:
        semester_type = status["current_events"][0].get("semester_type", "Odd-Semester")
    else:
        semester_type = "Odd-Semester"
    is_exam_period = 1 if status["is_exam_now"] else 0
    return semester_type, is_exam_period


@app.route("/api/circulation/active")
def api_circulation_active():
    with P.get_conn() as conn:
        import pandas as pd
        df = pd.read_sql("""
            SELECT t.transaction_id, t.book_id, t.student_id, t.borrow_date,
                   b.title, b.author, s.department AS student_department
            FROM borrow_transactions t
            JOIN books b ON b.book_id = t.book_id
            LEFT JOIN students s ON s.student_id = t.student_id
            WHERE t.return_date IS NULL OR t.return_date = ''
            ORDER BY t.borrow_date DESC
        """, conn)
    return jsonify(df.to_dict("records"))


@app.route("/api/circulation/history")
def api_circulation_history():
    limit = int(request.args.get("limit", 25))
    with P.get_conn() as conn:
        import pandas as pd
        df = pd.read_sql("""
            SELECT t.transaction_id, t.book_id, t.student_id, t.borrow_date, t.return_date,
                   b.title
            FROM borrow_transactions t
            JOIN books b ON b.book_id = t.book_id
            ORDER BY t.transaction_id DESC
            LIMIT ?
        """, conn, params=(limit,))
    return jsonify(df.to_dict("records"))


@app.route("/api/circulation/issue", methods=["POST"])
def api_circulation_issue():
    data = request.get_json(force=True, silent=True) or {}
    book_id = data.get("book_id")
    student_id = data.get("student_id")
    if not book_id or not student_id:
        return jsonify({"error": "book_id and student_id are required"}), 400

    with P.get_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT title, available_copies FROM books WHERE book_id = ?", (book_id,))
        row = cur.fetchone()
        if row is None:
            return jsonify({"error": "book not found"}), 404
        title, available = row
        if available <= 0:
            return jsonify({"error": f'"{title}" has no available copies right now.'}), 409

        cur.execute("SELECT student_id FROM students WHERE student_id = ?", (student_id,))
        if cur.fetchone() is None:
            return jsonify({"error": "student not found"}), 404

        semester_type, is_exam_period = _current_semester_context()
        borrow_date = str(AC.TODAY)

        cur.execute("SELECT COALESCE(MAX(transaction_id), 0) + 1 FROM borrow_transactions")
        transaction_id = cur.fetchone()[0]

        cur.execute("""
            INSERT INTO borrow_transactions
                (transaction_id, student_id, book_id, borrow_date, return_date, semester, is_exam_period)
            VALUES (?, ?, ?, ?, NULL, ?, ?)
        """, (transaction_id, student_id, book_id, borrow_date, semester_type, is_exam_period))
        cur.execute("UPDATE books SET available_copies = available_copies - 1 WHERE book_id = ?", (book_id,))
        conn.commit()

    return jsonify({
        "success": True,
        "transaction_id": transaction_id,
        "message": f'Issued "{title}" to student #{student_id}.',
        "due_date": str(AC.TODAY + datetime.timedelta(days=14)),
    })


@app.route("/api/circulation/return", methods=["POST"])
def api_circulation_return():
    data = request.get_json(force=True, silent=True) or {}
    transaction_id = data.get("transaction_id")
    if not transaction_id:
        return jsonify({"error": "transaction_id is required"}), 400

    with P.get_conn() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT book_id, return_date FROM borrow_transactions WHERE transaction_id = ?
        """, (transaction_id,))
        row = cur.fetchone()
        if row is None:
            return jsonify({"error": "transaction not found"}), 404
        book_id, return_date = row
        if return_date:
            return jsonify({"error": "this book was already returned"}), 409

        return_dt = str(AC.TODAY)
        cur.execute("UPDATE borrow_transactions SET return_date = ? WHERE transaction_id = ?",
                    (return_dt, transaction_id))
        cur.execute("""
            UPDATE books SET available_copies = MIN(available_copies + 1, total_copies)
            WHERE book_id = ?
        """, (book_id,))
        conn.commit()

    return jsonify({"success": True, "message": "Book returned successfully."})


# ==========================================================================
# CSV EXPORT — download any of the key report tables as a spreadsheet-ready
# CSV file, for offline reporting / attaching to a project report.
# ==========================================================================

EXPORTABLE_REPORTS = {
    "demand-prediction": lambda: ADV.get_enhanced_predictions(),
    "alerts": lambda: P.get_alerts(),
    "inventory": lambda: P.inventory_suggestions(),
    "overstock": lambda: ADV.overstock_detection(),
    "low-usage": lambda: P.low_usage_books(),
    "shortage-risk": lambda: ADV.early_shortage_prediction(),
    "acquisition-plan": lambda: ADV.smart_acquisition_planner(),
}


@app.route("/api/export/<report_name>")
def api_export_report(report_name):
    if report_name not in EXPORTABLE_REPORTS:
        return jsonify({"error": "unknown report", "available": list(EXPORTABLE_REPORTS.keys())}), 404

    df = EXPORTABLE_REPORTS[report_name]()
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    filename = f"{report_name}_{AC.TODAY}.csv"
    return Response(
        buf.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


# ==========================================================================
# MODEL PERFORMANCE — surfaces the DNN's own training metrics (MAE/RMSE/R2),
# the loss curve, and the feature set, for the project viva/report.
# ==========================================================================

@app.route("/api/model-performance")
def api_model_performance():
    import json
    metrics_path = os.path.join(P.ARTIFACT_DIR, "metrics.json")
    plot_path = os.path.join(P.ARTIFACT_DIR, "training_history.png")

    metrics = None
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)

    return jsonify({
        "predictor_mode": P.PREDICTOR.mode,
        "metrics": metrics,
        "has_training_plot": os.path.exists(plot_path),
        "feature_columns": FEATURE_COLUMNS,
        "n_features": len(FEATURE_COLUMNS),
        "architecture": "Input -> Dense(64, relu) -> BatchNorm -> Dropout(0.2) -> "
                        "Dense(32, relu) -> BatchNorm -> Dropout(0.2) -> Dense(16, relu) -> Dense(1, relu)",
    })


@app.route("/model-artifacts/<path:filename>")
def model_artifact_file(filename):
    return send_from_directory(P.ARTIFACT_DIR, filename)


if __name__ == "__main__":
    print(f"Predictor running in mode: {P.PREDICTOR.mode}")
    app.run(debug=True, host="0.0.0.0", port=5000)
