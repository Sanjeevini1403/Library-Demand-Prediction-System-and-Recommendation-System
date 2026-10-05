// ---------------------------------------------------------------------
// dashboard.js — talks to the Flask JSON API and renders the dashboard
// ---------------------------------------------------------------------

let META = { categories: [], departments: [], sample_student_ids: [], sample_book_ids: [] };
let trendChartRef = null, categoryChartRef = null, departmentChartRef = null;
let lastStudentId = null;

const PAGE_TITLES = {
  dashboard: "🏠 Dashboard",
  books: "📚 Books",
  circulation: "🔄 Issue & Return",
  predictions: "📈 Demand Prediction",
  forecast: "🔮 Forecast",
  inventory: "📦 Inventory",
  shortage: "🚨 Shortage Alerts",
  budget: "💰 Budget Optimizer",
  recommend: "👨‍🎓 Student Recommendations",
  readingpath: "🧠 Reading Path",
  whatif: "🔬 What-If Analysis",
  analytics: "📊 Analytics",
  modelperf: "🧪 Model Performance",
};

// ---------- Sidebar navigation ----------
document.querySelectorAll(".side-link").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".side-link").forEach(b => b.classList.remove("active"));
    document.querySelectorAll(".tab-panel").forEach(p => p.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById(btn.dataset.tab).classList.add("active");
    document.getElementById("pageTitle").textContent = PAGE_TITLES[btn.dataset.tab] || "";
    if (window.innerWidth <= 880) document.querySelector(".sidebar")?.classList.add("collapsed");

    if (btn.dataset.tab === "dashboard") loadProactiveAlerts();
    if (btn.dataset.tab === "books") loadBooksCatalog();
    if (btn.dataset.tab === "circulation") loadCirculation();
    if (btn.dataset.tab === "predictions") loadPredictions();
    if (btn.dataset.tab === "forecast") { loadForecastTrend(); loadAcademicStatus(); loadUpcomingSubjects(); }
    if (btn.dataset.tab === "inventory") loadInventoryPage();
    if (btn.dataset.tab === "shortage") loadShortagePage();
    if (btn.dataset.tab === "budget") { loadBudgetOptimizer(); loadAcquisitionPlan(); }
    if (btn.dataset.tab === "readingpath") loadReadingPath();
    if (btn.dataset.tab === "whatif") loadWhatIf();
    if (btn.dataset.tab === "analytics") { loadAnalytics(); loadDepartmentForecast(); }
    if (btn.dataset.tab === "modelperf") loadModelPerformance();
  });
});

document.getElementById("hamburgerBtn")?.addEventListener("click", () => {
  document.querySelector(".sidebar").classList.toggle("collapsed");
});

async function getJSON(url) {
  const res = await fetch(url);
  if (res.status === 401) { window.location.href = "/login"; throw new Error("unauthorized"); }
  if (!res.ok) throw new Error("Request failed: " + url);
  return res.json();
}

async function postJSON(url, body) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  if (res.status === 401) { window.location.href = "/login"; throw new Error("unauthorized"); }
  const data = await res.json().catch(() => ({}));
  return { ok: res.ok, data };
}

function exportReport(name) {
  window.location.href = "/api/export/" + name;
}

function scoreCell(score) {
  const color = score >= 70 ? "#ff6b6b" : score >= 40 ? "#ffb648" : "#5b8cff";
  return `<span class="score-bar"><span class="score-bar-fill" style="width:${score}%;background:${color}"></span></span>${score}`;
}

// ---------- Init ----------
async function init() {
  const meta = await getJSON("/api/meta");
  META = meta;

  const catOptions = meta.categories.map(c => `<option value="${c}">${c}</option>`).join("");
  const deptOptions = meta.departments.map(d => `<option value="${d}">${d}</option>`).join("");
  document.getElementById("predCategory").innerHTML += catOptions;
  document.getElementById("predDepartment").innerHTML += deptOptions;
  document.getElementById("whatifCategory").innerHTML += catOptions;
  document.getElementById("bookCategoryFilter").innerHTML += catOptions;
  document.getElementById("bookDepartmentFilter").innerHTML += deptOptions;

  document.getElementById("sampleStudentHint").textContent =
    "Try IDs: " + meta.sample_student_ids.join(", ");
  document.getElementById("sampleBookHint").textContent =
    "Try IDs: " + meta.sample_book_ids.join(", ");

  document.getElementById("syllabusDept").innerHTML += deptOptions;
  document.getElementById("readingPathCategory").innerHTML += catOptions;
  document.getElementById("newCourseDept").innerHTML = deptOptions;
  document.getElementById("newCourseCategory").innerHTML = catOptions;
  document.getElementById("enrollDept").innerHTML = deptOptions;

  await loadSummary();
  await loadTrend();
  await loadProactiveAlerts();
}

// ---------- Overview ----------
async function loadSummary() {
  const s = await getJSON("/api/summary");
  const cards = [
    { label: "Total Books", value: s.total_books },
    { label: "Total Students", value: s.total_students },
    { label: "Avg Demand Score", value: s.avg_demand_score },
    { label: "High-Demand Books", value: s.high_demand_books },
    { label: "Active Alerts", value: s.active_alerts },
    { label: "Proactive Alerts", value: s.proactive_alerts_count },
    { label: "Extra Copies Suggested", value: s.extra_copies_suggested },
  ];
  document.getElementById("summaryCards").innerHTML = cards.map(c =>
    `<div class="card"><div class="value">${c.value}</div><div class="label">${c.label}</div></div>`
  ).join("");
  document.getElementById("modeBadge").textContent = "Predictor mode: " + s.predictor_mode;

  const banner = document.getElementById("academicStatusBanner");
  let statusLine = "";
  if (s.is_exam_now) statusLine = "🔴 Semester exams are underway university-wide — demand is at its peak.";
  else if (s.is_internal_test_now) statusLine = "🟠 An internal test is underway university-wide — expect elevated demand.";
  else statusLine = "🟢 No exam/internal test running right now.";
  if (s.next_event) {
    statusLine += ` Next up: <strong>${s.next_event.event_type}</strong> in ${s.next_event.days_away} day(s) (${s.next_event.start_date}).`;
  }
  banner.innerHTML = statusLine;
}

async function loadProactiveAlerts() {
  const alerts = await getJSON("/api/proactive-alerts");
  const tbody = document.querySelector("#proactiveAlertsTable tbody");
  tbody.innerHTML = alerts.length ? alerts.slice(0, 20).map(a => `
    <tr>
      <td>${a.title}</td>
      <td>${a.category}</td>
      <td>${a.alert_type}</td>
      <td><span class="badge badge-${a.severity.toLowerCase()}">${a.severity}</span></td>
      <td>${a.message}</td>
    </tr>`).join("") : `<tr><td colspan="5" class="hint">No proactive alerts right now.</td></tr>`;
}

function renderTrendChart(canvasId, chartRef, data, label) {
  const labels = data.history.map(h => h.week);
  const values = data.history.map(h => h.demand);
  if (data.forecast) {
    labels.push(data.forecast.week);
    values.push(data.forecast.demand);
  }
  const ctx = document.getElementById(canvasId);
  if (chartRef) chartRef.destroy();
  return new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [{
        label,
        data: values,
        borderColor: "#5b8cff",
        backgroundColor: "rgba(91,140,255,0.15)",
        fill: true,
        tension: 0.3,
        pointBackgroundColor: values.map((_, i) => i === values.length - 1 ? "#7ee8c2" : "#5b8cff"),
      }]
    },
    options: {
      plugins: { legend: { labels: { color: "#e8ecf7" } } },
      scales: {
        x: { ticks: { color: "#9aa4c0" }, grid: { color: "#2a3454" } },
        y: { ticks: { color: "#9aa4c0" }, grid: { color: "#2a3454" } },
      }
    }
  });
}

// Dashboard overview: always the whole-library trend.
async function loadTrend() {
  const data = await getJSON("/api/trend");
  trendChartRef = renderTrendChart("trendChart", trendChartRef, data, "Total borrow demand (all books)");

  // Also refresh the Forecast page chart if no specific book is selected there.
  const forecastInput = document.getElementById("forecastBookIdInput");
  if (forecastInput && !forecastInput.value) {
    forecastChartRef = renderTrendChart("forecastTrendChart", forecastChartRef, data, "Total borrow demand (all books)");
  }
}

// Forecast page: whole library, or a single book if an ID is entered.
let forecastChartRef = null;
async function loadForecastTrend() {
  const bookId = document.getElementById("forecastBookIdInput").value;
  const params = new URLSearchParams();
  if (bookId) params.set("book_id", bookId);
  const data = await getJSON("/api/trend?" + params.toString());
  const label = bookId ? `Borrow demand — Book #${bookId}` : "Total borrow demand (all books)";
  forecastChartRef = renderTrendChart("forecastTrendChart", forecastChartRef, data, label);
}

// ---------- Predictions ----------
async function loadPredictions() {
  const cat = document.getElementById("predCategory").value;
  const dept = document.getElementById("predDepartment").value;
  const params = new URLSearchParams({ limit: 60 });
  if (cat) params.set("category", cat);
  if (dept) params.set("department", dept);

  const rows = await getJSON("/api/predict/enhanced?" + params.toString());
  const tbody = document.querySelector("#predictionsTable tbody");
  tbody.innerHTML = rows.map(r => `
    <tr>
      <td>${r.title}</td>
      <td>${r.category}</td>
      <td>${r.department}</td>
      <td>${r.predicted_demand}</td>
      <td>${r.adjusted_predicted_demand}${r.boost_reason ? " 🔺" : ""}</td>
      <td>${scoreCell(r.adjusted_demand_score)}</td>
      <td>${r.confidence_label || "-"}</td>
      <td>${r.available_copies} / ${r.total_copies}</td>
      <td><button onclick="loadExplain(${r.book_id})">Why?</button></td>
    </tr>`).join("");
}

async function loadExplain(bookId) {
  const box = document.getElementById("explainBox");
  box.style.display = "block";
  box.innerHTML = "Loading explanation...";
  const ex = await getJSON(`/api/explain/${bookId}`);
  const factors = ex.factor_breakdown.map(f =>
    `${f.factor}: ${f.direction} demand (${f.contribution_pct}%)`
  ).join(" &nbsp;|&nbsp; ");
  box.innerHTML = `
    <strong>Why is book #${ex.book_id}'s demand what it is?</strong><br>
    Base (DNN) predicted demand: ${ex.baseline_predicted_demand} → Context-adjusted: ${ex.context_adjusted_demand}<br>
    <em>${ex.plain_summary}</em><br>
    <span style="font-size:12px;">${factors}</span>`;
}

// ---------- Books catalog ----------
async function loadBooksCatalog() {
  const q = document.getElementById("bookSearchInput").value.trim();
  const category = document.getElementById("bookCategoryFilter").value;
  const department = document.getElementById("bookDepartmentFilter").value;
  const params = new URLSearchParams({ limit: 100 });
  if (q) params.set("q", q);
  if (category) params.set("category", category);
  if (department) params.set("department", department);

  const rows = await getJSON("/api/books/catalog?" + params.toString());
  const tbody = document.querySelector("#booksCatalogTable tbody");
  tbody.innerHTML = rows.length ? rows.map(r => `
    <tr>
      <td>${r.title}</td>
      <td>${r.author}</td>
      <td>${r.category}</td>
      <td>${r.department}</td>
      <td>${r.available_copies} / ${r.total_copies}</td>
      <td>${scoreCell(r.demand_score)}</td>
      <td><button class="btn-small" onclick="document.getElementById('bookIdInput').value=${r.book_id}; loadAlternatives();">Alternatives</button></td>
    </tr>`).join("") : `<tr><td colspan="7" class="hint">No books match that search.</td></tr>`;
}

// ---------- Shortage Alerts ----------
async function loadShortagePage() {
  const alerts = await getJSON("/api/alerts");
  const shortage = await getJSON("/api/shortage-risk");
  const anomalies = await getJSON("/api/anomalies");

  const alertsBody = document.querySelector("#alertsTable tbody");
  alertsBody.innerHTML = alerts.length ? alerts.map(a => `
    <tr>
      <td>${a.title}</td>
      <td>${a.category}</td>
      <td>${scoreCell(a.demand_score)}</td>
      <td>${a.available_copies} / ${a.total_copies}</td>
      <td><span class="badge badge-${a.severity.toLowerCase()}">${a.severity}</span></td>
    </tr>`).join("") : `<tr><td colspan="5" class="hint">No active alerts right now for the current week.</td></tr>`;

  const shortageBody = document.querySelector("#shortageTable tbody");
  shortageBody.innerHTML = shortage.length ? shortage.map(s => `
    <tr>
      <td>${s.title}</td>
      <td>${s.category}</td>
      <td>${s.available_copies}</td>
      <td>${s.adjusted_predicted_demand}</td>
      <td>${s.weeks_to_shortage}</td>
      <td><span class="badge badge-${s.risk_level.toLowerCase()}">${s.risk_level}</span></td>
    </tr>`).join("") : `<tr><td colspan="6" class="hint">No books projected to run short soon.</td></tr>`;

  const anomalyBody = document.querySelector("#anomalyTable tbody");
  anomalyBody.innerHTML = anomalies.length ? anomalies.map(a => `
    <tr>
      <td>${a.title}</td>
      <td>${a.category}</td>
      <td>${a.latest_week_demand}</td>
      <td>${a.historical_avg}</td>
      <td>${a.z_score}</td>
      <td>${a.anomaly_type}</td>
    </tr>`).join("") : `<tr><td colspan="6" class="hint">No unusual borrowing patterns detected.</td></tr>`;
}

// ---------- Inventory ----------
async function loadInventoryPage() {
  const invRows = await getJSON("/api/inventory-suggestions");
  const overstock = await getJSON("/api/overstock?limit=15");
  const lowUsage = await getJSON("/api/low-usage-books");

  const invBody = document.querySelector("#inventoryTable tbody");
  invBody.innerHTML = invRows.length ? invRows.map(r => `
    <tr>
      <td>${r.title}</td>
      <td>${r.predicted_demand}</td>
      <td>${r.total_copies}</td>
      <td><strong style="color:#7ee8c2">+${r.suggested_extra_copies}</strong></td>
    </tr>`).join("") : `<tr><td colspan="4" class="hint">No extra copies needed right now.</td></tr>`;

  const overstockBody = document.querySelector("#overstockTable tbody");
  overstockBody.innerHTML = overstock.length ? overstock.map(o => `
    <tr>
      <td>${o.title}</td>
      <td>${o.category}</td>
      <td>${o.total_copies}</td>
      <td>${scoreCell(o.adjusted_demand_score)}</td>
      <td>${o.excess_copies}</td>
      <td>${o.potential_savings}</td>
    </tr>`).join("") : `<tr><td colspan="6" class="hint">No overstocked books detected.</td></tr>`;

  const lowUsageBody = document.querySelector("#lowUsageTable tbody");
  lowUsageBody.innerHTML = lowUsage.length ? lowUsage.map(l => `
    <tr>
      <td>${l.title}</td>
      <td>${l.category}</td>
      <td>${scoreCell(l.demand_score)}</td>
      <td>${l.total_copies}</td>
      <td>${l.suggestion}</td>
    </tr>`).join("") : `<tr><td colspan="5" class="hint">No dead-stock books detected.</td></tr>`;
}

// ---------- Recommendations ----------
async function loadStudentView() {
  const sid = document.getElementById("studentIdInput").value;
  if (!sid) return alert("Enter a student ID");
  lastStudentId = sid;

  const pattern = await getJSON(`/api/student/${sid}/pattern`);
  const box = document.getElementById("readingPatternBox");
  box.style.display = "block";
  if (pattern.total_borrows === 0) {
    box.innerHTML = `Student ${sid}: no borrowing history yet — showing cold-start recommendations based on overall demand.`;
  } else {
    const cats = pattern.top_categories.map(c => `${c.category} (${c.count})`).join(", ");
    box.innerHTML = `<strong>Student ${sid}</strong> — Total borrows: ${pattern.total_borrows} &nbsp;|&nbsp; Top interests: ${cats}`;
  }

  const recs = await getJSON(`/api/student/${sid}/recommend`);
  const tbody = document.querySelector("#recommendTable tbody");
  tbody.innerHTML = recs.map(r => `
    <tr>
      <td>${r.title}</td>
      <td>${r.category}</td>
      <td>${r.interest_score}</td>
      <td>${r.availability_score === 100 ? "Available" : "Unavailable"}</td>
      <td>${scoreCell(r.demand_score)}</td>
      <td><strong>${r.final_score.toFixed(1)}</strong></td>
    </tr>`).join("");

  await loadInterestForecast(sid);
}

async function loadInterestForecast(sid) {
  const rows = await getJSON(`/api/student/${sid}/interest-forecast`);
  const tbody = document.querySelector("#interestForecastTable tbody");
  const trendColor = t => t === "Rising" || t === "New interest" ? "#7ee8c2" : t === "Fading" ? "#ff6b6b" : "#9aa4c0";
  tbody.innerHTML = rows.length ? rows.map(r => `
    <tr>
      <td>${r.category}</td>
      <td>${r.recent_borrows}</td>
      <td>${r.earlier_borrows}</td>
      <td><span style="color:${trendColor(r.trend)}">${r.trend}</span></td>
    </tr>`).join("") : `<tr><td colspan="4" class="hint">Not enough history yet to detect a trend.</td></tr>`;
}

async function loadReadingPath() {
  const sidField = document.getElementById("readingPathStudentId");
  const sid = sidField.value || lastStudentId;
  if (!sid) {
    document.getElementById("readingPathBox").textContent = "Enter a Student ID to build a reading path.";
    document.querySelector("#readingPathTable tbody").innerHTML = "";
    return;
  }
  sidField.value = sid;
  lastStudentId = sid;
  const category = document.getElementById("readingPathCategory").value;
  const params = new URLSearchParams();
  if (category) params.set("category", category);

  const result = await getJSON(`/api/student/${sid}/reading-path?` + params.toString());
  const infoBox = document.getElementById("readingPathBox");
  infoBox.textContent = result.category ? `Path for category: ${result.category}` : "No path available (no borrow history / category found).";

  const tbody = document.querySelector("#readingPathTable tbody");
  tbody.innerHTML = result.path && result.path.length ? result.path.map(p => `
    <tr>
      <td>${p.step}</td>
      <td>${p.title}</td>
      <td>${p.difficulty_level}</td>
      <td>${p.available_copies}</td>
      <td>${scoreCell(p.adjusted_demand_score)}</td>
    </tr>`).join("") : `<tr><td colspan="5" class="hint">No path available.</td></tr>`;
}

// ---------- Alternatives ----------
async function loadAlternatives() {
  const bid = document.getElementById("bookIdInput").value;
  if (!bid) return alert("Enter a book ID");
  const rows = await getJSON(`/api/book/${bid}/alternatives`);
  const tbody = document.querySelector("#alternativesTable tbody");
  tbody.innerHTML = rows.length ? rows.map(r => `
    <tr>
      <td>${r.title}</td>
      <td>${r.category}</td>
      <td>${r.match_reason}</td>
      <td>${r.available_copies > 0 ? "Yes (" + r.available_copies + ")" : "No"}</td>
      <td>${scoreCell(r.demand_score)}</td>
    </tr>`).join("") : `<tr><td colspan="5" class="hint">No alternatives found.</td></tr>`;
}

// ---------- Analytics ----------
async function loadAnalytics() {
  const data = await getJSON("/api/analytics");

  const catCtx = document.getElementById("categoryChart");
  if (categoryChartRef) categoryChartRef.destroy();
  categoryChartRef = new Chart(catCtx, {
    type: "bar",
    data: {
      labels: data.by_category.map(c => c.category),
      datasets: [{
        label: "Total predicted demand",
        data: data.by_category.map(c => c.total_predicted_demand),
        backgroundColor: "#5b8cff",
      }]
    },
    options: {
      plugins: { legend: { display: false } },
      scales: {
        x: { ticks: { color: "#9aa4c0", maxRotation: 90, minRotation: 45 }, grid: { display: false } },
        y: { ticks: { color: "#9aa4c0" }, grid: { color: "#2a3454" } },
      }
    }
  });

  const deptCtx = document.getElementById("departmentChart");
  if (departmentChartRef) departmentChartRef.destroy();
  departmentChartRef = new Chart(deptCtx, {
    type: "bar",
    data: {
      labels: data.by_department.map(d => d.department),
      datasets: [{
        label: "Total predicted demand",
        data: data.by_department.map(d => d.total_predicted_demand),
        backgroundColor: "#7ee8c2",
      }]
    },
    options: {
      plugins: { legend: { display: false } },
      scales: {
        x: { ticks: { color: "#9aa4c0", maxRotation: 90, minRotation: 45 }, grid: { display: false } },
        y: { ticks: { color: "#9aa4c0" }, grid: { color: "#2a3454" } },
      }
    }
  });
}

async function loadDepartmentForecast() {
  const rows = await getJSON("/api/department-forecast");
  const tbody = document.querySelector("#deptForecastTable tbody");
  tbody.innerHTML = rows.map(r => `
    <tr>
      <td>${r.department}</td>
      <td>${r.total_adjusted_demand}</td>
      <td>${scoreCell(Math.round(r.avg_adjusted_score))}</td>
      <td>${r.n_books}</td>
      <td>${r.primary_semester_type || "-"}</td>
      <td>${r.next_calendar_event || "-"}${r.next_event_days_away != null ? " (" + r.next_event_days_away + "d)" : ""}</td>
    </tr>`).join("");
}

// ---------- What-if ----------
async function loadWhatIf() {
  const scenario = document.getElementById("whatifScenario").value;
  const category = document.getElementById("whatifCategory").value;
  const params = new URLSearchParams({ scenario, limit: 25 });
  if (category) params.set("category", category);

  const rows = await getJSON("/api/whatif?" + params.toString());
  const tbody = document.querySelector("#whatifTable tbody");
  tbody.innerHTML = rows.map(r => `
    <tr>
      <td>${r.title}</td>
      <td>${r.category}</td>
      <td>${r.simulated_demand}</td>
      <td>${scoreCell(r.simulated_demand_score)}</td>
      <td>${r.available_copies}</td>
    </tr>`).join("");
}

async function loadNewCourseWhatIf() {
  const department = document.getElementById("newCourseDept").value;
  const category = document.getElementById("newCourseCategory").value;
  const intensity = document.getElementById("newCourseIntensity").value;
  if (!department || !category) return alert("Pick a department and category");

  const params = new URLSearchParams({ department, category, intensity });
  const rows = await getJSON("/api/whatif/new-course?" + params.toString());
  const tbody = document.querySelector("#newCourseTable tbody");
  tbody.innerHTML = rows.length ? rows.slice(0, 25).map(r => `
    <tr>
      <td>${r.title}</td>
      <td>${r.category}</td>
      <td>${r.predicted_demand}</td>
      <td>${r.simulated_demand}</td>
      <td>${scoreCell(r.simulated_demand_score)}</td>
    </tr>`).join("") : `<tr><td colspan="5" class="hint">No books found for that department/category combination.</td></tr>`;
}

async function loadEnrollmentWhatIf() {
  const department = document.getElementById("enrollDept").value;
  const pct = document.getElementById("enrollPct").value || 20;
  if (!department) return alert("Pick a department");

  const params = new URLSearchParams({ department, pct_change: pct });
  const rows = await getJSON("/api/whatif/enrollment-change?" + params.toString());
  const tbody = document.querySelector("#enrollTable tbody");
  tbody.innerHTML = rows.length ? rows.slice(0, 25).map(r => `
    <tr>
      <td>${r.title}</td>
      <td>${r.category}</td>
      <td>${r.baseline_demand}</td>
      <td>${r.simulated_demand}</td>
      <td>${scoreCell(r.simulated_demand_score)}</td>
    </tr>`).join("") : `<tr><td colspan="5" class="hint">No books found for that department.</td></tr>`;
}

// ---------- Syllabus & Calendar ----------
async function loadAcademicStatus() {
  const status = await getJSON("/api/academic-status");
  const box = document.getElementById("calendarStatusBox");
  const currentTxt = status.current_events.length
    ? status.current_events.map(e => e.event_type).join(", ")
    : "No event running right now";
  box.innerHTML = `<strong>Right now:</strong> ${currentTxt} &nbsp;(as of ${status.reference_date})`;

  const tbody = document.querySelector("#calendarTable tbody");
  const rows = [...status.current_events.map(e => ({ ...e, days_away: 0 })), ...status.upcoming_events];
  tbody.innerHTML = rows.map(e => `
    <tr>
      <td>${e.event_type}</td>
      <td>${e.semester_type}</td>
      <td>${e.start_date}</td>
      <td>${e.end_date}</td>
      <td>${e.days_away === 0 ? "Now" : e.days_away + " day(s)"}</td>
    </tr>`).join("");
}

async function loadUpcomingSubjects() {
  const department = document.getElementById("syllabusDept").value;
  const params = new URLSearchParams({ weeks_ahead: 6 });
  if (department) params.set("department", department);

  const rows = await getJSON("/api/upcoming-subjects?" + params.toString());
  const tbody = document.querySelector("#subjectsTable tbody");
  tbody.innerHTML = rows.length ? rows.map(s => `
    <tr>
      <td>${s.department}</td>
      <td>${s.subject_name}</td>
      <td>${s.category}</td>
      <td>${s.semester_type}</td>
    </tr>`).join("") : `<tr><td colspan="4" class="hint">No active/upcoming subjects found.</td></tr>`;
}

// ---------- Smart Planning ----------
async function loadAcquisitionPlan() {
  const rows = await getJSON("/api/acquisition-plan");
  const tbody = document.querySelector("#acquisitionTable tbody");
  tbody.innerHTML = rows.length ? rows.map(r => `
    <tr>
      <td>${r.title}</td>
      <td>${r.category}</td>
      <td>${r.adjusted_predicted_demand}</td>
      <td>${scoreCell(r.adjusted_demand_score)}</td>
      <td><strong style="color:#7ee8c2">${r.recommended_extra_copies}</strong></td>
      <td>${r.unit_cost}</td>
      <td>${r.estimated_cost}</td>
    </tr>`).join("") : `<tr><td colspan="7" class="hint">No acquisitions recommended right now.</td></tr>`;
}

async function loadBudgetOptimizer() {
  const budgetInput = document.getElementById("budgetInput").value;
  const params = new URLSearchParams();
  if (budgetInput) params.set("budget", budgetInput);

  const res = await getJSON("/api/budget-optimizer?" + params.toString());
  const box = document.getElementById("budgetSummaryBox");
  box.innerHTML = `Budget: ${res.total_budget} + reinvested overstock savings ${res.reinvested_overstock_savings}
    = effective ${res.effective_budget} &nbsp;|&nbsp; Spent: ${res.total_spent} &nbsp;|&nbsp; Remaining: ${res.remaining_budget}
    &nbsp;|&nbsp; ${res.unfunded.length} book(s) partially/unfunded`;

  const tbody = document.querySelector("#budgetTable tbody");
  tbody.innerHTML = res.allocations.length ? res.allocations.map(a => `
    <tr>
      <td>${a.title}</td>
      <td>${a.category}</td>
      <td>${a.copies_funded}</td>
      <td>${a.amount_spent}</td>
      <td>${a.priority_score}</td>
    </tr>`).join("") : `<tr><td colspan="5" class="hint">Nothing funded at this budget level.</td></tr>`;
}

// ---------- Circulation Desk ----------
async function loadCirculation() {
  const active = await getJSON("/api/circulation/active");
  const activeBody = document.querySelector("#activeLoansTable tbody");
  activeBody.innerHTML = active.length ? active.map(a => `
    <tr>
      <td>${a.transaction_id}</td>
      <td>${a.title}</td>
      <td>#${a.student_id}</td>
      <td>${a.student_department || "-"}</td>
      <td>${a.borrow_date}</td>
      <td><button class="btn-small" onclick="document.getElementById('returnTransactionId').value=${a.transaction_id}; returnBook();">Return</button></td>
    </tr>`).join("") : `<tr><td colspan="6" class="hint">No books currently on loan.</td></tr>`;

  const history = await getJSON("/api/circulation/history?limit=25");
  const histBody = document.querySelector("#circHistoryTable tbody");
  histBody.innerHTML = history.length ? history.map(h => `
    <tr>
      <td>${h.transaction_id}</td>
      <td>${h.title}</td>
      <td>#${h.student_id}</td>
      <td>${h.borrow_date}</td>
      <td>${h.return_date || "<em>On loan</em>"}</td>
    </tr>`).join("") : `<tr><td colspan="5" class="hint">No circulation history yet.</td></tr>`;
}

async function issueBook() {
  const student_id = document.getElementById("issueStudentId").value;
  const book_id = document.getElementById("issueBookId").value;
  const box = document.getElementById("issueResultBox");
  if (!student_id || !book_id) { alert("Enter both a Student ID and a Book ID"); return; }

  const { ok, data } = await postJSON("/api/circulation/issue", { student_id, book_id });
  box.style.display = "block";
  if (ok) {
    box.innerHTML = `✅ ${data.message} Due back by <strong>${data.due_date}</strong>. (Txn #${data.transaction_id})`;
    document.getElementById("issueBookId").value = "";
    loadCirculation();
  } else {
    box.innerHTML = `⚠️ ${data.error || "Could not issue this book."}`;
  }
}

async function returnBook() {
  const transaction_id = document.getElementById("returnTransactionId").value;
  const box = document.getElementById("returnResultBox");
  if (!transaction_id) { alert("Enter a Transaction ID"); return; }

  const { ok, data } = await postJSON("/api/circulation/return", { transaction_id });
  box.style.display = "block";
  if (ok) {
    box.innerHTML = `✅ ${data.message}`;
    document.getElementById("returnTransactionId").value = "";
    loadCirculation();
  } else {
    box.innerHTML = `⚠️ ${data.error || "Could not return this book."}`;
  }
}

// ---------- Model Performance ----------
async function loadModelPerformance() {
  const res = await getJSON("/api/model-performance");
  const banner = document.getElementById("modelModeBanner");
  const cardsBox = document.getElementById("modelMetricCards");

  if (res.predictor_mode === "dnn" && res.metrics) {
    banner.innerHTML = "🟢 A trained DNN model is loaded and being used for all predictions.";
    const m = res.metrics;
    cardsBox.innerHTML = [
      { label: "Test MAE", value: m.mae.toFixed(3) },
      { label: "Test RMSE", value: m.rmse.toFixed(3) },
      { label: "Test R²", value: m.r2.toFixed(3) },
      { label: "Training Rows", value: m.n_train },
      { label: "Test Rows", value: m.n_test },
    ].map(c => `<div class="card"><div class="value">${c.value}</div><div class="label">${c.label}</div></div>`).join("");
  } else {
    banner.innerHTML = "🟠 Running in statistical fallback mode — no trained DNN artifacts found. " +
      "Run <code>python model/train_model.py</code> to train the DNN and see real MAE/RMSE/R² here.";
    cardsBox.innerHTML = "";
  }

  const plotWrap = document.getElementById("trainingPlotWrap");
  if (res.has_training_plot) {
    document.getElementById("trainingPlotImg").src = "/model-artifacts/training_history.png?t=" + Date.now();
    plotWrap.style.display = "block";
  } else {
    plotWrap.style.display = "none";
  }

  document.getElementById("modelArchText").textContent = res.architecture;

  const tbody = document.querySelector("#featureTable tbody");
  tbody.innerHTML = res.feature_columns.map((f, i) => `<tr><td>${i + 1}</td><td>${f}</td></tr>`).join("");
}

init();
