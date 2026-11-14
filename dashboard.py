# dashboard.py - Autonomous Retail Intelligence Dashboard (advanced version)
import os
import sys
import time
import subprocess
import numpy as np
import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ---- Optional layers: the dashboard still opens if any of these are missing ----
try:
    from insights import score, generate_ml_insights, METRICS, IMPORTANCE
    MODEL_READY = True
except Exception:
    MODEL_READY = False

LOAD_ERR = ""
try:
    import analytics
    import advanced as AD
    ANALYTICS_READY = True
except Exception as e:
    ANALYTICS_READY = False
    LOAD_ERR = str(e)

try:
    import gemini_advisor as ga
    GEMINI_READY = True
except Exception:
    GEMINI_READY = False

try:
    import period_compare as PC
    PC_READY = True
except Exception:
    PC_READY = False

st.set_page_config(page_title="Autonomous Retail Intelligence", layout="wide")
st.title("Autonomous Retail Intelligence System")
st.markdown("Real-Time Store Analytics & Customer Footfall Insights")

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Data Source")
    source_label = st.radio("Show data from",
                            ["Mock data (75 days)", "Live video pipeline"])
    source = "mock" if source_label.startswith("Mock") else "video"
    auto_refresh = False
    uploaded, run_clicked, fast = None, False, True
    if source == "video":
        st.caption("Rows written by vision_pipeline.py from a video (video_data.db).")
        st.markdown("**Upload a video**")
        uploaded = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov", "mkv"])
        fast = st.checkbox("Faster processing (analyse fewer frames)", value=True)
        run_clicked = st.button("Process video", disabled=uploaded is None)
        auto_refresh = st.checkbox("Auto-refresh every 5 s (live demo)")
    else:
        st.caption("Generated data that stands in for a long retail recording (retail_data.db).")
    st.divider()

    st.header("Analysis Period")
    period_choice = st.radio("Show data for", ["All data", "Last 30 days", "Last 7 days"])
    period_days = {"All data": None, "Last 30 days": 30, "Last 7 days": 7}[period_choice]

    st.header("Business Assumptions")
    promo_cost = st.number_input("Promotion cost per visitor (Rs)", 0.0, 500.0, 3.0, 0.5)
    engage_cost = st.number_input("Engagement cost per affected visitor (Rs)", 0.0, 500.0, 1.0, 0.5)
    assist_lift = st.slider("Assumed lift from staff assistance (points)", 0, 30, 10)
    capacity = st.number_input("Visitors one staff member can serve per hour", 1, 100, 5)
    st.caption("These are your own assumptions. Change them and every tab updates.")

    # The key box only appears if no key is set in the environment
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        st.header("AI Advisor")
        api_key = st.text_input("AI access key (kept only in this session)", type="password")

# ---------------- Process an uploaded video (runs vision_pipeline.py) ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if source == "video" and run_clicked and uploaded is not None:
    video_path = os.path.join(BASE_DIR, "uploaded_video.mp4")
    with open(video_path, "wb") as f:
        f.write(uploaded.getbuffer())
    cmd = [sys.executable, os.path.join(BASE_DIR, "vision_pipeline.py"),
           "--video", video_path, "--fresh", "--no-show",
           "--skip", "3" if fast else "2"]
    with st.spinner("Processing video: detecting, tracking and logging visitors. This can take a few minutes..."):
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=BASE_DIR)
    if result.returncode == 0:
        st.success("Video processed. The data below is from your video.")
    else:
        st.error("The vision pipeline failed. Details below.")
        st.code((result.stderr or result.stdout)[-2000:])

API_URL = "http://127.0.0.1:8000"

# ---------------- Fetch data from FastAPI backend ----------------
try:
    summary_req = requests.get(f"{API_URL}/api/summary", params={"source": source}, timeout=5)
    events_req = requests.get(f"{API_URL}/api/events", params={"source": source}, timeout=5)
    if summary_req.status_code == 200 and events_req.status_code == 200:
        summary_res = summary_req.json()
        df_all = pd.DataFrame(events_req.json())
    else:
        st.error(f"API returned status code: {summary_req.status_code}")
        st.stop()
except Exception as e:
    st.error(f"Could not connect to FastAPI at {API_URL}. Error: {e}")
    st.info("Make sure 'uvicorn api:app --reload' is running in your first terminal tab.")
    st.stop()

if df_all.empty:
    if source == "video":
        st.warning("The video database is empty. Run the vision pipeline first, e.g. "
                   "`python vision_pipeline.py --video store.mp4 --fresh`, then refresh. "
                   "You can switch back to Mock data in the sidebar.")
    else:
        st.warning("The mock database is empty. Run `python generate_mock_db.py`.")
    st.stop()

st.caption(f"Data source: **{source_label}**  |  {len(df_all)} visitor records")

# ---------------- Apply the period filter ----------------
df = df_all.copy()
prev = df_all.iloc[0:0].copy()
if ANALYTICS_READY and period_days and not df_all.empty:
    cur, prev = AD.split_periods(df_all, period_days)
    if len(cur):
        df = cur.copy()
    prev = prev.copy()

# ---------------- Compute all analytics once, before drawing tabs ----------------
ready = MODEL_READY and not df.empty
insight_list, trend_text = [], ""
analytics_ok, analytics_problem = False, ""

if ready:
    df["purchase_probability"] = score(df)
    if len(prev):
        prev["purchase_probability"] = score(prev)
    hours_unique = pd.to_datetime(df["timestamp"]).dt.hour.nunique()
    # A "peak hour" from records all created at the same time is meaningless, so hide it
    insight_list = [(lvl, msg) for lvl, msg in generate_ml_insights(df)
                    if not ("Peak traffic" in msg and hours_unique <= 1)]

    if not ANALYTICS_READY:
        analytics_problem = f"analytics.py / advanced.py could not be loaded: {LOAD_ERR}"
    else:
        hist = analytics.load_history()
        if hist is None:
            analytics_problem = ("synthetic_retail_visitor_data.csv not found in the project "
                                 "folder (needed for history, forecasting, segments and Model Lab).")
        else:
            try:
                basket = analytics.avg_basket(hist)
                actions = analytics.revenue_actions(df, basket, promo_cost, engage_cost, assist_lift)

                daily, src_label, is_live = analytics.daily_counts(df_all, hist)
                forecast, fc_comp, mape = analytics.forecast_footfall(daily, align_to_today=is_live)
                if is_live:
                    ts_all = pd.to_datetime(df_all["timestamp"])
                    src_hours = pd.DataFrame({"date": ts_all.dt.normalize(), "hour": ts_all.dt.hour})
                else:
                    src_hours = hist
                heat, staff_plan = analytics.staffing_plan(src_hours, forecast, capacity)

                seg_df, seg_summary = analytics.segment_visitors(df, hist)
                alerts, daily_an = AD.detect_anomalies(df_all)
                analytics_ok = True
            except Exception as e:
                analytics_problem = f"Analytics error: {e}"


def pct_change(cur_v, prev_v):
    return f"{(cur_v - prev_v) / prev_v * 100:+.1f}%" if prev_v else None


# ---------------- Tabs ----------------
tab_over, tab_pred, tab_rev, tab_fc, tab_seg, tab_alert, tab_lab, tab_ai = st.tabs([
    "Overview", "Predictions & What-If", "Revenue & Budget", "Forecast & Staffing",
    "Customer Segments", "Alerts", "Model Lab", "AI Advisor",
])

# ============ TAB 1: Overview ============
with tab_over:
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Visitors (all time)", summary_res["total_visitors"])
    c2.metric("Average Dwell Time (all time)", f"{summary_res['avg_dwell_seconds']} sec")
    c3.metric("Monitored Zones", df_all["zone"].nunique() if not df_all.empty else 0)

    if ready and ANALYTICS_READY and period_days:
        st.markdown(f"#### Trend: {period_choice.lower()} vs the previous {period_days} days")
        if len(prev):
            cv, pv = AD.per_day(df), AD.per_day(prev)
            cd, pd_ = df["dwell_seconds"].mean(), prev["dwell_seconds"].mean()
            cp, pp = df["purchase_probability"].mean(), prev["purchase_probability"].mean()
            t1, t2, t3 = st.columns(3)
            t1.metric("Visitors per day", f"{cv:.0f}", pct_change(cv, pv))
            t2.metric("Average dwell (sec)", f"{cd:.1f}", pct_change(cd, pd_))
            t3.metric("Predicted purchase likelihood", f"{cp*100:.1f}%", f"{(cp-pp)*100:+.1f} points")
            trend_text = (f"Period: {period_choice} vs previous {period_days} days. "
                          f"Visitors/day {cv:.0f} vs {pv:.0f}; avg dwell {cd:.1f}s vs {pd_:.1f}s; "
                          f"predicted purchase likelihood {cp*100:.1f}% vs {pp*100:.1f}%.")
        else:
            st.caption("Not enough earlier data to compare with a previous period.")

    st.divider()
    st.subheader("Spatial Store Movement Heatmap")
    if source == "video" and os.path.exists("heatmap_output.png"):
        st.image("heatmap_output.png",
                 caption="Where people stood in the video (red = most time spent)",
                 use_container_width=True)
    elif source == "video":
        st.info("Heatmap not found yet. Run vision_pipeline.py to generate it.")
    else:
        st.info("The spatial heatmap comes from video footage. Switch the Data Source in the "
                "sidebar to 'Live video pipeline' to see it.")

    st.divider()
    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("Demographic Breakdown (Gender)")
        if not df.empty:
            st.plotly_chart(px.pie(df, names="gender", title="Gender Distribution",
                                   color_discrete_sequence=px.colors.qualitative.Set2),
                            use_container_width=True)
    with col_right:
        st.subheader("Customer Dwell Duration per Tracked ID")
        if not df.empty:
            st.plotly_chart(px.bar(df.tail(100), x="track_id", y="dwell_seconds", color="zone",
                                   title="Dwell Duration (Seconds) - latest 100 visitors"),
                            use_container_width=True)

    st.divider()
    st.subheader("Visitor Log Data")
    st.dataframe(df, use_container_width=True)
    if not df.empty:
        st.download_button("Export CSV Report", df.to_csv(index=False).encode("utf-8"),
                           "visitor_report.csv", "text/csv")

# ============ TAB 2: Predictions & What-If ============
with tab_pred:
    if not MODEL_READY:
        st.warning("Model not found. Run 'python train_model.py' first and make sure "
                   "insights.py is in the project folder.")
    elif df.empty:
        st.info("No visitor data yet.")
    else:
        st.caption(
            f"Model: Gradient Boosting classifier trained on a SYNTHETIC dataset "
            f"(accuracy {METRICS['accuracy']}, ROC-AUC {METRICS['roc_auc']}). "
            f"This demonstrates the pipeline; with real store data it would be retrained."
        )
        m1, m2 = st.columns(2)
        m1.metric("Avg. Predicted Purchase Likelihood",
                  f"{df['purchase_probability'].mean()*100:.1f}%")
        m2.metric("Likely Buyers (>50%)", int((df["purchase_probability"] > 0.5).sum()))

        st.markdown("#### Recommended Actions")
        for level, msg in insight_list:
            if level == "success":
                st.success(msg)
            elif level == "warning":
                st.warning(msg)
            else:
                st.info(msg)

        st.markdown("#### What-If Simulator")
        s1, s2 = st.columns(2)
        promo = s1.checkbox("Run a promotion")
        boost = s2.slider("Extra dwell time (seconds)", 0, 30, 0)
        new_avg = score(df, promotion=promo, dwell_boost=boost).mean()
        delta = new_avg - df["purchase_probability"].mean()
        st.metric("Predicted Purchase Likelihood (scenario)",
                  f"{new_avg*100:.1f}%", f"{delta*100:+.1f} points")

        k1, k2 = st.columns(2)
        with k1:
            zp = df.groupby("zone")["purchase_probability"].mean().reset_index()
            st.plotly_chart(px.bar(zp, x="zone", y="purchase_probability",
                                   title="Predicted Purchase Likelihood by Zone"),
                            use_container_width=True)
        with k2:
            imp = pd.DataFrame({"feature": list(IMPORTANCE.keys()),
                                "importance": list(IMPORTANCE.values())}).sort_values("importance")
            st.plotly_chart(px.bar(imp, x="importance", y="feature", orientation="h",
                                   title="What Drives Purchases (Model)"),
                            use_container_width=True)

# ============ TAB 3: Revenue & Budget ============
with tab_rev:
    if not ready:
        st.info("Needs the trained model and visitor data.")
    elif not analytics_ok:
        st.warning(analytics_problem)
    else:
        st.subheader("Ranked Actions")
        st.caption("Estimated gain for the visitors in the selected period. Gains overlap, so "
                   "do not add the rows together. 'Model' rows come from the trained model; "
                   "'Assumption' rows use the lift you set in the sidebar.")
        top = actions.iloc[0]
        r1, r2, r3 = st.columns(3)
        r1.metric("Average basket (synthetic history)", f"Rs {basket:.0f}")
        r2.metric("Best single action: net gain", f"Rs {top['Net gain (Rs)']:.0f}")
        r3.metric("Visitors analysed", len(df))
        st.success(f"Top priority: {top['Action']}")
        fig = px.bar(actions, x="Net gain (Rs)", y="Action", color="Basis", orientation="h",
                     title="Ranked Actions by Estimated Net Gain")
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(actions, hide_index=True, use_container_width=True)

        # ---------- Budget optimizer ----------
        st.divider()
        st.subheader("Budget Optimizer")
        st.caption("Chooses at most one action per zone (promotion, +10s engagement, or both) to "
                   "maximise total net gain without exceeding your budget. Solved exactly with "
                   "dynamic programming (multiple-choice knapsack).")
        opts = AD.zone_options(df, basket, promo_cost, engage_cost)
        max_budget = max(100, int(opts.groupby("zone")["cost"].max().sum()))
        budget = st.slider("Marketing budget (Rs)", 0, max_budget, int(max_budget * 0.4),
                           step=max(1, max_budget // 100))
        budget_plan, curve = AD.optimize_budget(opts, budget)
        b1, b2, b3 = st.columns(3)
        b1.metric("Money spent", f"Rs {budget_plan['Cost (Rs)'].sum():,}")
        b2.metric("Expected revenue gain", f"Rs {budget_plan['Revenue gain (Rs)'].sum():,}")
        b3.metric("Expected net gain", f"Rs {budget_plan['Net gain (Rs)'].sum():,}")
        st.dataframe(budget_plan, hide_index=True, use_container_width=True)
        step = max(1, len(curve) // 200)
        st.plotly_chart(px.line(x=np.arange(0, len(curve), step), y=curve[::step],
                                labels={"x": "Budget (Rs)", "y": "Best achievable net gain (Rs)"},
                                title="Diminishing Returns: Net Gain vs Budget"),
                        use_container_width=True)

        # ---------- A/B test planner ----------
        st.divider()
        st.subheader("A/B Test Planner")
        st.caption("Model estimates are predictions, not proof. This calculates how many visitors "
                   "a real experiment (for example promotion vs no promotion) needs before you "
                   "can trust the result.")
        promo_pts = float((score(df, promotion=True).mean() - df["purchase_probability"].mean()) * 100)
        a1, a2, a3, a4 = st.columns(4)
        base_pct = a1.number_input("Baseline purchase rate (%)", 1.0, 95.0,
                                   round(float(df["purchase_probability"].mean() * 100), 1), 0.5)
        uplift = a2.number_input("Uplift to detect (points)", 0.5, 40.0,
                                 float(max(1.0, round(promo_pts, 1))), 0.5)
        alpha = a3.selectbox("Significance level", [0.05, 0.01, 0.10], index=0)
        power = a4.selectbox("Statistical power", [0.8, 0.9], index=0)
        n_group = AD.ab_sample_size(base_pct / 100, uplift, alpha, power)
        daily_traffic = max(1.0, float(forecast["Forecast"].mean()))
        if n_group:
            days_needed = int(np.ceil(2 * n_group / daily_traffic))
            q1, q2, q3 = st.columns(3)
            q1.metric("Visitors needed per group", f"{n_group:,}")
            q2.metric("Total visitors needed", f"{2 * n_group:,}")
            q3.metric("Estimated test duration", f"{days_needed} days")
            ab_text = (f"To detect a {uplift:.1f}-point uplift from a {base_pct:.1f}% baseline "
                       f"(alpha {alpha}, power {power}) an A/B test needs {n_group} visitors per "
                       f"group, about {days_needed} days at ~{daily_traffic:.0f} visitors/day.")
            st.info(ab_text)
        else:
            ab_text = "A/B planner: uplift must be positive."
            st.warning(ab_text)

# ============ TAB 4: Forecast & Staffing ============
with tab_fc:
    if not analytics_ok:
        st.info("Needs the model, visitor data and the history CSV."
                if not analytics_problem else analytics_problem)
    else:
        st.caption(f"Data source: {src_label}. Two forecasting models are back-tested on the last "
                   f"14 days and the better one is used automatically.")
        f1, f2 = st.columns(2)
        f1.metric("Expected visitors, next 7 days", int(forecast["Forecast"].sum()))
        if mape is not None:
            f2.metric("Selected model back-test error", f"{mape:.1f}% MAPE")
        st.dataframe(fc_comp, hide_index=True, use_container_width=True)

        fc = forecast.copy()
        fc["up"] = fc["High"] - fc["Forecast"]
        fc["down"] = fc["Forecast"] - fc["Low"]
        st.plotly_chart(px.bar(fc, x="Day", y="Forecast", error_y="up", error_y_minus="down",
                               title="7-Day Visitor Forecast (bars show roughly an 80% range)"),
                        use_container_width=True)
        st.plotly_chart(px.imshow(heat, aspect="auto", color_continuous_scale="Blues",
                                  labels=dict(x="Hour of day", y="Day", color="Visitors"),
                                  title="Expected Visitors by Day and Hour"),
                        use_container_width=True)
        st.markdown("#### Staffing Plan")
        st.dataframe(staff_plan, hide_index=True, use_container_width=True)
        st.caption(f"Assumes one staff member serves {capacity} visitors per hour "
                   f"(change this in the sidebar).")

# ============ TAB 5: Customer Segments ============
with tab_seg:
    if not ready:
        st.info("Needs the trained model and visitor data.")
    elif not analytics_ok:
        st.warning(analytics_problem)
    else:
        st.caption("K-Means clustering (fitted on the 90-day synthetic history) on dwell time, "
                   "predicted purchase likelihood and zone. Visitors are assigned to the nearest "
                   "segment.")
        st.dataframe(seg_summary, hide_index=True, use_container_width=True)
        st.plotly_chart(px.scatter(seg_df.tail(1500), x="dwell_seconds", y="purchase_probability",
                                   color="segment", hover_data=["track_id", "zone"],
                                   title="Visitors by Segment (latest 1,500 shown)"),
                        use_container_width=True)

# ============ TAB 6: Alerts ============
with tab_alert:
    if not analytics_ok:
        st.info("Needs the model, visitor data and the history CSV."
                if not analytics_problem else analytics_problem)
    elif daily_an.empty or len(daily_an) < 14:
        st.info("Anomaly detection needs at least 14 days of visitor data.")
    else:
        st.caption("Flags days that are unusual for this store using robust statistics "
                   "(median and MAD): total traffic compared with the typical value for that "
                   "weekday, and average dwell time per zone. It checks all data, regardless of "
                   "the period selected in the sidebar.")
        if alerts.empty:
            st.success("No unusual days detected.")
        else:
            st.warning(f"{len(alerts)} unusual event(s) detected.")
            st.dataframe(alerts, hide_index=True, use_container_width=True)

        figa = go.Figure()
        figa.add_trace(go.Scatter(x=daily_an["date"], y=daily_an["visitors"], mode="lines",
                                  name="Visitors"))
        figa.add_trace(go.Scatter(x=daily_an["date"], y=daily_an["expected"], mode="lines",
                                  name="Typical for weekday", line=dict(dash="dot")))
        traffic = alerts[alerts["Where"] == "Whole store"] if not alerts.empty else alerts
        if len(traffic):
            tdates = pd.to_datetime(traffic["Date"])
            flagged = daily_an[daily_an["date"].isin(tdates)]
            figa.add_trace(go.Scatter(x=flagged["date"], y=flagged["visitors"], mode="markers",
                                      name="Flagged", marker=dict(size=12, color="red")))
        figa.update_layout(title="Daily Visitors and Flagged Days")
        st.plotly_chart(figa, use_container_width=True)
        st.caption("With the mock database, three incidents are injected on purpose (a busy day, "
                   "a quiet day and a long checkout queue) so this detector can be validated.")

# ============ TAB 7: Model Lab ============
with tab_lab:
    if not ANALYTICS_READY or not MODEL_READY:
        st.info("Needs insights.py, advanced.py and the trained model.")
    else:
        hist_lab = analytics.load_history()
        if hist_lab is None:
            st.warning("synthetic_retail_visitor_data.csv not found.")
        else:
            @st.cache_resource(show_spinner="Training and comparing models (first time only)...")
            def run_lab():
                return AD.model_lab(analytics.load_history())

            lab = run_lab()
            st.caption("Honest evaluation of the purchase model on the synthetic dataset: three "
                       "algorithms, 5-fold cross-validation and a held-out test set.")
            st.dataframe(lab["table"], hide_index=True, use_container_width=True)
            st.info("How to read this: ROC-AUC is 0.5 for random guessing and 1.0 for perfect. "
                    "Scores around 0.70 are expected because the synthetic data deliberately "
                    "contains random noise. When models differ by less than the CV spread, treat "
                    "them as tied. Simple models can match complex ones when the real pattern is "
                    "smooth. With real store data we would repeat this comparison and pick the "
                    "winner.")

            l1, l2 = st.columns(2)
            with l1:
                figr = go.Figure()
                for name, (fpr, tpr) in lab["roc"].items():
                    figr.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=name))
                figr.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random guess",
                                          line=dict(dash="dash", color="gray")))
                figr.update_layout(title="ROC Curves (hold-out set)",
                                   xaxis_title="False positive rate", yaxis_title="True positive rate")
                st.plotly_chart(figr, use_container_width=True)
            with l2:
                cm = lab["cm"]
                figc = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                                 x=["Predicted: no buy", "Predicted: buy"],
                                 y=["Actual: no buy", "Actual: buy"],
                                 title="Confusion Matrix (deployed model, threshold 0.5)")
                st.plotly_chart(figc, use_container_width=True)

            l3, l4 = st.columns(2)
            with l3:
                cal = lab["calibration"]
                figk = go.Figure()
                figk.add_trace(go.Scatter(x=cal["predicted"], y=cal["actual"], mode="lines+markers",
                                          name="Model"))
                figk.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect",
                                          line=dict(dash="dash", color="gray")))
                figk.update_layout(title="Calibration: when the model says X%, do X% buy?",
                                   xaxis_title="Predicted probability", yaxis_title="Actual purchase rate")
                st.plotly_chart(figk, use_container_width=True)
            with l4:
                st.plotly_chart(px.bar(lab["perm_importance"], x="importance", y="feature",
                                       orientation="h",
                                       title="Permutation Importance (drop in AUC when shuffled)"),
                                use_container_width=True)

# ============ TAB 8: AI Advisor ============
with tab_ai:
    if not analytics_ok:
        st.info("Needs the model, visitor data and the history CSV."
                if not analytics_problem else analytics_problem)
    elif not GEMINI_READY:
        st.info("Run 'pip install google-genai' and add gemini_advisor.py to enable this tab.")
    else:
        assumptions = {
            "promotion cost per visitor (Rs)": promo_cost,
            "engagement cost per visitor (Rs)": engage_cost,
            "assumed assistance lift (points)": assist_lift,
            "staff capacity (visitors/hour)": capacity,
            "analysis period": period_choice,
        }

        daily_tbl, cmp_tbl, cmp_text = None, None, ""
        if PC_READY:
            n_days = PC.total_days(df_all)
            with st.expander("Compare two periods (also sent to the advisor)"):
                first_days = st.number_input(
                    "Days in Period 1 (the rest becomes Period 2)",
                    1, max(n_days - 1, 1), max(n_days // 2, 1))
                cmp_tbl, cmp_text = PC.compare_periods(df_all, int(first_days))
                if cmp_tbl is not None:
                    st.dataframe(cmp_tbl, hide_index=True)
            daily_tbl = PC.daily_table(df_all)

        def build_extras():
            parts = []
            if daily_tbl is not None:
                parts += ["=== DAILY TIME SERIES (live store data, one row per day) ===",
                          daily_tbl.to_string(index=False), ""]
            if cmp_text:
                parts += ["=== TWO-PERIOD COMPARISON ===", cmp_text, ""]
            if trend_text:
                parts += ["=== TREND ===", trend_text, ""]
            parts += ["=== FORECAST MODEL SELECTION ===", f"Data source: {src_label}",
                      fc_comp.to_string(index=False), ""]
            parts += ["=== UNUSUAL DAYS DETECTED ===",
                      alerts.head(8).to_string(index=False) if len(alerts) else "None detected", ""]
            parts += [f"=== BUDGET-OPTIMISED PLAN (budget Rs {budget}) ===",
                      budget_plan.to_string(index=False), "", "=== A/B TEST PLANNER ===", ab_text]
            return "\n".join(parts)

        def get_facts():
            return ga.build_facts(df, insight_list, METRICS, actions, forecast, mape, staff_plan,
                                  seg_summary, analytics.history_text(hist), assumptions,
                                  build_extras())

        st.caption("The AI advisor only sees aggregated statistics computed by your system (no raw "
                   "rows, no video). Predictions come from a model trained on synthetic data.")

        st.markdown("#### AI Action Plan")
        if st.button("Generate AI Action Plan"):
            if not api_key:
                st.warning("Please enter your AI access key in the sidebar.")
            else:
                with st.spinner("Analysing your store data..."):
                    try:
                        st.session_state["plan"] = ga.generate_plan(api_key, get_facts())
                    except Exception as e:
                        st.error(f"AI request failed: {e}")
        for item in st.session_state.get("plan", []):
            with st.container(border=True):
                st.markdown(f"**#{item.get('priority', '?')}  {item.get('title', '')}**")
                st.write(item.get("action", ""))
                a, b = st.columns(2)
                a.markdown(f"**Expected impact:** {item.get('expected_impact', '-')}")
                b.markdown(f"**Effort:** {item.get('effort', '-')}")
                st.caption(f"Evidence: {item.get('evidence', '-')}")
                st.caption(f"Caveat: {item.get('caveat', '-')}")

        # ---- Downloadable report ----
        kpis = {
            "Period": period_choice,
            "Visitors analysed": len(df),
            "Average dwell (s)": round(float(df["dwell_seconds"].mean()), 1),
            "Predicted purchase likelihood": f"{df['purchase_probability'].mean()*100:.1f}%",
            "Best single action": f"{actions.iloc[0]['Action']} (net Rs {actions.iloc[0]['Net gain (Rs)']:.0f})",
            "Expected visitors, next 7 days": int(forecast["Forecast"].sum()),
        }
        report = AD.build_report_md(kpis, st.session_state.get("plan", []), actions, budget_plan,
                                    forecast, staff_plan, alerts, seg_summary, assumptions)
        st.download_button("Download report (Markdown)", report.encode("utf-8"),
                           "store_intelligence_report.md", "text/markdown")

        # ---- Chat with memory ----
        st.divider()
        st.markdown("#### Ask the Advisor")
        st.session_state.setdefault("chat", [])
        for m in st.session_state["chat"]:
            with st.chat_message("assistant" if m["role"] == "model" else "user"):
                st.markdown(m["text"])

        question = st.chat_input("e.g. How should I spend a Rs 5,000 budget this week?")
        if question:
            if not api_key:
                st.warning("Please enter your AI access key in the sidebar.")
            else:
                with st.chat_message("user"):
                    st.markdown(question)
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        try:
                            reply = ga.chat(api_key, get_facts(), st.session_state["chat"], question)
                            st.markdown(reply)
                            st.session_state["chat"] += [
                                {"role": "user", "text": question},
                                {"role": "model", "text": reply},
                            ]
                        except Exception as e:
                            st.error(f"AI request failed: {e}")
        if st.session_state["chat"] and st.button("Clear conversation"):
            st.session_state["chat"] = []
            st.rerun()

# ---------------- Live demo: reload the page every 5 s while the pipeline is running ----------------
if source == "video" and auto_refresh:
    time.sleep(5)
    st.rerun()