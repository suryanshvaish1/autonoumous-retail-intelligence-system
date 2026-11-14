# advanced.py - trends, anomaly alerts, model lab, budget optimizer, A/B planner, report
import datetime
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, brier_score_loss, confusion_matrix,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from insights import score, FEATURES

CAT = ["zone", "gender", "age_group"]
NUM = ["dwell_seconds", "hour", "is_weekend", "promotion_active"]


# ------------------------------------------------------------------
# 1. Period comparison (current vs previous period)
# ------------------------------------------------------------------
def split_periods(df, days):
    """days=None -> everything. Otherwise returns (current_df, previous_df)."""
    if days is None or df.empty:
        return df, df.iloc[0:0]
    ts = pd.to_datetime(df["timestamp"])
    end = ts.max()
    cur = df[ts > end - pd.Timedelta(days=days)]
    prev = df[(ts <= end - pd.Timedelta(days=days)) & (ts > end - pd.Timedelta(days=2 * days))]
    return cur, prev


def per_day(d):
    """Average visitors per active day."""
    if d.empty:
        return 0.0
    return len(d) / pd.to_datetime(d["timestamp"]).dt.normalize().nunique()


# ------------------------------------------------------------------
# 2. Anomaly alerts (robust statistics)
# ------------------------------------------------------------------
def _mod_z(x):
    """Modified z-score based on median and MAD (robust to outliers)."""
    x = np.asarray(x, dtype=float)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return np.zeros(len(x)) if mad == 0 else 0.6745 * (x - med) / mad


ALERT_COLS = ["Date", "Type", "Where", "Observed", "Typical", "Score", "What to check"]


def detect_anomalies(df, threshold=3.5):
    """
    Flags unusual days: (a) daily traffic vs the typical value for that weekday,
    (b) average dwell time per zone vs that zone's typical day.
    Returns (alerts_df, daily_df). Needs roughly 2+ weeks of data.
    """
    empty = pd.DataFrame(columns=ALERT_COLS)
    if df.empty:
        return empty, pd.DataFrame()
    d = df.copy()
    d["date"] = pd.to_datetime(d["timestamp"]).dt.normalize()

    daily = d.groupby("date").size().rename("visitors").reset_index()
    if len(daily) < 14:
        return empty, daily
    daily["dow"] = daily["date"].dt.dayofweek
    daily["expected"] = daily.groupby("dow")["visitors"].transform("median")
    rel = (daily["visitors"] - daily["expected"]) / daily["expected"].clip(lower=1)
    daily["z"] = _mod_z(rel)

    rows = []
    for _, r in daily[daily["z"].abs() > threshold].iterrows():
        up = r["z"] > 0
        rows.append({
            "Date": r["date"].date(), "Type": "Traffic spike" if up else "Traffic drop",
            "Where": "Whole store", "Observed": int(r["visitors"]),
            "Typical": int(round(r["expected"])), "Score": round(abs(r["z"]), 1),
            "What to check": ("Was there an event or promotion? Make sure staffing and stock "
                              "kept up." if up else
                              "Weather, closure, local event or camera outage? Verify the data."),
        })

    zd = d.groupby(["date", "zone"]).agg(avg=("dwell_seconds", "mean"),
                                         n=("track_id", "count")).reset_index()
    for zone, g in zd[zd["n"] >= 5].groupby("zone"):
        if len(g) < 14:
            continue
        g = g.copy()
        g["z"] = _mod_z(g["avg"].values)
        typical = g["avg"].median()
        for _, r in g[g["z"].abs() > threshold].iterrows():
            up = r["z"] > 0
            rows.append({
                "Date": r["date"].date(), "Type": "Dwell anomaly (long)" if up else "Dwell anomaly (short)",
                "Where": zone, "Observed": round(r["avg"], 1), "Typical": round(typical, 1),
                "Score": round(abs(r["z"]), 1),
                "What to check": ("Queues, confusion or a stock problem keeping people here?"
                                  if up else "Visitors leaving quickly: blocked access, "
                                             "stock-out or something off-putting?"),
            })
    alerts = pd.DataFrame(rows, columns=ALERT_COLS)
    if not alerts.empty:
        alerts = alerts.sort_values("Date", ascending=False).reset_index(drop=True)
    return alerts, daily


# ------------------------------------------------------------------
# 3. Model Lab: compare models honestly
# ------------------------------------------------------------------
def model_lab(hist):
    X = hist[FEATURES].copy()
    X[["is_weekend", "promotion_active"]] = X[["is_weekend", "promotion_active"]].astype(int)
    y = hist["converted"].astype(int)

    def make(clf):
        pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
                                 ("num", StandardScaler(), NUM)])
        return Pipeline([("pre", pre), ("clf", clf)])

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=200, min_samples_leaf=5,
                                                random_state=42, n_jobs=-1),
        "Gradient Boosting (deployed)": GradientBoostingClassifier(random_state=42),
    }
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    rows, roc, deployed = [], {}, None
    for name, clf in models.items():
        pipe = make(clf)
        cvs = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")
        pipe.fit(Xtr, ytr)
        proba = pipe.predict_proba(Xte)[:, 1]
        fpr, tpr, _ = roc_curve(yte, proba)
        roc[name] = (fpr, tpr)
        rows.append({
            "Model": name,
            "CV ROC-AUC": round(cvs.mean(), 3),
            "CV spread (+/-)": round(cvs.std(), 3),
            "Hold-out ROC-AUC": round(roc_auc_score(yte, proba), 3),
            "Hold-out accuracy": round(accuracy_score(yte, proba >= 0.5), 3),
            "Brier score (lower is better)": round(brier_score_loss(yte, proba), 3),
        })
        if "deployed" in name:
            deployed = (pipe, proba)

    pipe, proba = deployed
    cm = confusion_matrix(yte, proba >= 0.5)
    bins = pd.cut(proba, np.linspace(0, 1, 11), include_lowest=True)
    cal = (pd.DataFrame({"pred": proba, "actual": yte.values, "bin": bins})
           .groupby("bin", observed=True)
           .agg(predicted=("pred", "mean"), actual=("actual", "mean"), n=("pred", "size"))
           .reset_index(drop=True))
    cal = cal[cal["n"] >= 10]

    pi = permutation_importance(pipe, Xte, yte, scoring="roc_auc", n_repeats=5, random_state=42)
    imp = (pd.DataFrame({"feature": FEATURES, "importance": pi.importances_mean})
           .sort_values("importance"))
    return {"table": pd.DataFrame(rows), "roc": roc, "cm": cm, "calibration": cal,
            "perm_importance": imp}


# ------------------------------------------------------------------
# 4. Budget optimizer (multiple-choice knapsack) + A/B test planner
# ------------------------------------------------------------------
def zone_options(df, basket, promo_cost, engage_cost):
    """For every zone: gain and cost of each mutually exclusive option."""
    base = df["purchase_probability"].values
    rows = []
    for z in sorted(df["zone"].unique()):
        mask = (df["zone"] == z).values
        sub = df[mask]
        n = len(sub)
        opts = [
            ("Promotion", (score(sub, promotion=True) - base[mask]).sum(), n * promo_cost),
            ("+10s dwell engagement", (score(sub, dwell_boost=10) - base[mask]).sum(), n * engage_cost),
            ("Promotion + engagement",
             (score(sub, promotion=True, dwell_boost=10) - base[mask]).sum(),
             n * (promo_cost + engage_cost)),
        ]
        for name, gain, cost in opts:
            rows.append({"zone": z, "option": name, "extra_buyers": float(gain),
                         "revenue": float(gain * basket), "cost": float(np.ceil(cost)),
                         "net": float(gain * basket - np.ceil(cost))})
    return pd.DataFrame(rows)


def optimize_budget(opts, budget):
    """
    Picks at most one option per zone to maximise total net gain within the budget.
    Returns (plan_df, best_net_by_budget_array).
    """
    budget = int(budget)
    zones = list(opts["zone"].unique())
    dp = np.zeros((len(zones) + 1, budget + 1))
    choice = np.full((len(zones), budget + 1), -1, dtype=int)
    groups = [opts[opts["zone"] == z].reset_index(drop=True) for z in zones]

    for i, g in enumerate(groups):
        best = dp[i].copy()
        ch = np.full(budget + 1, -1, dtype=int)
        for j, (c, v) in enumerate(zip(g["cost"].astype(int), g["net"])):
            if c > budget:
                continue
            cand = np.full(budget + 1, -np.inf)
            cand[c:] = dp[i][: budget + 1 - c] + v
            better = cand > best
            best = np.where(better, cand, best)
            ch = np.where(better, j, ch)
        dp[i + 1] = best
        choice[i] = ch

    rows, b = [], budget
    for i in range(len(zones) - 1, -1, -1):
        j = choice[i][b]
        g = groups[i]
        if j >= 0:
            r = g.iloc[j]
            rows.append({"Zone": zones[i], "Chosen action": r["option"], "Cost (Rs)": int(r["cost"]),
                         "Revenue gain (Rs)": round(r["revenue"]), "Net gain (Rs)": round(r["net"])})
            b -= int(r["cost"])
        else:
            rows.append({"Zone": zones[i], "Chosen action": "No action", "Cost (Rs)": 0,
                         "Revenue gain (Rs)": 0, "Net gain (Rs)": 0})
    plan = pd.DataFrame(rows[::-1])
    return plan, dp[-1]


def ab_sample_size(p0, uplift_points, alpha=0.05, power=0.8):
    """Visitors needed PER GROUP to detect an uplift (two-proportion z-test)."""
    p1 = min(0.999, p0 + uplift_points / 100.0)
    if p1 <= p0:
        return None
    za, zb = norm.ppf(1 - alpha / 2), norm.ppf(power)
    n = (za + zb) ** 2 * (p0 * (1 - p0) + p1 * (1 - p1)) / (p1 - p0) ** 2
    return int(np.ceil(n))


# ------------------------------------------------------------------
# 5. Downloadable report
# ------------------------------------------------------------------
def _md_table(df, max_rows=15):
    if df is None or len(df) == 0:
        return "_None_\n"
    d = df.head(max_rows).astype(str)
    head = "| " + " | ".join(d.columns) + " |"
    sep = "|" + "---|" * len(d.columns)
    body = ["| " + " | ".join(r) + " |" for r in d.values.tolist()]
    return "\n".join([head, sep] + body) + "\n"


def build_report_md(kpis, plan_items, actions, budget_plan, forecast, staff_plan,
                    alerts, seg_summary, assumptions):
    lines = [
        "# Store Intelligence Report",
        f"_Generated {datetime.datetime.now():%Y-%m-%d %H:%M}. Predictions come from a model "
        "trained on SYNTHETIC data and demonstrate the pipeline; they are not real market "
        "findings._", "",
        "## Key figures",
        *[f"- **{k}:** {v}" for k, v in kpis.items()], "",
    ]
    if plan_items:
        lines.append("## AI action plan")
        for it in plan_items:
            lines += [f"### {it.get('priority', '?')}. {it.get('title', '')}",
                      str(it.get("action", "")),
                      f"- Expected impact: {it.get('expected_impact', '-')}",
                      f"- Effort: {it.get('effort', '-')}",
                      f"- Evidence: {it.get('evidence', '-')}",
                      f"- Caveat: {it.get('caveat', '-')}", ""]
    lines += ["## Ranked actions by estimated net gain", _md_table(actions),
              "## Budget-optimised plan", _md_table(budget_plan),
              "## 7-day forecast", _md_table(forecast[["Day", "Forecast", "Low", "High"]]),
              "## Staffing plan", _md_table(staff_plan),
              "## Alerts", _md_table(alerts),
              "## Customer segments", _md_table(seg_summary),
              "## Assumptions",
              *[f"- {k}: {v}" for k, v in assumptions.items()]]
    return "\n".join(lines)