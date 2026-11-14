# analytics.py - revenue impact, footfall forecast, staffing plan, customer segments
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from insights import MODEL, FEATURES, score

HISTORY_CSV = "synthetic_retail_visitor_data.csv"
DEFAULT_BASKET = 35.0


# ------------------------------------------------------------------
# History helpers (the 90-day SYNTHETIC dataset)
# ------------------------------------------------------------------
def load_history():
    """Returns the historical dataframe, or None if the CSV is missing."""
    try:
        h = pd.read_csv(HISTORY_CSV)
        h["date"] = pd.to_datetime(h["date"])
        return h
    except Exception:
        return None


def avg_basket(hist):
    """Average sale amount (Rs) of visitors who bought."""
    if hist is None or "sale_amount" not in hist.columns:
        return DEFAULT_BASKET
    sales = hist.loc[hist["converted"] == 1, "sale_amount"]
    return float(sales.mean()) if len(sales) else DEFAULT_BASKET


def history_text(hist):
    """Plain-text summary of the history, used as context for Gemini."""
    if hist is None:
        return "No historical dataset available."
    days = hist["date"].nunique()
    daily = hist.groupby("date").size()
    wk = hist["date"].dt.dayofweek >= 5
    lines = [
        f"Days of history: {days}",
        f"Total visitors: {len(hist)}",
        f"Overall conversion rate: {hist['converted'].mean()*100:.1f}%",
        f"Average basket (buyers only): Rs {avg_basket(hist):.0f}",
        f"Average daily visitors: weekdays {daily[~daily.index.dayofweek.isin([5, 6])].mean():.0f}, "
        f"weekends {daily[daily.index.dayofweek.isin([5, 6])].mean():.0f}",
        f"Busiest hour overall: {int(hist['hour'].value_counts().idxmax())}:00",
        f"Conversion rate by zone: "
        f"{(hist.groupby('zone')['converted'].mean()*100).round(1).to_dict()}",
        f"Conversion rate with vs without promotion: "
        f"{(hist.groupby('promotion_active')['converted'].mean()*100).round(1).to_dict()} "
        f"(1 = promotion active; observational, not a controlled experiment)",
    ]
    return "\n".join(lines)


# ------------------------------------------------------------------
# 1. Revenue impact + ranked action plan
# ------------------------------------------------------------------
def revenue_actions(df, basket, promo_cost, engage_cost, assist_lift):
    """
    df must contain 'purchase_probability'. Returns actions ranked by net gain (Rs).
    Gains are for the visitors currently tracked and are NOT additive (actions overlap).
    """
    base = df["purchase_probability"].values
    rows = []

    def add(action, basis, affected, extra_buyers, cost_per_visitor):
        revenue = extra_buyers * basket
        cost = affected * cost_per_visitor
        net = revenue - cost
        rows.append({
            "Action": action,
            "Basis": basis,
            "Visitors affected": int(affected),
            "Extra buyers": round(float(extra_buyers), 2),
            "Revenue gain (Rs)": round(float(revenue), 0),
            "Cost (Rs)": round(float(cost), 0),
            "Net gain (Rs)": round(float(net), 0),
            "ROI (%)": round(float(net / cost * 100), 0) if cost > 0 else None,
        })

    # Store-wide promotion (model what-if)
    gain = score(df, promotion=True) - base
    add("Run a store-wide promotion", "Model", len(df), gain.sum(), promo_cost)

    # +10 seconds dwell, per zone (model what-if)
    for z in sorted(df["zone"].unique()):
        mask = (df["zone"] == z).values
        sub = df[mask]
        g = score(sub, dwell_boost=10) - base[mask]
        add(f"Keep visitors 10s longer in {z}", "Model", len(sub), g.sum(), engage_cost)

    # Staff assistance for browsers who are unlikely to buy (assumption)
    stuck = ((df["dwell_seconds"] > 20) & (df["purchase_probability"] < 0.5)).sum()
    if stuck > 0:
        add(f"Staff assistance for browsers (assumed +{assist_lift} pt lift)",
            "Assumption", stuck, stuck * assist_lift / 100, engage_cost)

    out = pd.DataFrame(rows).sort_values("Net gain (Rs)", ascending=False).reset_index(drop=True)
    out.insert(0, "Rank", out.index + 1)
    return out


# ------------------------------------------------------------------
# 2. Footfall forecast (model selection) + staffing plan
# ------------------------------------------------------------------
FORECAST_MODELS = ["Seasonal average (8 weeks)", "Ridge trend + weekday"]


def daily_counts(live_df, hist):
    """
    Daily visitor counts. Uses the live database when it covers 28+ days,
    otherwise falls back to the synthetic history CSV.
    Returns (daily_df, source_label, is_live).
    """
    if live_df is not None and len(live_df):
        dates = pd.to_datetime(live_df["timestamp"]).dt.normalize()
        if dates.nunique() >= 28:
            daily = dates.value_counts().sort_index()
            daily = daily.asfreq("D", fill_value=0).rename("visitors")
            return daily.rename_axis("date").reset_index(), "live database", True
    if hist is None:
        return None, "none", False
    daily = hist.groupby("date").size().asfreq("D", fill_value=0).rename("visitors")
    return daily.rename_axis("date").reset_index(), "synthetic history CSV", False


def _design(dates, t_values):
    dow = pd.DatetimeIndex(dates).dayofweek.values
    X = np.zeros((len(dow), 8))
    X[np.arange(len(dow)), dow] = 1.0
    X[:, 7] = np.asarray(t_values, dtype=float) / 100.0
    return X


def _predict(model_name, train, dates, t_values):
    if model_name == FORECAST_MODELS[0]:
        dm = train.tail(56).groupby(train["date"].tail(56).dt.dayofweek)["visitors"].mean()
        return np.array([dm.get(d.dayofweek, dm.mean()) for d in pd.DatetimeIndex(dates)])
    reg = Ridge(alpha=1.0).fit(_design(train["date"], train["t"]), train["visitors"])
    return reg.predict(_design(dates, t_values))


def forecast_footfall(daily, days=7, align_to_today=False):
    """
    Compares two forecasting models by back-testing on the last 14 days, picks the better
    one, and forecasts the next `days` days starting tomorrow.
    Returns (forecast_df, comparison_df, best_mape).
    """
    daily = daily.copy().sort_values("date").reset_index(drop=True)
    daily["t"] = (daily["date"] - daily["date"].iloc[0]).dt.days

    mape, sd = {}, {}
    if len(daily) > 35:
        tr, te = daily.iloc[:-14], daily.iloc[-14:]
        for m in FORECAST_MODELS:
            p = _predict(m, tr, te["date"], te["t"].values)
            actual = te["visitors"].values.astype(float)
            mape[m] = float(np.mean(np.abs(actual - p) / np.maximum(actual, 1)) * 100)
            sd[m] = float(np.std(actual - p))
        best = min(mape, key=mape.get)
    else:
        best = FORECAST_MODELS[0]
        sd[best] = float(daily["visitors"].std() or 0.0)

    start = pd.Timestamp.today().normalize() + pd.Timedelta(days=1)
    dates = [start + pd.Timedelta(days=i) for i in range(days)]
    first, last_t = daily["date"].iloc[0], int(daily["t"].iloc[-1])
    t_future = [(d - first).days if align_to_today else last_t + i + 1
                for i, d in enumerate(dates)]
    pred = _predict(best, daily, dates, t_future)

    margin = 1.28 * sd[best]          # roughly an 80% range
    fc = pd.DataFrame({
        "date": dates,
        "Day": [d.strftime("%a %d %b") for d in dates],
        "Forecast": np.round(pred).astype(int),
        "Low": np.round(np.maximum(0, pred - margin)).astype(int),
        "High": np.round(pred + margin).astype(int),
    })
    comp = pd.DataFrame({
        "Model": list(mape.keys()) or [best],
        "Back-test MAPE (%)": [round(v, 1) for v in mape.values()] or [None],
    })
    comp["Selected"] = comp["Model"].eq(best).map({True: "Yes", False: ""})
    return fc, comp, mape.get(best)


def staffing_plan(hist, forecast, capacity_per_hour=12):  # hist needs 'date' and 'hour' columns
    """
    Spreads each forecast day over opening hours (using the historical hourly pattern
    for weekdays/weekends) and converts expected visitors into staff needed.
    Returns (heatmap_df: day x hour of expected visitors, plan_df: one row per day).
    """
    h = hist.copy()
    h["weekend"] = h["date"].dt.dayofweek >= 5
    hours = sorted(h["hour"].unique())
    share = {}
    for flag, g in h.groupby("weekend"):
        share[bool(flag)] = g["hour"].value_counts(normalize=True).reindex(hours, fill_value=0)
    fallback = next(iter(share.values()))

    grid, plan = [], []
    for _, r in forecast.iterrows():
        s = share.get(bool(r["date"].dayofweek >= 5), fallback)
        expected = s * r["Forecast"]
        staff = np.ceil(expected / capacity_per_hour).clip(lower=1)
        grid.append(expected.rename(r["Day"]))
        plan.append({
            "Day": r["Day"],
            "Expected visitors": int(r["Forecast"]),
            "Peak hour": f"{int(expected.idxmax())}:00",
            "Visitors at peak": round(float(expected.max()), 1),
            "Staff at peak": int(staff.max()),
            "Total staff-hours": int(staff.sum()),
        })
    return pd.DataFrame(grid), pd.DataFrame(plan)


# ------------------------------------------------------------------
# 3. Customer segments (K-Means clustering)
# ------------------------------------------------------------------
_CACHE = {}

ACTIONS = {
    "buyers": "Make buying effortless: keep stock available, speed up checkout, offer bundles or add-ons.",
    "medium": "Nudge them: clear signage, a product demo or a light promotion.",
    "browsers": "They browse but do not buy: offer staff assistance, clear pricing and a targeted discount.",
    "passers": "Just passing through: use entrance displays to attract them; do not spend promotion budget here.",
}


def _features(d, seg):
    num = seg["scaler"].transform(d[["dwell_seconds", "purchase_probability"]])
    zone = np.column_stack([(d["zone"] == z).astype(float).values for z in seg["zones"]])
    return np.hstack([num, zone])


def _get_segmenter(hist, k=5):
    if "seg" in _CACHE:
        return _CACHE["seg"]
    h = hist.copy()
    h[["is_weekend", "promotion_active"]] = h[["is_weekend", "promotion_active"]].astype(int)
    h["purchase_probability"] = MODEL.predict_proba(h[FEATURES])[:, 1]

    seg = {"zones": sorted(h["zone"].unique())}
    seg["scaler"] = StandardScaler().fit(h[["dwell_seconds", "purchase_probability"]])
    X = _features(h, seg)
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    h["cluster"] = km.labels_

    prof = h.groupby("cluster").agg(
        dwell=("dwell_seconds", "mean"),
        prob=("purchase_probability", "mean"),
        zone=("zone", lambda s: s.mode().iloc[0]),
    )
    med_dwell = h["dwell_seconds"].median()
    order = prof["prob"].rank(ascending=False, method="first")  # 1 = highest intent

    names, kinds, used = {}, {}, {}
    for c, r in prof.iterrows():
        if order[c] <= 1:
            kind, tier = "buyers", "High-intent"
        elif order[c] > k - 2:
            kind, tier = ("browsers" if r["dwell"] >= med_dwell else "passers"), "Low-intent"
        else:
            kind, tier = "medium", "Medium-intent"
        stay = "long stay" if r["dwell"] >= med_dwell else "short stay"
        label = f"{tier} | {r['zone']} | {stay}"
        used[label] = used.get(label, 0) + 1
        names[c] = label if used[label] == 1 else f"{label} (#{used[label]})"
        kinds[c] = kind

    seg.update(km=km, names=names, kinds=kinds)
    _CACHE["seg"] = seg
    return seg


def segment_visitors(df, hist):
    """df needs 'purchase_probability'. Returns (visitors_with_segment, summary_table)."""
    seg = _get_segmenter(hist)
    labels = seg["km"].predict(_features(df, seg))
    out = df.copy()
    out["segment"] = [seg["names"][l] for l in labels]
    action_by_segment = {seg["names"][l]: ACTIONS[seg["kinds"][l]] for l in seg["names"]}

    summary = out.groupby("segment").agg(
        Visitors=("track_id", "count"),
        Avg_dwell_sec=("dwell_seconds", "mean"),
        Avg_purchase_prob=("purchase_probability", "mean"),
    ).reset_index()
    summary["Share (%)"] = (summary["Visitors"] / len(out) * 100).round(1)
    summary["Avg_dwell_sec"] = summary["Avg_dwell_sec"].round(1)
    summary["Avg_purchase_prob"] = (summary["Avg_purchase_prob"] * 100).round(1)
    summary = summary.rename(columns={"segment": "Segment",
                                      "Avg_dwell_sec": "Avg dwell (s)",
                                      "Avg_purchase_prob": "Avg purchase likelihood (%)"})
    summary["Recommended action"] = summary["Segment"].map(action_by_segment)
    return out, summary.sort_values("Visitors", ascending=False).reset_index(drop=True)