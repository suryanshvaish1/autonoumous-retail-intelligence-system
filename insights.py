# insights.py - ML scoring, what-if simulation, and recommendation engine
import joblib
import pandas as pd

bundle = joblib.load("model.joblib")
MODEL = bundle["model"]
FEATURES = bundle["features"]
METRICS = bundle["metrics"]
IMPORTANCE = bundle["importance"]


def prepare(df):
    """Turn raw DB events into the features the model expects."""
    d = df.copy()
    ts = pd.to_datetime(d["timestamp"])
    d["hour"] = ts.dt.hour
    d["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)
    d["promotion_active"] = 0  # default: no promotion running
    return d


def score(df, promotion=None, dwell_boost=0.0):
    """Predicted purchase probability for every visitor, optionally under a what-if."""
    d = prepare(df)
    if promotion is not None:
        d["promotion_active"] = int(promotion)
    d["dwell_seconds"] = d["dwell_seconds"] + dwell_boost
    return MODEL.predict_proba(d[FEATURES])[:, 1]


def generate_ml_insights(df):
    """Returns a list of (level, message). level = success / info / warning."""
    if df.empty:
        return []
    out = []
    d = prepare(df)
    d["p"] = score(df)
    base = d["p"].mean()

    # 1. Promotion what-if
    promo_gain = score(df, promotion=True).mean() - base
    if promo_gain > 0.03:
        out.append(("success",
            f"Running a promotion is predicted to raise purchase likelihood by "
            f"{promo_gain*100:.1f} percentage points. Consider promotions during busy periods."))
    else:
        out.append(("info", "Promotions show only a small predicted effect on purchases."))

    # 2. Best zone to increase dwell time
    zone_gain = {}
    for z in d["zone"].unique():
        sub = d[d["zone"] == z]
        zone_gain[z] = score(sub, dwell_boost=10).mean() - sub["p"].mean()
    best_zone = max(zone_gain, key=zone_gain.get)
    if zone_gain[best_zone] > 0.01:
        out.append(("success",
            f"Keeping customers 10 seconds longer in '{best_zone}' gives the biggest predicted "
            f"gain (+{zone_gain[best_zone]*100:.1f} points). Add signage, seating or demos there."))

    # 3. Weakest zone
    zone_p = d.groupby("zone")["p"].mean()
    worst = zone_p.idxmin()
    out.append(("warning",
        f"'{worst}' has the lowest predicted purchase likelihood ({zone_p[worst]*100:.0f}%). "
        f"Review product placement, pricing labels or staffing there."))

    # 4. Peak hour staffing
    hour_counts = d["hour"].value_counts()
    peak = int(hour_counts.idxmax())
    out.append(("info",
        f"Peak traffic is around {peak}:00. Make sure enough staff are on the floor then."))

    # 5. High dwell, low predicted purchase
    stuck = d[(d["dwell_seconds"] > 20) & (d["p"] < 0.5)]
    if len(stuck) > 0:
        out.append(("warning",
            f"{len(stuck)} visitors spent over 20 seconds browsing but are unlikely to buy. "
            f"They may need staff assistance, clearer pricing or a nudge such as a discount."))

    # 6. Best demographic segment
    seg = d.groupby("age_group")["p"].mean()
    out.append(("info",
        f"Age group {seg.idxmax()} shows the highest predicted purchase likelihood "
        f"({seg.max()*100:.0f}%). Consider targeting offers at this group."))

    return out