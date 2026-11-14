# period_compare.py - gives the AI Advisor day-by-day data and two-period comparisons
# (aggregates only: no raw visitor rows are ever sent to Gemini)
import numpy as np
import pandas as pd
from insights import score


def _prep(df_all):
    d = df_all.copy()
    d["timestamp"] = pd.to_datetime(d["timestamp"])
    d["date"] = d["timestamp"].dt.normalize()
    d["hour"] = d["timestamp"].dt.hour
    d["p"] = score(d)
    return d


def total_days(df_all):
    """Number of calendar days covered by the live data."""
    if df_all.empty:
        return 0
    ts = pd.to_datetime(df_all["timestamp"])
    return int((ts.max().normalize() - ts.min().normalize()).days + 1)


def daily_table(df_all):
    """One row per day: weekday, visitors, avg dwell, avg purchase likelihood, visitors per zone."""
    d = _prep(df_all)
    g = d.groupby("date").agg(
        visitors=("track_id", "count"),
        avg_dwell_s=("dwell_seconds", "mean"),
        avg_buy_prob_pct=("p", "mean"),
    )
    g["avg_dwell_s"] = g["avg_dwell_s"].round(1)
    g["avg_buy_prob_pct"] = (g["avg_buy_prob_pct"] * 100).round(1)
    zones = d.pivot_table(index="date", columns="zone", values="track_id",
                          aggfunc="count", fill_value=0)
    zones.columns = [f"visitors_{c}" for c in zones.columns]
    g = g.join(zones)
    g.insert(0, "weekday", g.index.day_name().str[:3])
    g.index = g.index.strftime("%Y-%m-%d")
    g.index.name = "date"
    return g.reset_index()


def _stats(x):
    days = x["date"].nunique()
    out = {
        "Days": days,
        "Weekend days": int(x.drop_duplicates("date")["date"].dt.dayofweek.ge(5).sum()),
        "Visitors": len(x),
        "Visitors per day": round(len(x) / days, 1) if days else 0,
        "Avg dwell (s)": round(x["dwell_seconds"].mean(), 1),
        "Avg buy likelihood (%)": round(x["p"].mean() * 100, 1),
        "Busiest hour": f"{int(x['hour'].value_counts().idxmax())}:00",
    }
    for z, share in (x["zone"].value_counts(normalize=True) * 100).round(1).items():
        out[f"Share of visitors: {z} (%)"] = share
    for z, v in x.groupby("zone")["dwell_seconds"].mean().round(1).items():
        out[f"Avg dwell: {z} (s)"] = v
    for g_, share in (x["gender"].value_counts(normalize=True) * 100).round(1).items():
        out[f"Gender: {g_} (%)"] = share
    for a_, share in (x["age_group"].value_counts(normalize=True) * 100).round(1).items():
        out[f"Age group: {a_} (%)"] = share
    return out


def compare_periods(df_all, first_days):
    """Split the data into the first `first_days` days and the rest. Returns (table, text)."""
    d = _prep(df_all)
    start = d["date"].min()
    cut = start + pd.Timedelta(days=int(first_days))
    a, b = d[d["date"] < cut], d[d["date"] >= cut]
    if a.empty or b.empty:
        return None, "Not enough data to split into two periods."
    la = f"Period 1 ({a['date'].min():%Y-%m-%d} to {a['date'].max():%Y-%m-%d})"
    lb = f"Period 2 ({b['date'].min():%Y-%m-%d} to {b['date'].max():%Y-%m-%d})"
    sa, sb = _stats(a), _stats(b)
    rows = []
    for k in sa:
        va, vb = sa[k], sb.get(k, np.nan)
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)) and va:
            change = f"{(vb - va) / va * 100:+.1f}%"
        else:
            change = ""
        rows.append({"Metric": k, la: va, lb: vb, "Change": change})
    table = pd.DataFrame(rows)
    return table, table.to_string(index=False)