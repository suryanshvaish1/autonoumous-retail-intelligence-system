# api.py - FastAPI backend. Every endpoint takes ?source=mock or ?source=video
from fastapi import FastAPI, HTTPException
from sqlalchemy import func
from database import VisitorEvent
from data_sources import get_session, SOURCES

app = FastAPI(title="Retail Intelligence API")


def _check(source):
    if source not in SOURCES:
        raise HTTPException(status_code=400, detail=f"source must be one of {list(SOURCES)}")


@app.get("/api/summary")
def get_summary(source: str = "mock"):
    _check(source)
    session = get_session(source)
    total_visitors = session.query(VisitorEvent).count()
    gender_counts = session.query(VisitorEvent.gender, func.count(VisitorEvent.id)).group_by(VisitorEvent.gender).all()
    avg_dwell = session.query(func.avg(VisitorEvent.dwell_seconds)).scalar() or 0.0
    session.close()
    return {"total_visitors": total_visitors, "by_gender": dict(gender_counts),
            "avg_dwell_seconds": round(avg_dwell, 2)}


@app.get("/api/events")
def get_all_events(source: str = "mock"):
    _check(source)
    session = get_session(source)
    rows = session.query(VisitorEvent).all()
    session.close()
    return [{"id": r.id, "track_id": r.track_id, "gender": r.gender, "age_group": r.age_group,
             "zone": r.zone, "dwell_seconds": r.dwell_seconds,
             "timestamp": r.timestamp.strftime("%Y-%m-%d %H:%M:%S")} for r in rows]


@app.get("/api/sources")
def get_sources():
    """How many rows each source currently holds (shown in the dashboard sidebar)."""
    out = {}
    for s in SOURCES:
        session = get_session(s)
        out[s] = session.query(VisitorEvent).count()
        session.close()
    return out