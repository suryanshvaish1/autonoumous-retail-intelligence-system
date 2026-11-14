# generate_mock_db.py - fills retail_data.db with DAYS days of realistic mock visitors
import datetime
import numpy as np
from database import Session, VisitorEvent, Base, engine

DAYS = 75          # how many days of history to create (60 or more)
SEED = 7           # change for a different random dataset

rng = np.random.default_rng(SEED)

ZONES = ["Main Display Shelf Zone", "Checkout Counter", "Entrance Corridor"]
ZONE_P = [0.45, 0.25, 0.30]
ZONE_DWELL = {"Main Display Shelf Zone": 24, "Checkout Counter": 14, "Entrance Corridor": 8}
AGES = ["18-24", "25-34", "35-44", "45-54"]
AGE_P = [0.25, 0.32, 0.25, 0.18]
HOURS = list(range(9, 21))
HOUR_W = np.array([2, 3, 4, 5, 6, 5, 4, 4, 5, 7, 8, 5], dtype=float)
HOUR_P = HOUR_W / HOUR_W.sum()

# Known "incidents" injected on purpose so the dashboard's Alerts tab has something to find.
# Key = how many days ago. This also lets you VALIDATE the detector (it should find all 3).
INCIDENTS = {
    21: ("traffic", 1.8, None),                  # busy day (event / festival)
    10: ("traffic", 0.45, None),                 # quiet day (rain / closure)
    5:  ("dwell", 1.8, "Checkout Counter"),      # long checkout queue
}

Base.metadata.create_all(engine)
session = Session()

# Remove old mock records so the data does not get mixed
deleted = session.query(VisitorEvent).delete()
session.commit()
print(f"[INFO] Removed {deleted} old records.")

today = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None, hour=0, minute=0, second=0, microsecond=0)
events, track_id = [], 1

for d in range(DAYS, 0, -1):                     # oldest day first, ends yesterday
    day = today - datetime.timedelta(days=d)
    weekend = day.weekday() >= 5
    n = int(rng.normal(55, 7) * (1.35 if weekend else 1.0))
    inc = INCIDENTS.get(d)
    if inc and inc[0] == "traffic":
        n = int(n * inc[1])
    hours = rng.choice(HOURS, size=n, p=HOUR_P)
    for h in hours:
        zone = str(rng.choice(ZONES, p=ZONE_P))
        dwell = max(2.0, float(rng.gamma(3, ZONE_DWELL[zone] / 3)))
        if inc and inc[0] == "dwell" and zone == inc[2]:
            dwell *= inc[1]
        ts = day + datetime.timedelta(hours=int(h),
                                      minutes=int(rng.integers(0, 60)),
                                      seconds=int(rng.integers(0, 60)))
        events.append(VisitorEvent(
            track_id=track_id,
            gender=str(rng.choice(["Male", "Female"])),
            age_group=str(rng.choice(AGES, p=AGE_P)),
            zone=zone,
            dwell_seconds=round(dwell, 2),
            timestamp=ts,
        ))
        track_id += 1

session.add_all(events)
session.commit()
session.close()
for ago, (kind, mult, where) in INCIDENTS.items():
    print(f"[INCIDENT] {(today - datetime.timedelta(days=ago)).date()}: {kind} x{mult} {where or ''}")
print(f"[SUCCESS] Created {len(events)} mock visitors across {DAYS} days "
      f"({(today - datetime.timedelta(days=DAYS)).date()} to {(today - datetime.timedelta(days=1)).date()}).")