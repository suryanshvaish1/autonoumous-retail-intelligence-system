# vision_pipeline.py - Video -> YOLOv8 detection -> DeepSORT tracking -> zones + dwell time
#                      -> heatmap_output.png + rows in video_data.db
#
# Examples (run from the project folder, venv active):
#   python vision_pipeline.py --video store.mp4 --setup-zones --fresh   (draw zones, first run)
#   python vision_pipeline.py --video store.mp4 --fresh                 (reuse saved zones)
#   python vision_pipeline.py --webcam --fresh                          (laptop camera)
#   python vision_pipeline.py --video store.mp4 --no-show               (no preview window)
#
# Press  q  in the preview window to stop early (data is still saved).

import argparse
import datetime
import json
import os
import time

import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

from database import VisitorEvent
from data_sources import get_session

# ----------------------------- SETTINGS -----------------------------
# Zone names MUST match the ones the ML model was trained on.
ZONE_NAMES = ["Entrance Corridor", "Main Display Shelf Zone", "Checkout Counter"]

# Default zones = 3 vertical bands covering the whole frame, as fractions
# (x1, y1, x2, y2) of the frame width/height. Use --setup-zones to draw your own.
DEFAULT_ZONES = {
    "Entrance Corridor":       [0.00, 0.0, 0.33, 1.0],
    "Main Display Shelf Zone": [0.33, 0.0, 0.70, 1.0],
    "Checkout Counter":        [0.70, 0.0, 1.00, 1.0],
}
ZONES_FILE = "zones.json"
HEATMAP_FILE = "heatmap_output.png"

MIN_DWELL = 1.5       # seconds in a zone before a visitor counts (filters flickers)
EXIT_GRACE = 2.0      # seconds without being seen -> the visitor has left
AGES = ["18-24", "25-34", "35-44", "45-54"]

ZONE_COLORS = {       # BGR colours for drawing
    "Entrance Corridor": (255, 160, 0),
    "Main Display Shelf Zone": (0, 200, 0),
    "Checkout Counter": (0, 120, 255),
}


# ----------------------------- HELPERS -----------------------------
def placeholder_demographics(track_id):
    """PLACEHOLDER, NOT real recognition. Gender/age are NOT estimated from the image
    (no face analysis, which is a deliberate privacy choice). They are assigned from the
    track id so the database columns are filled. Say this honestly in the report."""
    return ["Male", "Female"][track_id % 2], AGES[(track_id * 7) % len(AGES)]


def load_zones():
    if os.path.exists(ZONES_FILE):
        with open(ZONES_FILE) as f:
            return json.load(f)
    return DEFAULT_ZONES


def setup_zones(frame):
    """Let the user draw one rectangle per zone with the mouse."""
    h, w = frame.shape[:2]
    zones = {}
    for name in ZONE_NAMES:
        print(f"[ZONES] Drag a rectangle for: {name}. Press ENTER to confirm "
              f"(or C to skip this zone).")
        x, y, bw, bh = cv2.selectROI(f"Draw zone: {name}  (ENTER = confirm)", frame, False)
        cv2.destroyAllWindows()
        if bw > 0 and bh > 0:
            zones[name] = [x / w, y / h, (x + bw) / w, (y + bh) / h]
    if not zones:
        print("[ZONES] Nothing drawn, using default zones.")
        return DEFAULT_ZONES
    with open(ZONES_FILE, "w") as f:
        json.dump(zones, f, indent=2)
    print(f"[ZONES] Saved to {ZONES_FILE}")
    return zones


def zone_of(zones, nx, ny):
    for name, (x1, y1, x2, y2) in zones.items():
        if x1 <= nx <= x2 and y1 <= ny <= y2:
            return name
    return None


def draw_zones(img, zones):
    h, w = img.shape[:2]
    for name, (x1, y1, x2, y2) in zones.items():
        c = ZONE_COLORS.get(name, (200, 200, 200))
        p1, p2 = (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h))
        cv2.rectangle(img, p1, p2, c, 2)
        cv2.putText(img, name, (p1[0] + 6, p1[1] + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)


def save_heatmap(acc, background, zones, path):
    """Blur the footprint counts, colour them (blue = little, red = a lot), overlay on the scene."""
    if acc.max() <= 0:
        return
    h, w = acc.shape
    heat = cv2.GaussianBlur(acc, (0, 0), sigmaX=max(8, 0.03 * w))
    heat = heat / heat.max()
    colour = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_JET)
    alpha = np.clip(heat * 1.2, 0, 0.75)[..., None]
    out = (background.astype(np.float32) * (1 - alpha) + colour.astype(np.float32) * alpha)
    out = out.astype(np.uint8)
    draw_zones(out, zones)
    cv2.imwrite(path, out)


def finalize(track_id, s, start_wall, session):
    """Turn one finished visit into one database row (zone where the visitor stayed longest)."""
    if not s["zone_time"]:
        return False
    zone = max(s["zone_time"], key=s["zone_time"].get)
    dwell = s["zone_time"][zone]
    if dwell < MIN_DWELL:
        return False
    gender, age = placeholder_demographics(track_id)
    session.add(VisitorEvent(
        track_id=track_id, gender=gender, age_group=age, zone=zone,
        dwell_seconds=round(dwell, 2),
        timestamp=start_wall + datetime.timedelta(seconds=s["first_t"]),
    ))
    session.commit()
    print(f"[DB LOG] Track #{track_id} | {gender} | {age} | {zone} | {dwell:.1f}s")
    return True


# ----------------------------- MAIN -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Retail vision pipeline")
    ap.add_argument("--video", help="path to a video file")
    ap.add_argument("--webcam", action="store_true", help="use the laptop camera")
    ap.add_argument("--fresh", action="store_true", help="empty video_data.db before starting")
    ap.add_argument("--setup-zones", action="store_true", help="draw the 3 zones with the mouse")
    ap.add_argument("--no-show", action="store_true", help="no preview window")
    ap.add_argument("--save-video", help="also save the annotated video, e.g. annotated.mp4")
    ap.add_argument("--skip", type=int, default=2, help="process every Nth frame (speed)")
    ap.add_argument("--conf", type=float, default=0.4, help="detection confidence (0-1)")
    ap.add_argument("--width", type=int, default=960, help="resize frames to this width")
    args = ap.parse_args()

    if not args.video and not args.webcam:
        ap.error("Give a video with --video FILE or use --webcam")
    if args.video and not os.path.exists(args.video):
        ap.error(f"Video file not found: {args.video}")

    cap = cv2.VideoCapture(0 if args.webcam else args.video)
    if not cap.isOpened():
        raise SystemExit("Could not open the video source.")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps or fps < 1:
        fps = 30.0

    ok, first = cap.read()
    if not ok:
        raise SystemExit("Could not read the first frame.")
    if first.shape[1] > args.width:
        scale = args.width / first.shape[1]
        first = cv2.resize(first, None, fx=scale, fy=scale)
    H, W = first.shape[:2]

    zones = setup_zones(first) if args.setup_zones else load_zones()
    print(f"[INFO] Zones: {list(zones)}")

    session = get_session("video")
    if args.fresh:
        n = session.query(VisitorEvent).delete()
        session.commit()
        print(f"[INFO] Emptied video_data.db ({n} old rows removed).")

    print("[INFO] Loading YOLOv8n (downloads ~6 MB the first time)...")
    model = YOLO("yolov8n.pt")
    tracker = DeepSort(max_age=15, n_init=3, embedder="mobilenet", embedder_gpu=False)

    writer = None
    if args.save_video:
        writer = cv2.VideoWriter(args.save_video, cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps / args.skip, (W, H))

    acc = np.zeros((H, W), dtype=np.float32)        # heatmap accumulator
    background = first.copy()
    state = {}                                      # track_id -> visit info
    logged = 0
    start_wall = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
    t0 = time.time()
    frame_idx = -1
    if not args.webcam:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)      # rewind: the first frame was only used for setup
    print("[INFO] Running. Press q in the video window to stop.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            if frame_idx % args.skip:
                continue
            if frame.shape[1] != W:
                frame = cv2.resize(frame, (W, H))
            vt = (time.time() - t0) if args.webcam else frame_idx / fps   # video time, seconds

            # 1) DETECT people (class 0 = person)
            res = model(frame, classes=[0], conf=args.conf, imgsz=640, verbose=False)[0]
            dets = []
            for (x1, y1, x2, y2), c in zip(res.boxes.xyxy.cpu().numpy(),
                                           res.boxes.conf.cpu().numpy()):
                dets.append(([float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                             float(c), "person"))

            # 2) TRACK (DeepSORT gives each person a stable id)
            tracks = tracker.update_tracks(dets, frame=frame)

            view = frame.copy()
            draw_zones(view, zones)
            seen_now = set()
            for t in tracks:
                if not t.is_confirmed() or t.time_since_update > 0:
                    continue
                tid = int(t.track_id)
                l, tp, r, b = t.to_ltrb()
                fx = min(max((l + r) / 2, 0), W - 1)       # feet = bottom centre of the box
                fy = min(max(b, 0), H - 1)
                zone = zone_of(zones, fx / W, fy / H)
                seen_now.add(tid)

                s = state.get(tid)
                if s is None:
                    s = state[tid] = {"first_t": vt, "last_t": vt, "zone_time": {}}
                dt = min(vt - s["last_t"], 1.0)
                s["last_t"] = vt

                # 3) ZONE + DWELL: add the elapsed time to the zone the person stands in
                if zone:
                    s["zone_time"][zone] = s["zone_time"].get(zone, 0.0) + dt

                # 4) HEATMAP: add this footprint
                acc[int(fy), int(fx)] += 1.0

                colour = ZONE_COLORS.get(zone, (180, 180, 180))
                cv2.rectangle(view, (int(l), int(tp)), (int(r), int(b)), colour, 2)
                dw = s["zone_time"].get(zone, 0.0) if zone else 0.0
                cv2.putText(view, f"ID {tid} | {dw:.1f}s", (int(l), max(15, int(tp) - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2)

            # 5) VISITORS WHO LEFT -> write to the database
            for tid in list(state):
                if tid not in seen_now and vt - state[tid]["last_t"] > EXIT_GRACE:
                    logged += finalize(tid, state.pop(tid), start_wall, session)

            if (frame_idx // args.skip) % 50 == 0:
                save_heatmap(acc, background, zones, HEATMAP_FILE)

            cv2.putText(view, f"Inside now: {len(seen_now)}   Logged visitors: {logged}",
                        (10, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            if writer:
                writer.write(view)
            if not args.no_show:
                cv2.imshow("Retail Vision Pipeline  (q = stop)", view)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("[INFO] Stopped by user.")
                    break
    except KeyboardInterrupt:
        print("[INFO] Stopped with Ctrl+C.")
    finally:
        for tid in list(state):
            logged += finalize(tid, state.pop(tid), start_wall, session)
        save_heatmap(acc, background, zones, HEATMAP_FILE)
        session.close()
        cap.release()
        if writer:
            writer.release()
        if not args.no_show:
            cv2.destroyAllWindows()

    print(f"[SUCCESS] Done. {logged} visitors saved to video_data.db, heatmap saved to {HEATMAP_FILE}.")


if __name__ == "__main__":
    main()