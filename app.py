import base64, os, tempfile
from collections import Counter, defaultdict
 
import cv2
import numpy as np
import yt_dlp
from deepface import DeepFace
from flask import Flask, jsonify, render_template, request
 
MODEL = "Facenet512"
DETECTORS = {"fast": "opencv", "accurate": "retinaface"}
MATCH_THRESHOLD = 0.30       # cosine distance; lower = stricter matching
MAX_SAMPLES = 150            # safety cap on analyzed video frames
MAX_VIDEO_SECONDS = 900      # YouTube videos longer than 15 min are refused
 
MIN_CONF = 40                # ignore faces where the model is under 40% sure
MIN_FACE = 40                # ignore tiny faces (pixels)
HS_MAP = {"ang": "angry", "con": "neutral", "dis": "disgust", "fea": "fear",
          "hap": "happy", "neu": "neutral", "sad": "sad", "sur": "surprise"}
try:   # more accurate emotion model; falls back to DeepFace's own if unavailable
    from hsemotion_onnx.facial_emotions import HSEmotionRecognizer
    HS = HSEmotionRecognizer(model_name="enet_b0_8_best_vgaf")
except Exception as err:
    HS = None
    HS_ERROR = str(err)
 
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024
 
 
def cosine_distance(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return 1 - float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
 
 
def shrink(img, width):
    h, w = img.shape[:2]
    return img if w <= width else cv2.resize(img, (width, int(h * width / w)))
 
 
def warm_up():
    """Load the AI models once at startup so the first analysis isn't slow."""
    blank = np.zeros((224, 224, 3), dtype=np.uint8)
    for det in set(DETECTORS.values()):
        try:
            DeepFace.analyze(blank, actions=["emotion"], detector_backend=det,
                             enforce_detection=False, silent=True)
        except Exception:
            pass
    try:
        DeepFace.represent(img_path=blank, model_name=MODEL,
                           detector_backend="skip", enforce_detection=False)
    except Exception:
        pass
 
 
def load_known(files, tmp, det):
    known, failed = {}, []
    for f in files:
        name = os.path.splitext(os.path.basename(f.filename))[0]
        path = os.path.join(tmp, "ref_" + f.filename.replace("/", "_"))
        f.save(path)
        for d in (det, "retinaface"):      # fall back to the accurate detector
            try:
                rep = DeepFace.represent(img_path=path, model_name=MODEL,
                                         detector_backend=d, enforce_detection=True)
                known[name] = rep[0]["embedding"]
                break
            except Exception:
                continue
        else:
            failed.append(name)
    return known, failed
 
 
def identify(crop, known):
    if not known:
        return None
    try:
        emb = DeepFace.represent(img_path=crop, model_name=MODEL,
                                 detector_backend="skip", enforce_detection=False)[0]["embedding"]
    except Exception:
        return "Unknown"
    name, dist = min(((n, cosine_distance(emb, r)) for n, r in known.items()), key=lambda t: t[1])
    return name if dist < MATCH_THRESHOLD else "Unknown"
 
 
def analyze_frame(frame, known, det):
    try:
        results = DeepFace.analyze(frame, actions=["emotion"], detector_backend=det,
                                   enforce_detection=False, silent=True)
    except Exception:
        return []
    faces = []
    for r in results:
        if r.get("face_confidence", 1) < 0.8:
            continue
        g = r["region"]
        x, y, w, h = max(g["x"], 0), max(g["y"], 0), g["w"], g["h"]
        crop = frame[y:y + h, x:x + w]
        if not crop.size or min(w, h) < MIN_FACE:
            continue
        emo, conf = r["dominant_emotion"], max(r["emotion"].values())
        if HS is not None:
            try:
                label, scores = HS.predict_emotions(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), logits=False)
                emo, conf = HS_MAP.get(label[:3].lower(), emo), float(max(scores)) * 100
            except Exception:
                pass
        if conf < MIN_CONF:                 # too unsure: skip instead of guessing
            continue
        faces.append((identify(crop, known), emo, (x, y, w, h)))
    return faces
 
 
def people_summary(per_person, known):
    out = {}
    for name, c in per_person.items():
        n = sum(c.values())
        out[name] = {"samples": n, "emotions": {e: round(100 * v / n) for e, v in c.most_common()}}
    return out, [n for n in known if n not in per_person]
 
 
def face_thumb(frame, box):
    """Small square JPEG of one face, as a data URL for the web page."""
    x, y, w, h = box
    pad = int(0.15 * max(w, h))
    crop = frame[max(y - pad, 0): y + h + pad, max(x - pad, 0): x + w + pad]
    if crop.size == 0:
        return None
    ok, buf = cv2.imencode(".jpg", cv2.resize(crop, (96, 96)))
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()
 
 
def run_photo(path, known, det):
    img = shrink(cv2.imread(path), 1000)
    faces = analyze_frame(img, known, det)
    counts, per_person, listing = Counter(), defaultdict(Counter), []
    for i, (name, emo, (x, y, w, h)) in enumerate(faces, 1):
        label = f"{name or 'Person ' + str(i)}: {emo}"
        counts[emo] += 1
        listing.append({"label": name or f"Person {i}", "emotion": emo})
        if name:
            per_person[name][emo] += 1
        cv2.rectangle(img, (x, y), (x + w, y + h), (80, 200, 120), 2)
        cv2.putText(img, label, (x, max(y - 8, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 200, 120), 2)
    ok, buf = cv2.imencode(".jpg", img)
    people, missing = people_summary(per_person, known)
    return {"kind": "photo", "total": len(faces), "counts": counts, "faces": listing,
            "people": people, "missing": missing,
            "image": "data:image/jpeg;base64," + base64.b64encode(buf).decode()}
 
 
def run_video(path, known, every, det):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    step = max(int(fps * every), 1)
    total, peak, per_person = Counter(), Counter(), defaultdict(Counter)
    gallery, per_emo = [], Counter()
    idx = sampled = 0
    while sampled < MAX_SAMPLES:
        if not cap.grab():                  # grab() skips decoding, much faster
            break
        if idx % step == 0:
            ok, frame = cap.retrieve()
            if ok:
                sampled += 1
                here = Counter()
                small = shrink(frame, 800)
                for name, emo, box in analyze_frame(small, known, det):
                    total[emo] += 1
                    here[emo] += 1
                    if name:
                        per_person[name][emo] += 1
                    if per_emo[emo] < 6 and len(gallery) < 30:   # sample faces to show
                        thumb = face_thumb(small, box)
                        if thumb:
                            per_emo[emo] += 1
                            gallery.append({"img": thumb, "emotion": emo, "name": name,
                                            "time": round(idx / fps)})
                for e, n in here.items():
                    peak[e] = max(peak[e], n)
        idx += 1
    cap.release()
    people, missing = people_summary(per_person, known)
    return {"kind": "video", "frames": sampled, "counts": total, "peak": peak,
            "people": people, "missing": missing, "gallery": gallery}
 
 
def download_youtube(url, tmp):
    # Video-only is enough (we only need the picture), and sorting by resolution
    # works for both normal and vertical (Shorts) videos.
    opts = {"format": "bv*/b", "format_sort": ["res:480", "vcodec:h264"],
            "outtmpl": os.path.join(tmp, "yt.%(ext)s"), "quiet": True,
            "no_color": True, "noplaylist": True}
    with yt_dlp.YoutubeDL(opts) as y:
        info = y.extract_info(url, download=False)
        if (info.get("duration") or 0) > MAX_VIDEO_SECONDS:
            raise ValueError("This video is longer than 15 minutes. Try a shorter one.")
        info = y.extract_info(url, download=True)
        return y.prepare_filename(info), info.get("title", "")
 
 
@app.route("/")
def index():
    return render_template("index.html")
 
 
@app.route("/analyze", methods=["POST"])
def analyze():
    mode = request.form.get("mode", "photo")
    det = DETECTORS.get(request.form.get("speed"), "retinaface")
    every = max(float(request.form.get("every", 2)), 0.2)
    with tempfile.TemporaryDirectory() as tmp:
        title = ""
        if mode == "youtube":
            url = request.form.get("url", "").strip()
            if not url:
                return jsonify(error="Paste a YouTube link first."), 400
            try:
                path, title = download_youtube(url, tmp)
            except Exception as e:
                return jsonify(error=f"Could not download this link: {e}"), 400
        else:
            media = request.files.get("media")
            if not media:
                return jsonify(error="Choose a photo or video first."), 400
            path = os.path.join(tmp, "media_" + os.path.basename(media.filename or "capture.jpg"))
            media.save(path)
        known, failed = load_known(request.files.getlist("known"), tmp, det)
        try:
            data = run_photo(path, known, det) if mode == "photo" else run_video(path, known, every, det)
        except Exception as e:
            return jsonify(error=f"Could not analyze this file: {e}"), 500
    data["title"] = title
    data["unreadable_refs"] = failed
    return jsonify(data)
 
 
if __name__ == "__main__":
    print("Loading AI models, one moment...")
    print("Emotion model: HSEmotion (accurate)" if HS else f"Emotion model: DeepFace basic ({HS_ERROR})")
    warm_up()
    app.run(debug=False, port=5001)