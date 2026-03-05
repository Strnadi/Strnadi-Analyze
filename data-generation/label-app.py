import os
import random
import sqlite3
import io
import time
import datetime
import email.utils
import numpy as np
import wave
import librosa
import librosa.display
import matplotlib
from flask import Flask, render_template_string, send_file, request, redirect, url_for, session, make_response, abort, g
from threading import Lock
from ai_edge_litert.interpreter import Interpreter
import logging

matplotlib.use("Agg")
import matplotlib.pyplot as plt

app = Flask(__name__, static_folder='')
app.logger.setLevel(logging.INFO)


DATABASE_PATH = os.environ.get("DATABASE_PATH", "database.db")
MODEL_LABELS = ["BC", "BE", "BhBl", "BlBh", "None", "Unfinished", "XB"]


def get_db():
    db = g.get("_database")
    if db is None:
        db = g._database = sqlite3.connect(DATABASE_PATH)
    return db


@app.teardown_appcontext
def close_db(exception):
    db = g.pop("_database", None)
    if db is not None:
        db.close()

# Configure a dedicated access logger (file or stdout)
access_logger = logging.getLogger("access")
access_logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
# formatter may reference attributes provided via logging.extra
fmt = '%(remote_addr)s - %(user)s [%(asctime)s] "%(method)s %(path)s %(protocol)s" %(status)d took %(duration_ms)dms (sent %(content_length)s bytes)'
handler.setFormatter(logging.Formatter(fmt))
access_logger.addHandler(handler)

# Avoid duplicate lines from Werkzeug's own logger
logging.getLogger("werkzeug").setLevel(logging.ERROR)

@app.before_request
def start_timer():
    g.start_time = time.perf_counter()

@app.after_request
def log_request(response):
    # compute duration
    start = getattr(g, "start_time", None)
    duration_ms = int((time.perf_counter() - start) * 1000) if start else -1

    # content length (dash if unknown)
    try:
        content_length = response.calculate_content_length()
    except Exception:
        content_length = None
    content_length = content_length if content_length is not None else "-"

    # supply fields to formatter via extra
    access_logger.info(
        "",  # message can be empty; formatter will use extra fields
        extra={
            "remote_addr": request.remote_addr or "-",
            "user": getattr(request, "remote_user", "-") or "-",
            "method": request.method,
            "path": request.full_path if request.query_string else request.path,
            "protocol": request.environ.get("SERVER_PROTOCOL", "-"),
            "status": response.status_code,
            "content_length": content_length,
            "duration_ms": duration_ms,
        },
    )

    # optional: expose timing header for clients
    response.headers["X-Response-Time"] = f"{duration_ms}ms"
    return response

def init_db():
    db = get_db()
    cur = db.cursor()

    cur.execute(
        """CREATE TABLE IF NOT EXISTS sessions (
        session_id INTEGER PRIMARY KEY AUTOINCREMENT
    )"""
    )

    cur.execute(
        """CREATE TABLE IF NOT EXISTS labels (
        filepath TEXT PRIMARY KEY,
        timestamp INTEGER,
        original_label TEXT,
        user_label TEXT,
        model_labels TEXT,
        locked_until INTEGER,
        locked_for INTEGER
    )"""
    )

    cur.execute(
        """CREATE TABLE IF NOT EXISTS undo_stack (
        undo_id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id INTEGER REFERENCES sessions(session_id),
        filepath TEXT REFERENCES labels(filepath),
        timestamp INTEGER,
        UNIQUE(session_id, filepath)
    )"""
    )

    db.commit()


with app.app_context():
    init_db()

app.secret_key = os.environ.get("FLASK_SECRET", "dev_secret_change_me")


def db_cursor():
    return get_db().cursor()


def get_session_id():
    db = get_db()
    cur = db.cursor()

    sid = session.get("sid")
    if not sid:
        cur.execute("INSERT INTO sessions DEFAULT VALUES RETURNING session_id")
        sid = cur.fetchone()[0]
        session["sid"] = sid
        db.commit()

    return sid


def get_all_labels():
    return MODEL_LABELS


def choose_class(label, curr_label):
    if label == curr_label:
        return "btn-current"

    if label == "None":
        return "btn-none"

    if label == "Unfinished":
        return "btn-unfinished"

    return "btn-label"

def load_and_normalize_audio(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=48000, mono=False)
        if audio.ndim > 1:
            if audio.shape[0] > 1 and np.any(audio[1]):
                audio = np.mean(audio, axis=0)
            else:
                audio = audio[0]

        audio = librosa.util.normalize(audio)
        audio = np.clip(audio, -1.0, 1.0)
        return audio

    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def tocsv(predictions: dict) -> str:
    return ",".join(f"{k}={v:.2f}%" for k, v in predictions.items())

def get_top_prediction(predictions: dict) -> tuple:
    """Returns (label, probability) of the top prediction"""
    if not predictions:
        return (None, 0.0)
    top_label = max(predictions, key=predictions.get)
    return (top_label, predictions[top_label])

def format_prob(prob: float) -> str:
    """Format probability as percentage string"""
    return f"{prob * 100:.2f}%"


interpreter = Interpreter(model_path="good-model-1.tflite", num_threads=8)
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
interpreter.allocate_tensors()

# ---------- routes ----------
@app.route("/")
def index():
    requested_file = request.args.get("file")

    cur = db_cursor()

    session_id = get_session_id()
    current_time = int(time.time())
    lock_duration = 300  # seconds

    if requested_file:
        cur.execute(
            "SELECT filepath, original_label FROM labels WHERE filepath = ? AND (locked_until IS NULL OR locked_until < ? OR (locked_until >= ? AND locked_for = ?)) LIMIT 1",
            (requested_file, current_time, current_time, session_id)
        )

        row = cur.fetchone()

    if not requested_file or (requested_file and not row):
        cur.execute(
            "SELECT filepath, original_label FROM labels WHERE user_label IS NULL AND (locked_until IS NULL OR locked_until < ? OR (locked_until >= ? AND locked_for = ?)) ORDER BY RANDOM() LIMIT 1",
            (current_time, current_time, session_id)
        )

        row = cur.fetchone()

    if not row:
        return "No files left to label! 🎉"

    filepath, original_label = row

    audio = load_and_normalize_audio(filepath)
    # print(audio, audio.shape)

    interpreter.set_tensor(input_details[0]['index'], np.array([audio]))
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    predictions = dict(zip(MODEL_LABELS, map(float, output_data[0])))

    # print(predictions)

    cur.execute("SELECT COUNT(*) FROM undo_stack WHERE session_id = ?", (session_id,))
    undo_count = cur.fetchone()[0]

    cur.execute("""
        UPDATE labels
        SET locked_until = ?,
            locked_for = ?
        WHERE filepath = ?
        """, (current_time + lock_duration, session_id, filepath))

    get_db().commit()

    top_label, top_prob = get_top_prediction(predictions)
    
    return render_template_string(
        TEMPLATE,
        filepath=filepath,
        current_label=original_label,
        choose_class=choose_class,
        labels=get_all_labels(),
        undo_available=(undo_count > 0),
        undo_count=undo_count,
        predictions=predictions,
        tocsv=tocsv,
        format_prob=format_prob,
        top_prediction=top_label,
        top_prob=top_prob
    )


@app.route("/spectrogram/<path:filepath>")
def spectrogram(filepath):
    if not os.path.exists(filepath):
        return "File not found", 404

    y, sr = librosa.load(filepath, sr=48000, mono=False)
    fig, ax = plt.subplots(figsize=(6, 3))
    S = librosa.feature.melspectrogram(y=y, sr=sr, fmin=2500, fmax=12000)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max), sr=sr, ax=ax, cmap="magma")
    ax.axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return send_file(buf, mimetype='image/png')


@app.route("/label", methods=["POST"])
def label_file():
    filepath = request.form["filepath"]
    new_label = request.form["label"]
    sid = get_session_id()
    cur = db_cursor()

    # # Check if the file is already labeled
    # cur.execute("SELECT user_label FROM labels WHERE filepath = ?", (filepath,))
    # row = cur.fetchone()
    # if row and row[0] is not None:
    #     # File is already labeled, so just return to index page (silent fail)
    #     return redirect(url_for("index"))

    cur.execute("SELECT locked_until, locked_for FROM labels WHERE filepath = ?", (filepath,))
    row = cur.fetchone()
    if row:
        locked_until, locked_for = row
        current_time = int(time.time())
        if locked_until and locked_until >= current_time and locked_for != sid:
            # File is locked by another session, so just return to index page (silent fail)
            return redirect(url_for("index"))

    cur.execute("""
        UPDATE labels
        SET user_label = ?,
            model_labels = ?,
            timestamp = ?
        WHERE filepath = ?
        """, (new_label, request.form.get("model_labels"), int(time.time()), filepath))

    cur.execute("INSERT INTO undo_stack (session_id, filepath, timestamp) VALUES (?, ?, ?)", (sid, filepath, int(time.time())))
    get_db().commit()
    return redirect(url_for("index"))


@app.route("/undo", methods=["POST"])
def undo():
    sid = get_session_id()

    cur = db_cursor()
    cur.execute(
        "SELECT undo_id, filepath FROM undo_stack WHERE session_id = ? ORDER BY undo_id DESC LIMIT 1",
        (sid,),
    )
    entry = cur.fetchone()

    if not entry:
        return redirect(url_for("index"))

    undo_id, filepath = entry

    cur.execute(
        "UPDATE labels SET user_label = NULL, timestamp = NULL WHERE filepath = ?",
        (filepath,),
    )

    cur.execute("DELETE FROM undo_stack WHERE undo_id = ?", (undo_id,))

    get_db().commit()

    return redirect(url_for("index", file=filepath))


# ---------- template ----------

# For syntax highlighting (Python Inline Source Syntax Highlighting extension)
html = str

TEMPLATE: html = """
<!DOCTYPE html>
<html>
    <head>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            button {
                /* width: 100px; */
                height: 60px;
                margin: 5px;
            }

            .page {
                width: 100%;
                /* max-width: 1200px; */
                /* margin: 10px; */
                position: relative;
            }

            .page.swiping {
                transition: none;
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2), 0 4px 12px rgba(0, 0, 0, 0.15);
            }

            .page.swipe-right {
                transform: translateX(150vw) rotate(15deg);
                opacity: 0;
            }

            .page.swipe-left {
                transform: translateX(-150vw) rotate(-15deg);
                opacity: 0;
            }

            .page.swipe-up {
                transform: translateY(-150vh) rotate(-5deg);
                opacity: 0;
            }

            .media-row {
                display: flex;
                flex-direction: row;
                min-width: 100%;
                align-items: center;
                justify-content: center;
                gap: 2rem;
            }

            .media-img {
                max-width: 600px;
                width: 100%;
            }

            .btn-label {}

            .btn-current {
                background-color: green;
                color: white;
            }

            .btn-unfinished {
                background-color: orange;
                color: white;
            }

            .btn-none {
                background-color: red;
                color: white;
            }

            .undo-btn {
                margin-top: 5px;
                width: 120px;
                height: 40px;
                background-color: #555;
                color: white;
            }

            .skip-btn {
                margin-top: 5px;
                width: 120px;
                height: 40px;
                background-color: #555;
                color: white;
            }

            .status-msg {
                color: #222;
                margin-bottom: 8px;
            }

            /* extra predictions area */
            #extra_predictions {
                margin-top: 12px;
                border-top: 1px solid #ddd;
                padding-top: 12px;
                display: flex;
                flex-wrap: wrap;
                justify-content: center;
                gap: 8px;
            }

            .pred-pill {
                padding: 6px 10px;
                border-radius: 8px;
                background: #f4f4f4;
                font-size: 90%;
                box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
            }

            .custom-form {
                margin-top: 12px;
            }

            input[name="label"] {
                padding: 8px;
                height: 36px;
                width: 240px;
                font-size: 14px;
            }

            .custom-submit {
                height: 36px;
                margin-left: 6px;
            }

            .desktop-only {}

            .mobile-only {
                display: none;
            }

            .swipe-hint {
                margin-top: 10px;
                color: #444;
                font-size: 14px;
            }

            .labels-overlay {
                position: fixed;
                inset: 0;
                background: rgba(0, 0, 0, 0.45);
                display: none;
                align-items: center;
                justify-content: center;
                z-index: 20;
                padding: 18px;
            }

            .labels-overlay.open {
                display: flex;
            }

            .labels-sheet {
                background: #fff;
                width: 100%;
                max-width: 520px;
                max-height: 90vh;
                overflow-y: auto;
                border-radius: 14px;
                box-shadow: 0 10px 36px rgba(0, 0, 0, 0.18);
                padding: 16px;
            }

            .labels-sheet h4 {
                margin: 0 0 10px 0;
            }

            .labels-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
                gap: 8px;
            }

            .labels-grid button {
                height: 44px;
                border-radius: 10px;
            }

            .overlay-close {
                margin-top: 12px;
                width: 100%;
                height: 40px;
            }

            #swipe_zone {
                width: 100%;
            }

            @media (max-width: 768px) {
                body {
                    /* padding: 12px; */
                }

                .page {
                    max-width: 100%;
                    transition: transform 0.18s ease, opacity 0.18s ease;
                    will-change: transform, opacity;
                    background: #fff;
                    border-radius: 16px;
                    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15), 0 2px 8px rgba(0, 0, 0, 0.1);
                    padding: 16px;
                    margin: 12px;
                    max-width: calc(100% - 24px);
                    box-sizing: border-box;
                }

                h3 {
                    font-size: 16px;
                    margin-bottom: 10px;
                    word-break: break-word;
                }

                .media-row {
                    flex-direction: column;
                    gap: 12px;
                }

                .media-img {
                    max-width: 100%;
                    width: 100%;
                }

                button {
                    width: 100%;
                    max-width: 320px;
                }

                .swipe-hint {
                    font-size: 13px;
                }

                .labels-sheet {
                    padding: 14px;
                }

                .custom-form input[name="label"] {
                    width: 100%;
                    max-width: 320px;
                }

                .custom-submit {
                    width: 100%;
                    max-width: 320px;
                    margin-left: 0;
                    margin-top: 6px;
                }

                .desktop-only {
                    display: none;
                }

                .mobile-only {
                    display: block;
                }
            }

            .swipe-tooltip {
                position: fixed;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                background: rgba(0, 0, 0, 0.8);
                color: white;
                padding: 16px 24px;
                border-radius: 30px;
                font-size: 24px;
                font-weight: bold;
                pointer-events: none;
                opacity: 0;
                transition: opacity 0.2s;
                z-index: 100;
            }

            .swipe-tooltip.visible {
                opacity: 1;
            }
        </style>
    </head>
    <body style="overflow: hidden; width: 100vw; margin: 0; background: #fff; min-height: 100vh;">
        <div id="swipe_tooltip" class="swipe-tooltip"></div>
        <div
            style="font-family: sans-serif; text-align: center; display: flex; flex-direction: column; align-items: center; margin: 10px;">
            <div id="swipe_zone" class="page">
                <h3>{{ filepath }}</h3>
                <div class="media-row">
                    <img class="media-img" src="{{ url_for('spectrogram', filepath=filepath) }}" width="600"><br>
                    <img class="media-img" src="{{ url_for('static', filename='dialekty.png') }}" alt="Logo"
                        width="600"><br>
                </div>

                <!-- Server-side prediction info -->
                <div class="mobile-only">
                    <strong>Model prediction:</strong> {{ top_prediction }} ({{ format_prob(top_prob) if top_prediction else '' }})
                </div>

                <div class="mobile-only">
                    <strong>Original label:</strong> {{ current_label }}
                </div>
                <div class="swipe-hint mobile-only" aria-live="polite" id="swipe_status">Swipe right = model • Swipe up = original • Swipe left = all labels</div>

                <!-- BUTTONS FORM -->
                <form id="buttons_form" class="desktop-only" action="/label" method="post">
                    {% for label in labels %}
                    <button type="submit" name="label" id="{{ label }}"
                        style="cursor: pointer;{% if label == top_prediction %} font-weight: bold;{% endif %}"
                        class="{{ choose_class(label, current_label) }}" value="{{ label }}">{{ label }}{% if label in predictions %} ({{ format_prob(predictions[label]) }}){% endif %}
                    </button>
                    {% endfor %}

                    <input type="hidden" name="filepath" value="{{ filepath }}">
                    <input type="hidden" id="model_labels_buttons" name="model_labels" value="{{ tocsv(predictions) }}">
                    <input type="hidden" id="swipe_label_input" name="label" value="" disabled>
                </form>

                <!-- CUSTOM LABEL FORM (separate so text input can be used) -->
                <div class="custom-form desktop-only">
                    <form id="custom_form" action="/label" method="post" style="display:inline-block;">
                        <input type="hidden" name="filepath" value="{{ filepath }}">
                        <input type="hidden" id="model_labels_custom" name="model_labels" value="{{ tocsv(predictions) }}">
                        <input type="text" id="custom_label_input" name="label" placeholder="Custom dialect">
                        <button type="submit" class="custom-submit" style="cursor: pointer;">Submit Custom</button>
                    </form>
                </div>

                <!-- extra predictions will be inserted here by JS for labels that don't have buttons -->
                <div id="extra_predictions" aria-live="polite" aria-atomic="true"></div>

                <br>

                <form action="/undo" method="post" style="display:inline-block;">
                    <button type="submit" class="undo-btn" style="cursor: pointer;" {% if not undo_available %}disabled{% endif %}>
                        Undo {% if undo_available %}({{ undo_count }}){% endif %}
                    </button>
                </form>

                <form action="/" style="display:inline-block;">
                    <button type="submit" class="skip-btn" style="cursor: pointer;">Skip</button>
                </form>
            </div>
            <div class="labels-overlay" id="labels_overlay" aria-hidden="true">
                <div class="labels-sheet" role="dialog" aria-modal="true" aria-label="All labels">
                    <h4>Pick a label</h4>
                    <div id="overlay_info" class="mobile-only" style="text-align:left; font-size:14px; margin-bottom:10px;">
                    </div>
                    <div class="labels-grid" id="overlay_grid">
                        {% for label in labels %}
                        <button type="button" class="btn-label" data-label="{{ label }}">{{ label }}</button>
                        {% endfor %}
                    </div>
                    <button type="button" class="overlay-close" id="overlay_close">Close</button>
                </div>
            </div>
        </div>

        <script>
            (function() {
                const isMobile = /Android|iPhone|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
                const originalLabel = {{ current_label|tojson }};
                const swipeStatus = document.getElementById('swipe_status');
                const buttonsForm = document.getElementById('buttons_form');
                const hiddenLabelInput = document.getElementById('swipe_label_input');
                const labelOverlay = document.getElementById('labels_overlay');
                const overlayClose = document.getElementById('overlay_close');
                const overlayInfo = document.getElementById('overlay_info');
                const tooltip = document.getElementById('swipe_tooltip');
                // Server-side predictions
                const topPredLabel = {{ top_prediction|tojson }};
                const serverPredictions = {{ predictions|tojson }};

                function setStatus(msg) {
                    if (swipeStatus) swipeStatus.textContent = msg;
                }

                function updateTooltip(text, visible) {
                    if (!tooltip) return;
                    tooltip.textContent = text;
                    if (visible) tooltip.classList.add('visible');
                    else tooltip.classList.remove('visible');
                }

                function performLabel(labelValue, reason = '') {
                    if (!labelValue) return;
                    if (labelOverlay) labelOverlay.classList.remove('open');
                    hiddenLabelInput.disabled = false;
                    hiddenLabelInput.value = labelValue;
                    if (buttonsForm.requestSubmit) {
                        buttonsForm.requestSubmit();
                    } else {
                        buttonsForm.submit();
                    }
                }

                function showOverlay() {
                    if (labelOverlay) {
                        labelOverlay.classList.add('open');
                        labelOverlay.setAttribute('aria-hidden', 'false');
                        updateTooltip('', false);
                    }
                }

                function hideOverlay() {
                    if (labelOverlay) {
                        labelOverlay.classList.remove('open');
                        labelOverlay.setAttribute('aria-hidden', 'true');
                    }
                    resetCard();
                }

                if (overlayClose) overlayClose.addEventListener('click', hideOverlay);
                if (labelOverlay) {
                    labelOverlay.addEventListener('click', (e) => {
                        if (e.target === labelOverlay) hideOverlay();
                    });
                }

                document.querySelectorAll('#overlay_grid button[data-label]').forEach(btn => {
                    btn.addEventListener('click', () => performLabel(btn.dataset.label, 'overlay'));
                });

                // Pre-populate overlay with server-side predictions
                if (overlayInfo && serverPredictions) {
                    const label_names = {{ predictions.keys()|list|tojson }};
                    const rows = label_names.map(lbl => {
                        const prob = serverPredictions[lbl];
                        const pct = prob !== undefined ? (prob * 100).toFixed(2) + '%' : '';
                        const bold = lbl === topPredLabel ? 'font-weight:600;' : '';
                        return `<div style="display:flex; justify-content:space-between; gap:8px; ${bold}"><span>${lbl}</span><span>${pct}</span></div>`;
                    }).join('');
                    overlayInfo.innerHTML = `<div style="margin-bottom:6px;">Current label: <strong>${originalLabel || ''}</strong></div><div style="margin-bottom:6px;">Model prediction: <strong>${topPredLabel || ''}</strong></div>${rows}`;
                }

                let touchStartX = 0;
                let touchStartY = 0;
                let touchStartTime = 0;
                let lastTouchX = 0;
                let lastTouchY = 0;
                let lastTouchTime = 0;
                let rafId = 0;
                let pendingDx = 0;
                let pendingDy = 0;
                let cardX = 0;
                let cardY = 0;
                let velocityX = 0;
                let velocityY = 0;
                let isAnimating = false;
                const swipeZone = document.getElementById('swipe_zone');

                function resetCard() {
                    if (!swipeZone) return;
                    if (rafId) {
                        cancelAnimationFrame(rafId);
                        rafId = 0;
                    }
                    isAnimating = false;
                    cardX = 0;
                    cardY = 0;
                    velocityX = 0;
                    velocityY = 0;
                    swipeZone.classList.remove('swiping','swipe-right','swipe-left','swipe-up','swipe-down');
                    swipeZone.style.transform = '';
                    swipeZone.style.opacity = '';
                    updateTooltip('', false);
                }

                const VELOCITY_THRESHOLD = 0.4;
                const FRICTION = 0.92;
                const MIN_VELOCITY = 0.5;

                function applyTransform(dx, dy) {
                    if (!swipeZone) return;
                    const rot = dx / 20;
                    const dist = Math.sqrt(dx * dx + dy * dy);
                    const opacity = Math.max(0, 1 - dist / 800);
                    swipeZone.style.transform = `translate(${dx}px, ${dy}px) rotate(${rot}deg)`;
                    swipeZone.style.opacity = opacity;
                }

                function applyPreview(dx, dy) {
                    if (!swipeZone) return;
                    swipeZone.classList.add('swiping');
                    applyTransform(dx, dy);

                    const screenW = window.innerWidth;
                    const threshold = Math.min(200, screenW * 0.35);
                    const absX = Math.abs(dx);
                    const absY = Math.abs(dy);
                    
                    let text = '';
                    let show = false;

                    if (absX > absY) {
                        if (absX > threshold) {
                            if (dx > 0) {
                                text = topPredLabel ? 'Model: ' + topPredLabel : 'Model...';
                                show = true;
                            } else {
                                text = 'All Labels';
                                show = true;
                            }
                        }
                    } else {
                        if (dy < 0 && absY > threshold && absY > absX * 1.5) {
                            text = 'Original: ' + (originalLabel || 'None');
                            show = true;
                        }
                    }
                    updateTooltip(text, show);
                }

                function previewCard(dx, dy) {
                    if (!swipeZone || isAnimating) return;
                    pendingDx = dx;
                    pendingDy = dy;
                    cardX = dx;
                    cardY = dy;
                    if (rafId) return;
                    rafId = requestAnimationFrame(() => {
                        applyPreview(pendingDx, pendingDy);
                        rafId = 0;
                    });
                }

                function animateWithMomentum(labelValue, reason, direction) {
                    if (!swipeZone || isAnimating) return;
                    isAnimating = true;
                    swipeZone.classList.add('swiping');
                    updateTooltip('', false);

                    const screenW = window.innerWidth;
                    const screenH = window.innerHeight;
                    const exitX = direction === 'right' ? screenW + 200 : direction === 'left' ? -screenW - 200 : 0;
                    const exitY = direction === 'up' ? -screenH - 200 : 0;
                    let frameCount = 0;
                    const maxFrames = 120;

                    function finish() {
                        if (!isAnimating) return;
                        isAnimating = false;
                        if (rafId) {
                            cancelAnimationFrame(rafId);
                            rafId = 0;
                        }
                        if (direction === 'left') {
                            showOverlay();
                        } else {
                            performLabel(labelValue, reason);
                        }
                    }

                    function tick() {
                        if (!isAnimating) return;
                        frameCount++;

                        velocityX *= FRICTION;
                        velocityY *= FRICTION;

                        const targetVelX = (exitX - cardX) * 0.08;
                        const targetVelY = (exitY - cardY) * 0.08;
                        velocityX += (targetVelX - velocityX) * 0.15;
                        velocityY += (targetVelY - velocityY) * 0.15;

                        cardX += velocityX;
                        cardY += velocityY;

                        applyTransform(cardX, cardY);

                        const offRight = direction === 'right' && cardX > screenW;
                        const offLeft = direction === 'left' && cardX < -screenW;
                        const offUp = direction === 'up' && cardY < -screenH;

                        if (offRight || offLeft || offUp || frameCount >= maxFrames) {
                            finish();
                            return;
                        }

                        rafId = requestAnimationFrame(tick);
                    }

                    rafId = requestAnimationFrame(tick);
                }

                function isStrongUpSwipe(dx, dy, vx, vy, threshold) {
                    const absX = Math.abs(dx);
                    const absY = Math.abs(dy);
                    const speedY = Math.abs(vy);
                    return dy < 0 && absY > 80 && absY > absX * 1.5 && (absY > threshold || speedY > 0.5);
                }

                function handleSwipe(dx, dy, vx, vy) {
                    const absX = Math.abs(dx);
                    const absY = Math.abs(dy);
                    const speedX = Math.abs(vx);
                    const screenW = window.innerWidth;
                    const threshold = Math.min(200, screenW * 0.35);

                    velocityX = vx * 16;
                    velocityY = vy * 16;

                    if (absX > absY) {
                        const isFlick = speedX > VELOCITY_THRESHOLD && absX > 40;
                        const isDrag = absX > threshold;
                        
                        if (isFlick || isDrag) {
                            if (dx > 0) {
                                if (topPredLabel) {
                                    setStatus('Confirmed model: ' + topPredLabel);
                                    animateWithMomentum(topPredLabel, 'swipe-right', 'right');
                                    return true;
                                } else {
                                    setStatus('Model prediction not ready');
                                }
                            } else {
                                setStatus('Choose a label');
                                animateWithMomentum(null, 'swipe-left', 'left');
                                return true;
                            }
                        }
                    } else if (isStrongUpSwipe(dx, dy, vx, vy, threshold)) {
                        setStatus('Confirmed original: ' + (originalLabel || ''));
                        animateWithMomentum(originalLabel, 'swipe-up', 'up');
                        return true;
                    }
                    return false;
                }

                const swipeTarget = document.body;
                swipeTarget.addEventListener('touchstart', (e) => {
                    if (!isMobile) return;
                    const t = e.changedTouches[0];
                    touchStartX = t.clientX;
                    touchStartY = t.clientY;
                    touchStartTime = performance.now();
                    lastTouchX = t.clientX;
                    lastTouchY = t.clientY;
                    lastTouchTime = performance.now();
                    resetCard();
                }, { passive: true });

                swipeTarget.addEventListener('touchmove', (e) => {
                    if (!isMobile || isAnimating) return;
                    const t = e.changedTouches[0];
                    const now = performance.now();
                    lastTouchX = t.clientX;
                    lastTouchY = t.clientY;
                    lastTouchTime = now;
                    const dx = t.clientX - touchStartX;
                    const dy = t.clientY - touchStartY;
                    previewCard(dx, dy);
                }, { passive: true });

                swipeTarget.addEventListener('touchend', (e) => {
                    if (!isMobile || isAnimating) return;
                    const t = e.changedTouches[0];
                    const now = performance.now();
                    const dx = t.clientX - touchStartX;
                    const dy = t.clientY - touchStartY;
                    const dt = now - lastTouchTime;
                    const vx = dt > 0 ? (t.clientX - lastTouchX) / dt : 0;
                    const vy = dt > 0 ? (t.clientY - lastTouchY) / dt : 0;
                    const fired = handleSwipe(dx, dy, vx, vy);
                    if (!fired) resetCard();
                });
            })();
        </script>
    </body>
</html>
"""

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8977)

