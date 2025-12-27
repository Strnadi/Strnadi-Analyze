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

matplotlib.use("Agg")
import matplotlib.pyplot as plt

app = Flask(__name__, static_folder='')

DATABASE_PATH = os.environ.get("DATABASE_PATH", "database.db")


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
        model_labels TEXT
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
    cur = db_cursor()
    cur.execute("SELECT DISTINCT original_label FROM labels UNION SELECT DISTINCT user_label FROM labels")
    labels = cur.fetchall()
    return [label[0] for label in labels]


def choose_class(label, curr_label):
    if label == curr_label:
        return "btn-current"

    if label == "None":
        return "btn-none"

    if label == "Unfinished":
        return "btn-unfinished"

    return "btn-label"


# ---------- routes ----------
@app.route("/")
def index():
    requested_file = request.args.get("file")

    cur = db_cursor()

    if requested_file:
        cur.execute(
            "SELECT filepath, original_label FROM labels WHERE filepath = ? LIMIT 1",
            (requested_file,),
        )
    else:
        cur.execute(
            "SELECT filepath, original_label FROM labels WHERE user_label IS NULL ORDER BY RANDOM() LIMIT 1"
        )

    row = cur.fetchone()
    if not row:
        return "No files left to label! 🎉"

    filepath, original_label = row

    sid = get_session_id()
    cur.execute("SELECT COUNT(*) FROM undo_stack WHERE session_id = ?", (sid,))
    undo_count = cur.fetchone()[0]

    return render_template_string(
        TEMPLATE,
        filepath=filepath,
        current_label=original_label,
        choose_class=choose_class,
        labels=get_all_labels(),
        undo_available=(undo_count > 0),
        undo_count=undo_count
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


@app.route("/sound/<path:filepath>")
def sound(filepath):
    if not os.path.exists(filepath):
        return "File not found", 404

    audio, sr = librosa.load(filepath, sr=48000, mono=False)
    if audio.ndim > 1:
        if audio.shape[0] > 1 and np.any(audio[1]):
            audio = np.mean(audio, axis=0)
        else:
            audio = audio[0]

    audio = librosa.util.normalize(audio)
    audio = np.clip(audio, -1.0, 1.0)

    int16 = (audio * 32767.0).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(int16.tobytes())

    buf.seek(0)
    return send_file(buf, mimetype='audio/wav')


@app.route("/model")
def model():
    return send_file("good-model-1.tflite", mimetype='application/octet-stream', conditional=True)


@app.route("/label", methods=["POST"])
def label_file():
    filepath = request.form["filepath"]
    new_label = request.form["label"]
    sid = get_session_id()
    cur = db_cursor()

    # Check if the file is already labeled
    cur.execute("SELECT user_label FROM labels WHERE filepath = ?", (filepath,))
    row = cur.fetchone()
    if row and row[0] is not None:
        # File is already labeled, so just return to index page (silent fail)
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
TEMPLATE = """
<!DOCTYPE html>
<html>
  <head>
    <style>
      button {
        /* width: 100px; */ height: 60px; margin:5px;
      }
      .btn-label { }
      .btn-current { background-color: green; color: white; }
      .btn-unfinished { background-color: orange; color: white; }
      .btn-none { background-color: red; color: white; }
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
      .status-msg { color: #222; margin-bottom: 8px; }
      /* extra predictions area */
      #extra_predictions {
        margin-top: 12px;
        border-top: 1px solid #ddd;
        padding-top: 12px;
        display:flex;
        flex-wrap:wrap;
        justify-content:center;
        gap:8px;
      }
      .pred-pill {
        padding:6px 10px;
        border-radius:8px;
        background:#f4f4f4;
        font-size:90%;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
      }
      .custom-form {
        margin-top: 12px;
      }
      input[name="label"] {
        padding:8px;
        height:36px;
        width:240px;
        font-size:14px;
      }
      .custom-submit {
        height:36px;
        margin-left:6px;
      }
    </style>
  </head>
  <body style="font-family: sans-serif; text-align: center; display: flex; flex-direction: column; align-items: center;">
    <h3>{{ filepath }}</h3>
    <div style="display: flex; flex-direction: row; min-width: 100%; align-items: center; justify-content: center; gap: 2rem;">
      <img src="{{ url_for('spectrogram', filepath=filepath) }}" width="600"><br>
      <img src="{{ url_for('static', filename='dialekty.png') }}" alt="Logo" width="600"><br>
    </div>

    <!-- BUTTONS FORM -->
    <form id="buttons_form" action="/label" method="post">
      {% for label in labels %}
        <button
          type="submit"
          name="label"
          id="{{ label }}"
          style="cursor: pointer;"
          class="{{ choose_class(label, current_label) }}"
          value="{{ label }}"
        >{{ label }}</button>
      {% endfor %}

      <input type="hidden" name="filepath" value="{{ filepath }}">
      <input type="hidden" id="model_labels_buttons" name="model_labels" value="">
    </form>

    <!-- CUSTOM LABEL FORM (separate so text input can be used) -->
    <div class="custom-form">
      <form id="custom_form" action="/label" method="post" style="display:inline-block;">
        <input type="hidden" name="filepath" value="{{ filepath }}">
        <input type="hidden" id="model_labels_custom" name="model_labels" value="">
        <input type="text" id="custom_label_input" name="label" placeholder="Type custom label (e.g. MySpecies)">
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

    <script type="module">
      import { loadLiteRt, loadAndCompile, Tensor } from 'https://esm.sh/@litertjs/core@0.2.1';

      let webGpu = !!navigator.gpu;

      (async function() {

        if (!window._liteRtPromise) {
            window._liteRtPromise = loadLiteRt("https://cdn.jsdelivr.net/npm/@litertjs/core@0.2.1/wasm/");
        }
        await window._liteRtPromise;

        if (!window._birdModelPromise) {
            window._birdModelPromise = (async () => {
                const buffer = await fetch("/model").then(r => r.arrayBuffer());
                return await loadAndCompile(
                    new Uint8Array(buffer),
                    { accelerator: webGpu ? "webgpu" : "wasm" }
                );
            })();
        }
        const model = await window._birdModelPromise;

        const fileBuffer = await fetch("/sound/{{ filepath }}").then(response => response.arrayBuffer());

        const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        let decoded;
        try {
            decoded = await audioCtx.decodeAudioData(fileBuffer.slice(0));
        } catch (err) {
            decoded = await new Promise((resolve, reject) => {
                audioCtx.decodeAudioData(fileBuffer.slice(0), resolve, reject);
            });
        }

        const targetRate = 48000;
        const OfflineCtx = window.OfflineAudioContext || window.webkitOfflineAudioContext;

        let samples;

        if (OfflineCtx) {
            try {
                const length = Math.ceil(decoded.duration * targetRate);
                const offlineCtx = new OfflineCtx(1, Math.max(1, length), targetRate);
                const bufferSource = offlineCtx.createBufferSource();
                bufferSource.buffer = decoded;
                bufferSource.connect(offlineCtx.destination);
                bufferSource.start(0);
                const renderedBuffer = await offlineCtx.startRendering();
                samples = renderedBuffer.getChannelData(0);
            } catch (err) {
                console.warn('OfflineAudioContext resampling failed, using linear interpolation.', err);
            }
        }

        if (!samples) {
            const origRate = decoded.sampleRate;
            samples = decoded.getChannelData(0);

            if (origRate !== targetRate) {
                const ratio = origRate / targetRate;
                const newLen = Math.ceil(samples.length / ratio);
                const output = new Float32Array(newLen);
                for (let i = 0; i < newLen; i++) {
                    const srcIndex = i * ratio;
                    const i0 = Math.floor(srcIndex);
                    const i1 = Math.min(i0 + 1, samples.length - 1);
                    const frac = srcIndex - i0;
                    output[i] = samples[i0] * (1 - frac) + samples[i1] * frac;
                }

                samples = output;
            }
        }

        await audioCtx.close();

        const shape = [1, samples.length];
        const inputTensor = new Tensor(samples, shape);
        if (webGpu) inputTensor.moveTo("webgpu");

        let outputs;
        let startTime, endTime;
        try {
            startTime = performance.now();
            outputs = await model.run(inputTensor);
            endTime = performance.now();
        } catch (err) {
            startTime = performance.now();
            outputs = await model.run([inputTensor]);
            endTime = performance.now();
        }

        console.log(`Inference took ${endTime - startTime} ms`);

        inputTensor.delete();

        const outs = Array.isArray(outputs) ? outputs : [outputs];
        const out = outs[0];
        try {
            const cpu = webGpu ? await out.moveTo('wasm') : out;
            const typed = cpu.toTypedArray();
            // keep label_names in sync with your model
            const label_names = ['BC', 'BE', 'BhBl', 'BlBh', 'None', 'Unfinished', 'XB']
            const probs = Array.from(typed);
            const probsPct = probs.map(x => (Math.round(x * 10000) / 100).toFixed(2) + '%');
            const result = Object.fromEntries(label_names.map((lbl, i) => [lbl, probsPct[i] ?? '']));

            const modelLabelsStr = Object.entries(result).map(([label, prob]) => `${label}=${prob}`).join(',');
            // set model_labels into both hidden fields (buttons and custom forms)
            const mb = document.getElementById('model_labels_buttons');
            const mc = document.getElementById('model_labels_custom');
            if (mb) mb.value = modelLabelsStr;
            if (mc) mc.value = modelLabelsStr;

            // append probs to existing buttons, and collect labels missing DOM buttons
            const missing = [];
            const max = Math.max(...probs);
            const maxLabel = label_names[probs.indexOf(max)];

            label_names.forEach(label => {
                const elem = document.getElementById(label);
                if (elem) {
                    // append probability to button text (preserve existing inner text)
                    elem.innerHTML = label + ' (' + result[label] + ')';
                    if (label === maxLabel) {
                        elem.style.fontWeight = 'bold';
                    }
                } else {
                    missing.push({label, prob: result[label]});
                }
            });

            // render missing predictions under the buttons
            const extras = document.getElementById('extra_predictions');
            extras.innerHTML = '';
            if (missing.length > 0) {
                missing.forEach(item => {
                    const pill = document.createElement('div');
                    pill.className = 'pred-pill';
                    pill.textContent = item.label + ' — ' + item.prob;
                    extras.appendChild(pill);
                });
            } else {
                // optional: show top prediction if nothing missing
                const top = document.createElement('div');
                top.className = 'pred-pill';
                top.textContent = 'Top prediction: ' + maxLabel + ' (' + (Math.round(max*10000)/100).toFixed(2) + '%)';
                extras.appendChild(top);
            }

            cpu.delete();
        } catch (err) {
            console.warn('Failed to read output tensor:', err);
            console.log('Failed to read output tensor: ' + err);
        }

      })();

    </script>
  </body>
</html>
"""

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8977)

