# server.py
import asyncio, threading, time, csv, json
from datetime import datetime
from pathlib import Path
from queue import Queue, Empty

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import torch
import websockets
from flask import Flask, Response, jsonify, request, render_template_string

# ================== Config ==================
FLASK_HOST, FLASK_PORT = "0.0.0.0", 5000
WS_HOST, WS_PORT = "0.0.0.0", 8765
SAMPLING_RATE_HZ = 50.0

OUT_DIR = Path("recordings"); OUT_DIR.mkdir(exist_ok=True)
rec_lock = threading.Lock()

# ================== Global State ==================
recording: bool = False
csv_file = None
csv_writer = None
csv_filename: Path | None = None
clients_queues: list[Queue] = []
last_analysis: dict = {}

# ================== Helpers ==================
def broadcast_message(msg: str):
    for q in clients_queues:
        try:
            q.put_nowait(msg)
        except Exception:
            pass

def load_model(model_name: str):
    """Return dict: {'type':..., 'runner':..., 'id2label':{idx:label}}"""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if model_name == "mae":
        from mae.loader import load_mae as load_mae_model
        runner, label_map, id2label = load_mae_model("mae/weight", device=device)
        return {"type": "mae", "runner": runner, "id2label": id2label}

    elif model_name == "bert":
        from bert.loader import BertBeatModel
        runner = BertBeatModel("bert/weight")
        return {"type": "bert", "runner": runner, "id2label": runner.id2label}

    elif model_name == "conformer":
        from conformer.loader import load_conformer_model
        model, classes = load_conformer_model(
            "conformer/weight/model.pt",
            "conformer/weight/config.json",
            device=device
        )
        return {"type": "conformer", "runner": model, "id2label": {i: c for i, c in enumerate(classes)}}

    elif model_name == "fold4":
        from fold4.loader import load_fold4_model
        model, id2label = load_fold4_model(
            "fold4/weight/model.pt",
            "fold4/weight/config.json",
            device=device
        )
        return {"type": "fold4", "runner": model, "id2label": id2label}

    else:
        raise ValueError(f"Unknown model: {model_name}")

def analyze_ecg_file(filename: str, model_name: str = "mae"):
    """Hitung BPM, RR, peak2peak, dan klasifikasi model pilihan."""
    print(f"[ANALYSIS] Using model: {model_name}")
    df = pd.read_csv(filename)

    cols = [c.lower() for c in df.columns]
    df.columns = cols
    if "ecg" not in df.columns:
        if df.shape[1] >= 3:
            df = df.rename(columns={df.columns[2]: "ecg"})
        else:
            return {"error": "CSV format unexpected (no ECG column)"}
    if "resp" not in df.columns:
        if df.shape[1] >= 4:
            df = df.rename(columns={df.columns[3]: "resp"})
        else:
            df["resp"] = np.nan

    ecg = pd.to_numeric(df["ecg"], errors="coerce").dropna().to_numpy()
    resp = pd.to_numeric(df["resp"], errors="coerce").dropna().to_numpy()
    if ecg.size < max(10, int(SAMPLING_RATE_HZ*2)):
        return {"error": "No data recorded (too short or NaN)"}

    duration_s = len(ecg) / SAMPLING_RATE_HZ

    # ECG peaks
    dist = max(1, int(SAMPLING_RATE_HZ * 0.4))
    thr = np.nanmean(ecg) + np.nanstd(ecg)
    peaks, _ = find_peaks(ecg, distance=dist, height=thr)
    bpm = (len(peaks) * 60.0) / duration_s if duration_s > 0 else 0.0

    # Resp peaks (opsional)
    try:
        rdist = max(1, int(SAMPLING_RATE_HZ * 0.8))
        rpeaks, _ = find_peaks(resp, distance=rdist)
        rr = (len(rpeaks) * 60.0) / duration_s if duration_s > 0 else 0.0
    except Exception:
        rpeaks = np.array([], dtype=int)
        rr = 0.0

    # Simple diagnosis by heart rate
    if bpm > 100:
        diagnosis = "Takikardi"
    elif bpm < 60:
        diagnosis = "Bradikardi"
    else:
        diagnosis = "Normal"

    # ===== Model inference =====
    pred_label = "?"
    try:
        mi = load_model(model_name)
        mtype = mi["type"]

        if mtype == "mae":
            runner = mi["runner"]
            seg = ecg[:512] if ecg.size >= 512 else np.pad(ecg, (0, 512-ecg.size))
            pred_ids, _ = runner.predict(seg[np.newaxis, :])
            pred_label = mi["id2label"].get(int(pred_ids[0]), "?")

        elif mtype == "bert":
            runner = mi["runner"]
            x = ecg[:256]
            if x.size < 256:
                x = np.pad(x, (0, 256-x.size))
            x = x - x.min()
            den = (x.max() if x.max() > 0 else 1.0)
            x = (x / den) * 255.0
            txt = " ".join(str(int(v)) for v in x)
            res = runner.predict_texts([txt])[0]
            pred_label = res["label"]

        elif mtype == "conformer":
            from conformer.loader import conformer_predict
            idx, _ = conformer_predict(mi["runner"], ecg)
            pred_label = mi["id2label"].get(idx, "?")

        elif mtype == "fold4":
            from fold4.loader import fold4_predict
            idx, _ = fold4_predict(mi["runner"], ecg)
            pred_label = mi["id2label"].get(idx, "?")

    except Exception as e:
        pred_label = f"Model error: {e}"

    return {
        "filename": filename,
        "duration_s": round(duration_s, 2),
        "bpm": round(bpm, 2),
        "n_peaks": int(len(peaks)),
        "peak_indices": peaks.tolist()[:200],
        "resp_rate": round(rr, 2),
        "n_resp_peaks": int(len(rpeaks)),
        "diagnosis": diagnosis,
        "model_class": pred_label,
    }

# ================== WebSocket (ESP32) ==================
async def ws_handler(websocket):
    global recording, csv_writer, csv_file 
    print(f"[WS] Connected: {websocket.remote_address}")
    try:
        async for message in websocket:
            parts = message.strip().split(",")
            if len(parts) < 2:
                continue
            try:
                esp_millis = float(parts[0]); ecg_val = float(parts[1])
                resp_val = float(parts[2]) if len(parts) > 2 else 0.0
            except Exception:
                continue

            payload = {
                "recv_time": time.time(),
                "esp_millis": esp_millis,
                "ecg": ecg_val,
                "resp": resp_val
            }
            broadcast_message(json.dumps(payload))

            with rec_lock:
                if recording and (csv_writer is not None) and (csv_file is not None) and (not csv_file.closed):
                    csv_writer.writerow([payload["recv_time"], esp_millis, ecg_val, resp_val])
    except Exception as e:
        print("[WS] Error:", e)
    finally:
        print(f"[WS] Disconnected: {websocket.remote_address}")

async def ws_server_main():
    print(f"[WS] Running on ws://{WS_HOST}:{WS_PORT}")
    async with websockets.serve(ws_handler, WS_HOST, WS_PORT):
        await asyncio.Future()

def start_ws_server_in_thread():
    def runner(): asyncio.run(ws_server_main())
    t = threading.Thread(target=runner, daemon=True)
    t.start()

# ================== Flask (UI + SSE) ==================
app = Flask(__name__)

INDEX_HTML = """
<!doctype html>
<html lang="id">
<head>
<meta charset="utf-8" />
<title>Realtime EKG & Respirasi Viewer</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
<style>
  body {
    font-family: 'Segoe UI', sans-serif;
    background: #f8f9fa;         
    color: #212529;            
    padding: 24px;
  }
  .card {
    border-radius: 14px;
    box-shadow: 0 4px 12px rgba(0,0,0,.1);
    background: #ffffff;
    padding: 18px;
  }
  h2 {
    text-align: center;
    color: #007bff;               
    font-weight: 700;
    margin-bottom: 12px;
  }
  canvas {
    margin-top: 10px;
    background: #ffffff;         
    border-radius: 10px;
    border: 1px solid #ddd;       
  }
  .controls > * {
    margin-right: 10px;
  }
  #ecgWrap, #respWrap {
    height: 260px;                
  }
  #respWrap {
    height: 200px;
  }
  .kpi .val {
    font-size: 28px;
    font-weight: 700;
    color: #007bff;
  }
  .kpi .lbl {
    color: #555;
    font-weight: 600;
  }
  pre {
    background: #f8f9fa;
    color: #333;
    border-radius: 10px;
    padding: 12px;
    max-height: 300px;
    overflow: auto;
    border: 1px solid #ccc;
  }
</style>
</head>

<body>
  <div class="container">
    <div class="card">
      <h2>Realtime EKG & Respirasi HUMIK</h2>

      <div class="controls d-flex align-items-center mb-2">
        <select id="model" class="form-select form-select-sm" style="width:auto;">
          <option value="mae">MAE</option>
          <option value="conformer">Conformer</option>
          <option value="bert">BERT</option>
          <option value="fold4">Fold4</option>
        </select>
        <button id="startBtn" class="btn btn-success btn-sm">▶️ Start</button>
        <button id="stopBtn" class="btn btn-danger btn-sm" disabled>⏹ Stop</button>
        <span id="status" class="ms-2 text-secondary">Idle</span>
        <span id="duration" class="ms-2 text-warning fw-bold"></span>
      </div>

      <div id="ecgWrap"><canvas id="ecgChart"></canvas></div>
      <div id="respWrap" class="mt-3"><canvas id="respChart"></canvas></div>

      <hr>
      <h5>Hasil Analisis Terakhir</h5>
      <div class="row text-center kpi">
        <div class="col-6 col-md-2"><div class="lbl">BPM</div><div id="bpm" class="val">-</div></div>
        <div class="col-6 col-md-2"><div class="lbl">Durasi (s)</div><div id="duration_s" class="val">-</div></div>
        <div class="col-6 col-md-2"><div class="lbl">R-Peaks</div><div id="n_peaks" class="val">-</div></div>
        <div class="col-6 col-md-2"><div class="lbl">RR (rpm)</div><div id="rr" class="val">-</div></div>
        <div class="col-6 col-md-2"><div class="lbl">Diagnosa</div><div id="diagnosis" class="val">-</div></div>
        <div class="col-6 col-md-2"><div class="lbl">Model Kelas</div><div id="model_class" class="val">-</div></div>
      </div>
      <pre id="analysis">-</pre>
    </div>
  </div>

<script>
  const startBtn = document.getElementById('startBtn');
  const stopBtn  = document.getElementById('stopBtn');
  const statusEl = document.getElementById('status');
  const durationEl = document.getElementById('duration');

  // ECG chart
  const ecgCtx = document.getElementById('ecgChart').getContext('2d');
  const ecgData = { labels: [], datasets: [{ label:'ECG', data:[], borderColor:'#19e3a6', borderWidth:1.2, pointRadius:0, tension:0 }] };
  const ecgChart = new Chart(ecgCtx, {
    type:'line',
    data: ecgData,
    options: { responsive:true, maintainAspectRatio:false, scales:{ x:{ display:false }, y:{ min:-600, max:600 } }, plugins:{ legend:{ display:false } } }
  });

  // Resp chart
  const respCtx = document.getElementById('respChart').getContext('2d');
  const respData = { labels: [], datasets: [{ label:'Resp', data:[], borderColor:'#60a6ff', borderWidth:1.2, pointRadius:0, tension:0.2 }] };
  const respChart = new Chart(respCtx, {
    type:'line',
    data: respData,
    options: { responsive:true, maintainAspectRatio:false, scales:{ x:{ display:false }, y:{ min:-5, max:5 } }, plugins:{ legend:{ display:false } } }
  });

  // SSE stream
  let t0 = null;
  const evt = new EventSource('/stream');
  evt.onmessage = e => {
    const obj = JSON.parse(e.data);
    if (t0 === null) t0 = obj.recv_time;
    const t = (obj.recv_time - t0).toFixed(2);

    ecgData.labels.push(t);
    ecgData.datasets[0].data.push(obj.ecg);

    respData.labels.push(t);
    respData.datasets[0].data.push(obj.resp ?? 0);

    const LIM = 600;
    if (ecgData.labels.length > LIM) {
      ecgData.labels.shift(); ecgData.datasets[0].data.shift();
      respData.labels.shift(); respData.datasets[0].data.shift();
    }
    ecgChart.update('none');
    respChart.update('none');
  };

  // Duration timer
  let recStart = null, durTimer = null;
  function tick() {
    if (!recStart) return;
    const s = (Date.now() - recStart) / 1000;
    durationEl.innerText = `Recording: ${s.toFixed(1)} s`;
  }

  startBtn.onclick = async () => {
    startBtn.disabled = true;
    const model = document.getElementById('model').value;
    const res = await fetch('/start?model='+model, {method:'POST'});
    const j = await res.json();
    statusEl.innerText = 'Recording...';
    stopBtn.disabled = false;
    recStart = Date.now();
    durationEl.innerText = 'Recording: 0.0 s';
    durTimer = setInterval(tick, 100);
  };

  stopBtn.onclick = async () => {
    stopBtn.disabled = true;
    const model = document.getElementById('model').value;
    const res = await fetch('/stop?model='+model, {method:'POST'});
    const j = await res.json();
    startBtn.disabled = false;
    statusEl.innerText = 'Idle';
    clearInterval(durTimer); durTimer = null; recStart = null; durationEl.innerText = '';

    // tampilkan ringkasan
    document.getElementById('bpm').innerText = j.bpm ?? '-';
    document.getElementById('duration_s').innerText = j.duration_s ?? '-';
    document.getElementById('n_peaks').innerText = j.n_peaks ?? '-';
    document.getElementById('rr').innerText = j.resp_rate ?? '-';
    document.getElementById('diagnosis').innerText = j.diagnosis ?? '-';
    document.getElementById('model_class').innerText = j.model_class ?? '-';
    document.getElementById('analysis').innerText = JSON.stringify(j, null, 2);
  };
</script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(INDEX_HTML)

@app.route("/stream")
def stream():
    def gen(q: Queue):
        try:
            while True:
                try:
                    msg = q.get(timeout=15)
                    yield f"data: {msg}\n\n"
                except Empty:
                    yield ": keep-alive\\n\\n"
        finally:
            try: clients_queues.remove(q)
            except ValueError: pass

    q = Queue()
    clients_queues.append(q)
    return Response(gen(q), mimetype="text/event-stream")

@app.route("/start", methods=["POST"])
def start_record():
    global recording, csv_file, csv_writer, csv_filename 
    model = request.args.get("model", "mae")
    with rec_lock:
        if recording:
            return jsonify({"status": "already recording", "file": str(csv_filename) if csv_filename else None})
        csv_filename = OUT_DIR / f"ecg_{datetime.now():%Y%m%d_%H%M%S}.csv"
        csv_file = open(csv_filename, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["server_time", "esp_millis", "ecg", "resp"])
        recording = True
        print(f"[REC] Started: {csv_filename}")
    return jsonify({"status": "started", "file": str(csv_filename), "model": model})

@app.route("/stop", methods=["POST"])
def stop_record():
    global recording, csv_file, csv_writer, csv_filename, last_analysis
    model_name = request.args.get("model", "mae")
    with rec_lock:
        if not recording:
            if csv_filename and Path(csv_filename).exists():
                analysis = analyze_ecg_file(str(csv_filename), model_name)
                last_analysis = analysis
                return jsonify(analysis)
            return jsonify({"status": "not recording"})
        recording = False
        if csv_file and not csv_file.closed:
            csv_file.close()
        csv_writer = None
        print("[REC] Stopped.")
    analysis = analyze_ecg_file(str(csv_filename), model_name)
    last_analysis = analysis
    return jsonify(analysis)

if __name__ == "__main__":
    start_ws_server_in_thread()
    print(f"[FLASK] Running on http://{FLASK_HOST}:{FLASK_PORT}")
    app.run(host=FLASK_HOST, port=FLASK_PORT)
