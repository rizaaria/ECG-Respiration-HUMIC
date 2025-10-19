# server.py
# Realtime ECG + RESP server: WebSocket (ESP32) -> Flask (SSE + UI) + CSV recording + analysis
import asyncio
import threading
import time
import csv
import json
from queue import Queue, Empty
from datetime import datetime
from pathlib import Path
import math
import random

from flask import Flask, Response, request, render_template_string, jsonify
import websockets
import pandas as pd
import numpy as np
from scipy.signal import find_peaks

# -------------------------
# Konfigurasi
# -------------------------
WS_HOST = "0.0.0.0"
WS_PORT = 8765

FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000

# Asumsi sampling rate ESP32 (ubah sesuai ESP)
SAMPLING_RATE_HZ = 50.0

# Folder output CSV
OUT_DIR = Path("./recordings")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------
# State global
# -------------------------
clients_queues = []      # list of Queue() untuk SSE listener browser
recording = False
csv_file = None
csv_writer = None
csv_filename = None
last_analysis = None     # dictionary hasil analisis terakhir

# thread-safe lock for recording toggles
from threading import Lock
rec_lock = Lock()

# -------------------------
# Utility: broadcast ke semua SSE listener
# -------------------------
def broadcast_message(msg: str):
    """Masukkan pesan (string) ke semua client queues."""
    for q in clients_queues:
        try:
            q.put_nowait(msg)
        except Exception:
            pass

# -------------------------
# WebSocket server (untuk ESP32)
# -------------------------
async def ws_handler(websocket):
    """
    Handler websocket menerima pesan format:
        "<millis>,<ecg>"  (legacy)
    atau
        "<millis>,<ecg>,<resp>"  (preferred)
    Kemudian broadcast JSON ke SSE clients:
       {"recv_time": <epoch>, "esp_millis": <float>, "ecg": <float>, "resp": <float>}
    """
    print(f"[WS] New connection: {websocket.remote_address}")
    try:
        async for message in websocket:
            # parse message
            try:
                parts = message.strip().split(",")
                if len(parts) >= 3:
                    esp_millis = float(parts[0])
                    ecg_val = float(parts[1])
                    resp_val = float(parts[2])
                elif len(parts) == 2:
                    esp_millis = float(parts[0])
                    ecg_val = float(parts[1])
                    resp_val = 0.0
                else:
                    # format tak sesuai, skip
                    print("[WS] bad format, skipping:", message)
                    continue
            except Exception as e:
                print("[WS] parse error:", e, "msg:", message)
                continue

            recv_time = time.time()
            payload = {
                "recv_time": recv_time,         # waktu server penerimaan (epoch float)
                "esp_millis": esp_millis,       # millis dari ESP32
                "ecg": ecg_val,
                "resp": resp_val
            }
            msg_text = json.dumps(payload)

            # broadcast ke browser (SSE)
            broadcast_message(msg_text)

            # jika sedang merekam, tulis ke CSV (thread-safe)
            with rec_lock:
                global recording, csv_writer, csv_filename
                if recording and csv_writer is not None:
                    # format recorded: server_time, esp_millis, ecg, resp
                    csv_writer.writerow([recv_time, esp_millis, ecg_val, resp_val])
    except websockets.exceptions.ConnectionClosed:
        print("[WS] Connection closed")
    except Exception as e:
        print("[WS] Exception:", e)
    finally:
        print(f"[WS] Disconnected: {websocket.remote_address}")

async def ws_server_main():
    print(f"[WS] Starting WebSocket server on ws://{WS_HOST}:{WS_PORT}")
    async with websockets.serve(ws_handler, WS_HOST, WS_PORT):
        await asyncio.Future()  # run forever

def start_ws_server_in_thread():
    def runner():
        asyncio.run(ws_server_main())
    t = threading.Thread(target=runner, daemon=True)
    t.start()

# -------------------------
# Flask app (web UI + SSE)
# -------------------------
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
        background: linear-gradient(135deg, #e3f2fd, #f3e5f5);
        padding: 30px;
      }
      .card {
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        padding: 20px;
        background: white;
      }
      h2 {
        text-align: center;
        color: #1976d2;
        font-weight: 700;
      }
      button {
        border-radius: 8px;
      }
      pre {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        max-height: 300px;
        overflow-y: auto;
      }
      #duration {
        font-weight: bold;
        margin-left: 10px;
        color: #d32f2f;
      }
      #status {
        font-weight: 500;
        margin-left: 10px;
      }
      canvas {
        margin-top: 15px;
      }
    </style>
  </head>
  <body>
    <div class="container">
      <div class="card">
        <h2>📊 Realtime EKG & Respirasi Viewer</h2>
        <div class="d-flex align-items-center mb-3">
          <button id="startBtn" class="btn btn-success me-2">▶️ Start Recording</button>
          <button id="stopBtn" class="btn btn-danger me-2" disabled>⏹ Stop Recording</button>
          <span id="status" class="text-secondary">Idle</span>
          <span id="duration"></span>
        </div>

        <!-- ECG Chart -->
        <h5 class="text-primary">EKG (mV)</h5>
        <canvas id="ecgChart" height="250"></canvas>

        <!-- Respiration Chart -->
        <h5 class="text-success mt-4">Respirasi (unit relatif)</h5>
        <canvas id="respChart" height="200"></canvas>

        <hr>
        <h4>🩺 Hasil Analisis Terakhir</h4>
        <div class="row text-center mb-3" id="analysis-summary" style="display: none;">
          <div class="col-md-3">
            <div class="fw-bold text-secondary">BPM</div>
            <div id="bpm" class="fs-4 fw-bold text-primary">-</div>
          </div>
          <div class="col-md-3">
            <div class="fw-bold text-secondary">Durasi (s)</div>
            <div id="duration_s" class="fs-4 fw-bold text-primary">-</div>
          </div>
          <div class="col-md-3">
            <div class="fw-bold text-secondary">Jumlah Puncak</div>
            <div id="n_peaks" class="fs-4 fw-bold text-primary">-</div>
          </div>
          <div class="col-md-3">
            <div class="fw-bold text-secondary">Diagnosis</div>
            <div id="diagnosis" class="fs-5 fw-bold text-danger">-</div>
          </div>
        </div>
        <pre id="analysis">-</pre>
      </div>
    </div>

    <script>
      const startBtn = document.getElementById('startBtn');
      const stopBtn = document.getElementById('stopBtn');
      const statusEl = document.getElementById('status');
      const analysisEl = document.getElementById('analysis');
      const durationEl = document.getElementById('duration');

      let isRecording = false;
      let startTime = null;
      let recordStartIndex = null;
      let durationTimer = null;
      let recordStartTime = null;

      // ECG chart setup
      const ecgCtx = document.getElementById('ecgChart').getContext('2d');
      const ecgData = {
        labels: [],
        datasets: [ {
          label: 'ECG',
          data: [],
          borderColor: '#1976d2',
          borderWidth: 1.5,
          pointRadius: 0,
          tension: 0
        } ]
      };

      // Respiration chart setup
      const respCtx = document.getElementById('respChart').getContext('2d');
      const respData = {
        labels: [],
        datasets: [ {
          label: 'Respiration',
          data: [],
          borderColor: '#43a047',
          borderWidth: 1.2,
          pointRadius: 0,
          tension: 0.2
        } ]
      };

      const configECG = {
        type: 'line',
        data: ecgData,
        options: {
          animation: false,
          responsive: true,
          scales: {
            x: { title: { display: true, text: 'Time (s)' } },
            y: { suggestedMin: -2000, suggestedMax: 2000 }
          },
          plugins: { legend: { display: false } }
        }
      };
      const configRESP = {
        type: 'line',
        data: respData,
        options: {
          animation: false,
          responsive: true,
          scales: {
            x: { title: { display: true, text: 'Time (s)' } },
            y: { suggestedMin: -1, suggestedMax: 1 }
          },
          plugins: { legend: { display: false } }
        }
      };

      const ecgChart = new Chart(ecgCtx, configECG);
      const respChart = new Chart(respCtx, configRESP);

      // SSE stream
      const evtSource = new EventSource('/stream');
      evtSource.onmessage = function(e) {
        try {
          const obj = JSON.parse(e.data);
          if (startTime === null) startTime = obj.recv_time;
          const t = (obj.recv_time - startTime).toFixed(2);

          // push ecg & resp
          ecgData.labels.push(t);
          ecgData.datasets[0].data.push(obj.ecg);

          respData.labels.push(t);
          // if `resp` missing in incoming payload, fallback to a gentle dummy
          const respVal = (typeof obj.resp !== 'undefined') ? obj.resp : (Math.sin(t / 5.0) * 0.5 + Math.random() * 0.05);
          respData.datasets[0].data.push(respVal);

          if (ecgData.labels.length > 1000) {
            ecgData.labels.shift();
            ecgData.datasets[0].data.shift();
            respData.labels.shift();
            respData.datasets[0].data.shift();
            if (recordStartIndex !== null) recordStartIndex--;
          }
          ecgChart.update();
          respChart.update();
        } catch (err) {
          console.warn("SSE parse err", err);
        }
      };

      function updateDuration() {
        if (!recordStartTime) return;
        const elapsed = (Date.now() - recordStartTime) / 1000;
        durationEl.innerText = `Recording: ${elapsed.toFixed(1)} s`;
      }

      startBtn.onclick = async () => {
        startBtn.disabled = true;
        const res = await fetch('/start', { method: 'POST' });
        const j = await res.json();
        statusEl.innerText = 'Recording → ' + j.filename;
        stopBtn.disabled = false;
        isRecording = true;
        recordStartIndex = ecgData.labels.length;
        recordStartTime = Date.now();
        durationEl.innerText = 'Recording: 0.0 s';
        durationTimer = setInterval(updateDuration, 100);
        ecgData.datasets[0].borderColor = '#d32f2f';
        ecgChart.update();
      };

      stopBtn.onclick = async () => {
        stopBtn.disabled = true;
        const res = await fetch('/stop', { method: 'POST' });
        const j = await res.json();
        statusEl.innerText = 'Idle';
        startBtn.disabled = false;
        isRecording = false;
        clearInterval(durationTimer);
        durationEl.innerText = '';
        recordStartTime = null;
        ecgData.datasets[0].borderColor = '#1976d2';
        ecgChart.update();

        // tampilkan hasil analisis
        showAnalysis(j.analysis);
      };

      async function fetchLast() {
        const r = await fetch('/last_analysis');
        if (r.ok) {
          const j = await r.json();
          if (j && j.analysis) showAnalysis(j.analysis);
        }
      }

      function showAnalysis(a) {
        analysisEl.innerText = JSON.stringify(a, null, 2);
        document.getElementById("analysis-summary").style.display = "flex";
        document.getElementById("bpm").innerText = a.bpm || "-";
        document.getElementById("duration_s").innerText = a.duration_s || "-";
        document.getElementById("n_peaks").innerText = a.n_peaks || "-";
        document.getElementById("diagnosis").innerText = a.diagnosis || "-";
      }

      fetchLast();
    </script>
  </body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(INDEX_HTML)

@app.route("/stream")
def stream():
    def event_stream(q: Queue):
        try:
            while True:
                try:
                    msg = q.get(timeout=15)  # wait up to 15s
                    # SSE: kirim data: <json>\n\n
                    yield f"data: {msg}\n\n"
                except Empty:
                    # kirim komentar keep-alive supaya koneksi tidak tertutup (optional)
                    yield ": keep-alive\n\n"
        finally:
            # saat client disconnect, hapus queue dari list
            try:
                clients_queues.remove(q)
            except ValueError:
                pass

    q = Queue()
    clients_queues.append(q)
    return Response(event_stream(q), mimetype="text/event-stream")

# API untuk start/stop recording (diperbarui untuk menyimpan RESP juga)
@app.route("/start", methods=["POST"])
def api_start():
    global recording, csv_file, csv_writer, csv_filename
    with rec_lock:
        if recording:
            return jsonify({"status": "already_recording"})
        # buat file baru
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = OUT_DIR / f"ecg_{timestamp}.csv"
        csv_file = open(csv_filename, "w", newline="")
        csv_writer = csv.writer(csv_file)
        # header sekarang menyertakan resp
        csv_writer.writerow(["server_time", "esp_millis", "ecg", "resp"])
        recording = True
        print("[REC] Started:", csv_filename)
    return jsonify({"status": "recording_started", "filename": str(csv_filename)})

@app.route("/stop", methods=["POST"])
def api_stop():
    global recording, csv_file, csv_writer, csv_filename, last_analysis
    with rec_lock:
        if not recording:
            return jsonify({"status": "not_recording"})
        recording = False
        if csv_file:
            csv_file.close()
            closed_filename = str(csv_filename)
            print("[REC] Stopped:", closed_filename)
            # langsung lakukan analisis sinkron di thread terpisah agar respons cepat
            analysis = analyze_ecg_file(closed_filename)
            last_analysis = analysis
            return jsonify({"status": "stopped", "filename": closed_filename, "analysis": analysis})
    return jsonify({"status": "ok"})

@app.route("/last_analysis")
def api_last():
    return jsonify({"analysis": last_analysis})

# -------------------------
# Analisis EKG sederhana (dengan RESP tambahan)
# -------------------------
def analyze_ecg_file(filename: str):
    """
    Baca CSV dan lakukan:
    - deteksi puncak (R-peaks) via scipy.find_peaks
    - hitung BPM = n_peaks * 60 / duration_seconds
    - hitung respiratory rate sederhana
    - klasifikasi: bradikardi (<60), takikardi (>100), normal (60-100)
    - kembalikan dict hasil (termasuk resp_rate)
    """
    print("[ANALYSIS] Loading:", filename)
    try:
        df = pd.read_csv(filename)
        # dukung file yang berisi atau tidak berisi header nama
        if "ecg" not in df.columns:
            # support jika header berbeda
            if df.shape[1] >= 4:
                df.columns = ["server_time", "esp_millis", "ecg", "resp"]
            elif df.shape[1] == 3:
                df.columns = ["server_time", "esp_millis", "ecg"]
                df["resp"] = 0.0
            else:
                return {"error": "file format unexpected"}

        # ambil array
        ecg = df["ecg"].astype(float).to_numpy()
        resp = df["resp"].astype(float).to_numpy()

        # ECG peak detection
        min_distance = int(0.4 * SAMPLING_RATE_HZ)  # 0.4s minimal antar R
        if min_distance < 1:
            min_distance = 1

        height_th = np.mean(ecg) + 0.5 * np.std(ecg)
        peaks, props = find_peaks(ecg, distance=min_distance, height=height_th)

        n_peaks = len(peaks)
        duration_seconds = len(ecg) / SAMPLING_RATE_HZ if SAMPLING_RATE_HZ > 0 else 1.0
        bpm = (n_peaks * 60.0) / duration_seconds if duration_seconds > 0 else 0.0

        # respiratory rate: deteksi puncak pada sinyal resp (lebih lambat)
        try:
            resp_min_dist = int(0.8 * SAMPLING_RATE_HZ)  # minimal 0.8s antar peak pernapasan
            if resp_min_dist < 1: resp_min_dist = 1
            resp_peaks, _ = find_peaks(resp, distance=resp_min_dist)
            n_resp_peaks = len(resp_peaks)
            resp_rate = (n_resp_peaks * 60.0) / duration_seconds if duration_seconds > 0 else 0.0
        except Exception:
            resp_peaks = []
            n_resp_peaks = 0
            resp_rate = 0.0

        if bpm < 60.0:
            diagnosis = "Bradikardi"
        elif bpm > 100.0:
            diagnosis = "Takikardi"
        else:
            diagnosis = "Normal"

        result = {
            "filename": filename,
            "n_samples": int(len(ecg)),
            "n_peaks": int(n_peaks),
            "duration_s": float(duration_seconds),
            "bpm": float(round(bpm,2)),
            "diagnosis": diagnosis,
            "peak_indices": peaks.tolist()[:100],  # beri sample indices (max 100)
            "n_resp_peaks": int(n_resp_peaks),
            "resp_rate": float(round(resp_rate,2))
        }
        print("[ANALYSIS] result:", result)
        return result
    except Exception as e:
        print("[ANALYSIS] Exception:", e)
        return {"error": str(e)}

# -------------------------
# (Optional) dummy generator for testing tanpa ESP32
# -------------------------
# async def ecg_dummy_generator():
#     print("[DUMMY] ECG+RESP simulator aktif...")
#     t0 = time.time()
#     while True:
#         t = time.time() - t0
#         ecg_val = 600 * math.sin(2 * math.pi * 1.2 * t) + (random.random()*40 - 20)
#         resp_val = 0.5 * math.sin(2 * math.pi * 0.25 * t) + (random.random()*0.05 - 0.025)
#         recv_time = time.time()
#         payload = {"recv_time": recv_time, "esp_millis": t*1000.0, "ecg": ecg_val, "resp": resp_val}
#         broadcast_message(json.dumps(payload))
#         with rec_lock:
#             global recording, csv_writer
#             if recording and csv_writer is not None:
#                 csv_writer.writerow([recv_time, t*1000.0, ecg_val, resp_val])
#         await asyncio.sleep(1.0/SAMPLING_RATE_HZ)

# -------------------------
# Main entry
# -------------------------
if __name__ == "__main__":
    # start WS server thread
    start_ws_server_in_thread()

    # Jika mau gunakan dummy (tanpa ESP32), uncomment bagian ini:
    # def run_dummy():
    #     asyncio.run(ecg_dummy_generator())
    # dummy_thread = threading.Thread(target=run_dummy, daemon=True)
    # dummy_thread.start()

    # jalankan Flask main thread
    print(f"[FLASK] Starting Flask on http://{FLASK_HOST}:{FLASK_PORT}")
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=False)