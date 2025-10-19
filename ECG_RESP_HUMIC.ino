#include "protocentralAds1292r.h"
#include "ecgRespirationAlgo.h"
#include <FS.h>
#include <SD.h>
#include <SPI.h>
#include <WiFi.h>
#include <WebSocketsClient.h>

// ====================== Pin Konfigurasi ======================
#define PIN_MISO 19
#define PIN_MOSI 23
#define PIN_SCK  18
#define ADS1292_CS_PIN     15
#define ADS1292_DRDY_PIN   25
#define ADS1292_START_PIN  26
#define ADS1292_PWDN_PIN   27

#define SD_MOSI  13
#define SD_MISO  12
#define SD_SCK   14
#define SD_CS    5

// ====================== WiFi & Server ======================
const char* ssid = "SHIZUDELTA 0931";
const char* password = "rizaaria12.";
const char* serverHost = "192.168.137.1";   // IP PC tempat server.py
const uint16_t serverPort = 8765;

// ====================== Objek ======================
SPIClass hspi(HSPI);
ads1292r ADS1292R;
ecg_respiration_algorithm ECG_RESPIRATION_ALGORITHM;
WebSocketsClient webSocket;

volatile uint8_t globalHeartRate = 0;
volatile uint8_t globalRespirationRate = 0;
int16_t ecgWaveBuff, ecgFilterout;
int16_t resWaveBuff, respFilterout;
float resp_dc_est = 0;   // estimasi DC untuk RESP

File dataFile;

// ====================== Helper fungsi ======================
// --- Konversi 24-bit signed (ADS1292R) ke int16_t ---
static inline int16_t to_int16_from_24(int32_t raw24) {
  if (raw24 & 0x00800000) raw24 |= 0xFF000000; // sign-extend
  return (int16_t)(raw24 >> 8);                 // skala lunak
}

// --- Buat nama file log baru ---
String getNewFileName() {
  int fileIndex = 0;
  String fileName;
  do {
    fileName = "/ecg_data_" + String(fileIndex) + ".csv";
    if (!SD.exists(fileName)) break;
    fileIndex++;
  } while (fileIndex < 10000);
  return fileName;
}

// ====================== WiFi & WebSocket ======================
void setupWiFi() {
  WiFi.begin(ssid, password);
  Serial.print("Connecting WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\n✅ WiFi connected: " + WiFi.localIP().toString());
}

void setupWebSocket() {
  Serial.println("Connecting to WebSocket server...");
  webSocket.begin(serverHost, serverPort, "/");
  webSocket.enableHeartbeat(15000, 3000, 2);
  webSocket.onEvent([](WStype_t type, uint8_t *payload, size_t length) {
    switch (type) {
      case WStype_CONNECTED:
        Serial.println("[WS] Connected to server.");
        break;
      case WStype_TEXT:
        Serial.printf("[WS] Server msg: %s\n", payload);
        break;
      case WStype_DISCONNECTED:
        Serial.println("[WS] Disconnected, retrying...");
        break;
      case WStype_ERROR:
        Serial.println("[WS] Error detected!");
        break;
      default: break;
    }
  });
  webSocket.setReconnectInterval(3000);
}

// ====================== Setup ======================
void setup() {
  Serial.begin(115200);
  delay(500);

  setupWiFi();
  setupWebSocket();

  SPI.begin(PIN_SCK, PIN_MISO, PIN_MOSI);
  SPI.beginTransaction(SPISettings(1000000, MSBFIRST, SPI_MODE1));

  hspi.begin(SD_SCK, SD_MISO, SD_MOSI, SD_CS);
  if (!SD.begin(SD_CS, hspi, 1000000)) {
    Serial.println("⚠️ SD Card Mount Failed");
  } else {
    Serial.println("✅ SD Card Initialized (1 MHz)");
    String newFile = getNewFileName();
    Serial.print("Logging to: ");
    Serial.println(newFile);
    dataFile = SD.open(newFile, FILE_WRITE);
    if (dataFile) {
      dataFile.println("ECG,RESP");
      dataFile.flush();
    } else {
      Serial.println("Failed to open file");
    }
  }

  pinMode(ADS1292_DRDY_PIN, INPUT);
  pinMode(ADS1292_CS_PIN, OUTPUT);
  pinMode(ADS1292_START_PIN, OUTPUT);
  pinMode(ADS1292_PWDN_PIN, OUTPUT);

  digitalWrite(ADS1292_CS_PIN, HIGH);
  digitalWrite(ADS1292_PWDN_PIN, LOW);
  delay(10);
  digitalWrite(ADS1292_PWDN_PIN, HIGH);
  delay(10);
  digitalWrite(ADS1292_START_PIN, LOW);

  ADS1292R.ads1292Init(ADS1292_CS_PIN, ADS1292_PWDN_PIN, ADS1292_START_PIN);
  Serial.println("✅ ADS1292R initialized.");
}

// ====================== Loop utama ======================
void loop() {
  webSocket.loop();

  ads1292OutputValues vals;
  if (!ADS1292R.getAds1292EcgAndRespirationSamples(ADS1292_DRDY_PIN, ADS1292_CS_PIN, &vals))
    return;

  // --- konversi 24bit ke int16 ---
  int16_t raw_ecg = to_int16_from_24(vals.sDaqVals[1]);
  int16_t raw_resp = to_int16_from_24(vals.sresultTempResp);

  if (!vals.leadoffDetected) {
  // === ECG ===
  ECG_RESPIRATION_ALGORITHM.ECG_ProcessCurrSample(&raw_ecg, &ecgFilterout);
  ECG_RESPIRATION_ALGORITHM.QRS_Algorithm_Interface(ecgFilterout, &globalHeartRate);

  // === RESP ===
  // 1️⃣ Hapus offset DC (high-pass sederhana)
  resp_dc_est = 0.995 * resp_dc_est + 0.005 * raw_resp;   // integrator DC
  float resp_hp = raw_resp - resp_dc_est;

  // 2️⃣ Proses melalui algoritma library (band-pass 0.1–0.5 Hz)
  int16_t resp_proc = ECG_RESPIRATION_ALGORITHM.Resp_ProcessCurrSample((int16_t)resp_hp);
  ECG_RESPIRATION_ALGORITHM.RESP_Algorithm_Interface(resp_proc, &globalRespirationRate);

  // 3️⃣ Normalisasi amplitude agar tidak terlalu besar
  respFilterout = constrain(resp_proc / 8, -500, 500);
  }
  else {
    ecgFilterout = 0;
    respFilterout = 0;
  }

  // --- Output ke Serial Plotter ---
  // Format label agar tampil dua channel terpisah
  Serial.print("ECG ");
  Serial.print(ecgFilterout);
  Serial.print("\tRESP ");
  Serial.println(respFilterout);

  // --- Logging ke SD ---
  static int flushCtr = 0;
  if (dataFile) {
    dataFile.printf("%d,%d\n", ecgFilterout, respFilterout);
    if (++flushCtr >= 100) { dataFile.flush(); flushCtr = 0; }
  }

  // --- Kirim ke server WS ---
  if (WiFi.status() == WL_CONNECTED && webSocket.isConnected()) {
    String msg = String(millis()) + "," + String(ecgFilterout) + "," + String(respFilterout);
    webSocket.sendTXT(msg);
  }
}