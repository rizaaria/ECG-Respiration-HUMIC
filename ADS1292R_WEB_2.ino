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
const char* ssid = "SSID";
const char* password = "PASSWORD";
const char* serverHost = "192.168.x.x";   // Local IP Address
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

File dataFile;

// ====================== Fungsi File Helper ======================
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

// ====================== WiFi & WebSocket Setup ======================
void setupWiFi() {
  WiFi.begin(ssid, password);
  Serial.print("Connecting WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi connected: " + WiFi.localIP().toString());
}

void setupWebSocket() {
  Serial.println("Connecting to WebSocket server...");
  webSocket.begin(serverHost, serverPort, "/");
  webSocket.enableHeartbeat(15000, 3000, 2);        // keepalive
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
      default:
        break;
    }
  });
  webSocket.setReconnectInterval(3000);
}

// ====================== Setup ======================
void setup() {
  Serial.begin(115200);
  delay(2000);

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
  Serial.println("ADS1292R initialized.");
}

// ====================== Loop ======================
void loop() {
  webSocket.loop();

  static uint32_t lastSampleTime = 0;
  const uint32_t sampleIntervalMs = 20;   // 50 Hz
  uint32_t nowMs = millis();
  if (nowMs - lastSampleTime < sampleIntervalMs) return;
  lastSampleTime = nowMs;

  ads1292OutputValues ecgRespirationValues;
  bool ret = ADS1292R.getAds1292EcgAndRespirationSamples(
                ADS1292_DRDY_PIN, ADS1292_CS_PIN, &ecgRespirationValues);

  if (!ret) return;

  ecgWaveBuff = (int16_t)(ecgRespirationValues.sDaqVals[1] >> 8);
  resWaveBuff = (int16_t)(ecgRespirationValues.sresultTempResp >> 8);

  if (!ecgRespirationValues.leadoffDetected) {
    ECG_RESPIRATION_ALGORITHM.ECG_ProcessCurrSample(&ecgWaveBuff, &ecgFilterout);
    ECG_RESPIRATION_ALGORITHM.QRS_Algorithm_Interface(ecgFilterout, &globalHeartRate);
    respFilterout = ECG_RESPIRATION_ALGORITHM.Resp_ProcessCurrSample(resWaveBuff);
    ECG_RESPIRATION_ALGORITHM.RESP_Algorithm_Interface(respFilterout, &globalRespirationRate);
  } else {
    ecgFilterout = 0;
    respFilterout = 0;
  }

  // --- tampil di serial (opsional) ---
  // Serial.printf("%d,%d\n", ecgFilterout, respFilterout);

  // --- simpan ke SD ---
  static int counter = 0;
  if (dataFile) {
    dataFile.printf("%d,%d\n", ecgFilterout, respFilterout);
    if (++counter >= 100) { dataFile.flush(); counter = 0; }
  }

  // --- kirim ke WebSocket server ---
  if (WiFi.status() == WL_CONNECTED && webSocket.isConnected()) {
    String msg = String(nowMs) + "," + String(ecgFilterout) + "," + String(respFilterout);
    webSocket.sendTXT(msg);
  }
}
