//////////////////////////////////////////////////////////////////////////////////////////
// HUMIK3 - ESP32 ADS1292R + WiFi + SD + WebSocket
// -----------------------------------------------
// Berdasarkan contoh ProtoCentral + integrasi WebSocket & SD Logging
// Pinout: sesuai ESP32 DevKit (ADS1292R di HSPI, SD di VSPI)
// -----------------------------------------------
// by Riza & ChatGPT, 2025
//////////////////////////////////////////////////////////////////////////////////////////

#include "protocentralAds1292r.h"
#include "ecgRespirationAlgo.h"
#include <WiFi.h>
#include <WebSocketsClient.h>
#include <FS.h>
#include <SD.h>
#include <SPI.h>

// ====================== Konfigurasi Pin ======================
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
const char* ssid = "SSID_KAMU";
const char* password = "PASSWORD_KAMU";
const char* serverHost = "192.168.x.x";  // IP komputer server.py
const uint16_t serverPort = 8765;

// ====================== Objek ======================
ads1292r ADS1292R;
ecg_respiration_algorithm ECG_RESPIRATION_ALGORITHM;
SPIClass hspi(HSPI);
WebSocketsClient webSocket;

File dataFile;
volatile uint8_t globalHeartRate = 0;
volatile uint8_t globalRespirationRate = 0;
int16_t ecgWaveBuff, ecgFilterout;
int16_t resWaveBuff, respFilterout;

// ====================== File Helper ======================
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

// ====================== WiFi ======================
void setupWiFi() {
  WiFi.begin(ssid, password);
  Serial.print("Connecting WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\n✅ WiFi connected: " + WiFi.localIP().toString());
}

// ====================== WebSocket ======================
void setupWebSocket() {
  webSocket.begin(serverHost, serverPort, "/");
  webSocket.enableHeartbeat(15000, 3000, 2);
  webSocket.onEvent([](WStype_t type, uint8_t *payload, size_t length) {
    switch (type) {
      case WStype_CONNECTED: Serial.println("[WS] Connected."); break;
      case WStype_DISCONNECTED: Serial.println("[WS] Disconnected."); break;
      case WStype_ERROR: Serial.println("[WS] Error!"); break;
      default: break;
    }
  });
  webSocket.setReconnectInterval(3000);
}

// ====================== Setup ======================
void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("\n=== HUMIK3 START ===");

  setupWiFi();
  setupWebSocket();

  // ADS1292R
  SPI.begin(PIN_SCK, PIN_MISO, PIN_MOSI);
  SPI.beginTransaction(SPISettings(1000000, MSBFIRST, SPI_MODE1));
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
  Serial.println("✅ ADS1292R initialized");

  // SD Card
  hspi.begin(SD_SCK, SD_MISO, SD_MOSI, SD_CS);
  if (!SD.begin(SD_CS, hspi, 1000000)) {
    Serial.println("⚠️ SD Card mount failed");
  } else {
    Serial.println("✅ SD Card initialized (1 MHz)");
    String newFile = getNewFileName();
    dataFile = SD.open(newFile, FILE_WRITE);
    if (dataFile) {
      Serial.println("Logging to " + newFile);
      dataFile.println("ECG,RESP");
      dataFile.flush();
    }
  }
}

// ====================== Loop ======================
void loop() {
  webSocket.loop();

  ads1292OutputValues vals;
  bool ret = ADS1292R.getAds1292EcgAndRespirationSamples(
                ADS1292_DRDY_PIN, ADS1292_CS_PIN, &vals);
  if (!ret) return;

  // 24-bit → 16-bit (buang 8 LSB, tapi tetap bisa bentuk ECG bagus)
  ecgWaveBuff = (int16_t)(vals.sDaqVals[1] >> 8);
  resWaveBuff = (int16_t)(vals.sresultTempResp >> 8);

  if (!vals.leadoffDetected) {
    ECG_RESPIRATION_ALGORITHM.ECG_ProcessCurrSample(&ecgWaveBuff, &ecgFilterout);
    ECG_RESPIRATION_ALGORITHM.QRS_Algorithm_Interface(ecgFilterout, &globalHeartRate);
    respFilterout = ECG_RESPIRATION_ALGORITHM.Resp_ProcessCurrSample(resWaveBuff);
    ECG_RESPIRATION_ALGORITHM.RESP_Algorithm_Interface(respFilterout, &globalRespirationRate);
  } else {
    ecgFilterout = 0;
    respFilterout = 0;
  }

  // === Output ke Serial Plotter ===
  Serial.printf("%d,%d\n", ecgFilterout, respFilterout);

  // === Simpan ke SD ===
  static int flushCtr = 0;
  if (dataFile) {
    dataFile.printf("%d,%d\n", ecgFilterout, respFilterout);
    if (++flushCtr >= 100) { dataFile.flush(); flushCtr = 0; }
  }

  // === Kirim ke Server (WebSocket) ===
  if (WiFi.status() == WL_CONNECTED && webSocket.isConnected()) {
    String msg = String(millis()) + "," +
                 String(ecgFilterout) + "," +
                 String(respFilterout);
    webSocket.sendTXT(msg);
  }
}
