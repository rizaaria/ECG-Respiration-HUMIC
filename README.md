# Smart-ECG-AI
Real-time ECG &amp; respiration monitoring using ADS1922R with ESP32, Flask, SD Card Logger, and multi-model AI heartbeat analysis (MAE, BERT, Conformer).

Smart-ECG-AI
A real-time ECG and respiration monitoring system using ESP32 and Flask, integrated with multiple deep learning models (MAE, BERT, Conformer) for heartbeat classification and diagnosis visualization.

This project allows users to stream ECG and respiration signals wirelessly from an ESP32 board to a web dashboard via WebSocket, visualize data in real time, record sessions, and automatically analyze the recorded ECG data using selected AI models.

Features

Real-time ECG & respiration signal visualization (50 Hz sampling rate)

Data acquisition via ESP32 + ADS1292R

WebSocket communication and Flask backend

Model selection dropdown (MAE, BERT, Conformer, Fold4)

Automatic ECG beat classification and diagnosis summary

Downloadable CSV recordings and model outputs

Clean light-mode web dashboard built with Bootstrap 5 & Chart.js
