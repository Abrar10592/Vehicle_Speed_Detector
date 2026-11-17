@echo off
rem Starts the Vehicle Speed Detector and keeps it running (see server.py).
cd /d "%~dp0"
title Vehicle Speed Detector
".venv\Scripts\python.exe" server.py
pause
