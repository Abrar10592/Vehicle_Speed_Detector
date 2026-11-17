@echo off
rem Updates the Vehicle Speed Detector without the app: for when the app does not open (see update.py).
cd /d "%~dp0"
title Vehicle Speed Detector - Update
if not exist ".venv\Scripts\python.exe" (echo Setting up Python for this program... & py -3.14 -m venv .venv)
rem One line on purpose: the update may replace this file while it runs, and a line is read whole before it starts.
".venv\Scripts\python.exe" update.py --apply --repair && (echo. & echo Done. Start the app with Start.bat.) || (echo. & echo The update did NOT succeed. Read the lines above.) & pause
