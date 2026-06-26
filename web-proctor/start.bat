@echo off
set "VITE=%~dp0node_modules\vite\bin\vite.js"
echo Starting Proctor Lite Web App...
echo Open http://localhost:5173 in Chrome
echo Press Ctrl+C to stop
node "%VITE%" --host
pause
