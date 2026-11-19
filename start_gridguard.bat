@echo off
cd /d "%~dp0"
start "GridGuard Backend" cmd /k python api.py
start "GridGuard Frontend" cmd /k npm run dev
timeout /t 3 /nobreak >nul
start http://localhost:8080/dashboard