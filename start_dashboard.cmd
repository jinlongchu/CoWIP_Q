@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0work\start_des_dashboard.ps1"
if errorlevel 1 pause
