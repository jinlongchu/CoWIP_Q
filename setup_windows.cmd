@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0work\setup_environment.ps1"
if errorlevel 1 (
  echo Setup failed. Please read the message above.
  pause
  exit /b 1
)
echo Setup complete. Launch the DES dashboard from the project folder.
pause
