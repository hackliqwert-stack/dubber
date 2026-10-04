@echo off
chcp 65001 >nul
title Dubber (не закрывайте это окно, пока смотрите видео)
cd /d "%~dp0server"
if not exist ".venv\Scripts\python.exe" (
  echo Dubber ещё не установлен. Запустите install.bat
  pause
  exit /b 1
)
set PYTHONUTF8=1
".venv\Scripts\python.exe" -m app.main
echo.
echo Dubber остановлен.
pause
