@echo off
chcp 65001 >nul
title Dubber - installation
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\install.ps1"
echo.
pause
