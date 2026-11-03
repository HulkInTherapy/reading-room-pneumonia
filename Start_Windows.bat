@echo off
cd /d "%~dp0"
py -3.13 --version >nul 2>&1
if not errorlevel 1 (
  py -3.13 setup_demo.py
) else (
  python setup_demo.py
)
if errorlevel 1 (
  echo.
  echo If Python is missing, install Python 3.13 from https://www.python.org/downloads/
  pause
)
