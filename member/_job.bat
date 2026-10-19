@echo off
rem Shared launcher of the M1 job: _job.bat <solo|A|B> [--check]. Finds or builds the Python 3.11 environment (exact package
rem versions, so both laptops compute identical camera frames), then runs tools\member_job.py. Started by RUN_ME.bat
rem (laptop B), RUN_LAPTOP_A.bat (laptop A), CHECK_SETUP.bat and RUN_M1_PRELIMINARY.bat (one laptop); not
rem meant to be double-clicked itself.
setlocal
set ROLE=%~1
set EXTRA=%~2
if "%ROLE%"=="" (
  echo Start RUN_ME.bat or RUN_LAPTOP_A.bat instead of this file.
  pause
  exit /b 1
)
cd /d "%~dp0\.."

set "PY="
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -c "import torch, numpy, scipy, yaml, PIL, matplotlib" >nul 2>nul
  if not errorlevel 1 set "PY=.venv\Scripts\python.exe"
)
if not defined PY if not "%ROLE%"=="B" (
  rem laptop A (or the one-laptop run) may already have Python 3.11 with the pinned packages for the whole user
  py -3.11 -c "import torch, numpy, scipy, yaml, PIL, matplotlib" >nul 2>nul
  if not errorlevel 1 set "PY=py -3.11"
)
if defined PY goto run

where py >nul 2>nul
if errorlevel 1 goto nopython
py -3.11 -c "import sys" >nul 2>nul
if errorlevel 1 goto nopython
py -3.11 -c "import os, sys; sys.exit(1 if len(os.getcwd()) > 90 else 0)"
if errorlevel 1 goto longpath
if not exist ".venv\Scripts\python.exe" (
  echo Creating the Python environment in .venv ...
  py -3.11 -m venv .venv
  if errorlevel 1 goto fail
)
echo Installing packages - one time, about 10 minutes, about 1 GB download ...
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 goto pipfail
".venv\Scripts\python.exe" -m pip install -r member\requirements.txt
if errorlevel 1 goto pipfail
set "PY=.venv\Scripts\python.exe"

:run
%PY% -u tools\member_job.py --role %ROLE% %EXTRA%
if errorlevel 1 goto fail
echo.
pause
exit /b 0

:nopython
echo.
echo  Python 3.11 is not installed. Install it ONE of these ways, then double-click the same file again:
echo    1. Open PowerShell and run:   winget install -e --id Python.Python.3.11
echo    2. Or download "Windows installer (64-bit)" for Python 3.11 from python.org
echo       and tick "Add python.exe to PATH" during installation.
pause
exit /b 1

:longpath
echo.
echo  This folder path is too long: %CD%
echo  PyTorch has very long file names and Windows refuses paths over 260 characters.
echo  Move the fyp folder to a short path such as C:\fyp and run it from there.
pause
exit /b 1

:pipfail
echo.
echo  Installing the Python packages failed. Check the internet connection (and that no VPN or
echo  firewall blocks pypi.org and download.pytorch.org), then double-click the same file again.
pause
exit /b 1

:fail
echo.
echo  The job stopped with an error. Send member\job_log.txt to Usman.
pause
exit /b 1
