@echo off
title KAL detection console
cd /d "%~dp0"
rem Uses the environment that member\RUN_ME.bat (or CHECK_SETUP.bat) creates in .venv; otherwise the system Python.
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -m app %*
) else (
  py -3.11 -m app %*
)
if errorlevel 1 pause
