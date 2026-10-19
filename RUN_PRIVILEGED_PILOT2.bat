@echo off
title KAL privileged-teacher pilot 2 (gated distillation, one teacher per sensor set, thermal arms, fresh scenes)
echo.
echo  ==================================================================
echo   Pilot 2 of the privileged-teacher test (docs\privileged_teacher_case.md, section 6c)
echo  ==================================================================
echo   * The laptop's part of pilot 2 (section 6d): the first three seeds and
echo     the interim evaluation, the two thermal arms (low heat, electric-like)
echo     of the 55 test scenes, and the fresh scenes 180-599, rendered and
echo     cached. The cloud GPU trains the decisive seeds (cloud\README.md).
echo   * About 15 hours. Finished work is skipped: after a restart, run this
echo     file again and it continues where it stopped.
echo     If a training run is already going, it waits for it to finish.
echo   * Keep the charger plugged in; the job keeps Windows awake.
echo   * Log: out\m1\privileged_pilot2_queue_log.txt
echo.
cd /d "%~dp0"
set PY="C:\Program Files\Python311\python.exe"
if not exist %PY% set PY=python
%PY% -u tools\privileged_ablation.py pilot2 --laptop >> out\m1\privileged_pilot2_queue_log.txt 2>&1
echo.
echo Finished (or stopped: see the log). Results: out\m1\privileged_ablation\pilot2_test55, thermal_low_heat,
echo thermal_electric_like and fresh (results.md in each).
pause
