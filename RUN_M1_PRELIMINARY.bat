@echo off
title KAL M1 preliminary evaluation - one laptop (render, cache, train seven models, evaluate)
echo.
echo  ==================================================================
echo   KAL M1 preliminary evaluation - everything on this laptop
echo  ==================================================================
echo   * Renders the 600 scenes (about 10 h), builds the camera frames
echo     (about 4 h), trains the seven models (about 4 h) and evaluates
echo     once (about 1 h). Finished work is skipped, so after a restart
echo     just run this file again: it continues where it stopped.
echo   * Keep the charger plugged in; the job keeps Windows awake.
echo   * Results: out\m1\det2\results.md (examples in out\m1\det2\gallery)
echo   * Setup check only (about 10 minutes): RUN_M1_PRELIMINARY.bat --check
echo.
call "%~dp0member\_job.bat" solo %1
