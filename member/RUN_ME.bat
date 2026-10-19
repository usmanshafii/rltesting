@echo off
title KAL M1 four-sensor job - laptop B (rendering and training)
echo.
echo  ==================================================================
echo   KAL M1 four-sensor job - laptop B (the member's laptop)
echo  ==================================================================
echo   * Keep this window open, the charger plugged in and the lid open.
echo   * If anything stops it (restart, crash, closed window), just
echo     double-click RUN_ME.bat again: it continues where it stopped.
echo   * It stops on purpose twice and says what to do:
echo       1. after rendering (3-4 h): send m1_part_B.zip to Usman, then
echo          wait for his m1_part_A.zip and double-click RUN_ME.bat again;
echo       2. at the end (about 3 h later): send the results zip to Usman.
echo   * First time? Run CHECK_SETUP.bat first (about 20 minutes).
echo.
call "%~dp0_job.bat" B %1
