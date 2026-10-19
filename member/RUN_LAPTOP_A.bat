@echo off
title KAL M1 four-sensor job - laptop A (rendering, single-camera models, evaluation)
echo.
echo  ==================================================================
echo   KAL M1 four-sensor job - laptop A (Usman's laptop)
echo  ==================================================================
echo   * Renders scenes 0-199 (3-4 h), packs them for laptop B, waits for
echo     m1_part_B.zip, trains lwir_only, lwir_cooled_only and mwir_only,
echo     waits for laptop B's results zip, then evaluates once.
echo   * Run it again whenever it stops to wait for a file, or after a
echo     restart: it continues where it stopped.
echo   * Setup check only: RUN_LAPTOP_A.bat --check
echo.
call "%~dp0_job.bat" A %1
