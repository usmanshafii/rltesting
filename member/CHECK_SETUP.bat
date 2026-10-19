@echo off
rem Setup check for laptop B (about 20 minutes the first time): installs the packages, downloads and verifies Blender
rem 4.2.23, renders 2 scenes at low quality and runs a short training test. It must end with "CHECK PASSED".
title KAL M1 - setup check (laptop B)
call "%~dp0_job.bat" B --check
