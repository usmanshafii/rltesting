# M1 four-sensor job: what the second laptop runs (optional)

Since 2026-10-17 the preliminary evaluation runs on one laptop (`RUN_M1_PRELIMINARY.bat` in the repository root).
This folder is kept for a later two-laptop split (for example the full detector).

This folder lets laptop B (the member's Vivobook) render its share of the M1 scenes and train its four models
**without Claude**. Everything is automatic: you double-click a file, send two files to Usman, and put one file from
Usman in your Downloads folder.

| | |
|---|---|
| Time | about 3-4 h of rendering, then about 3 h of training (the PC must stay on; you can sleep meanwhile) |
| Disk | at least 30 GB free on the drive with the `fyp` folder |
| Computer | Windows 10/11. Everything runs on the CPU: the Radeon graphics are not used (PyTorch on Windows does not support that chip), which is expected |
| Internet | for the setup (about 1.4 GB: Python packages and Blender) and for exchanging the files with Usman |

## 1. One-time setup

1. **Python 3.11** (exactly 3.11; newer versions do not work with this PyTorch). In PowerShell:

       winget install -e --id Python.Python.3.11

   Or download the "Windows installer (64-bit)" for Python 3.11 from python.org and tick **Add python.exe to PATH**.
2. **The code.** Usman sends you `kal_code_<date>.zip`. Right-click it, choose **Extract All...**, and extract it to
   `C:\` so that you get the folder **`C:\fyp`**. Use this short path: PyTorch has very long file names and Windows
   refuses paths over 260 characters, so do not use OneDrive or deep Documents folders.
3. **Blender** is downloaded by the job itself (Blender 4.2.23 LTS, checked against its official checksum). You do not
   need to install it.

## 2. Check the setup (about 20 minutes)

Double-click **`C:\fyp\member\CHECK_SETUP.bat`**. It:
- installs the exact package versions into `C:\fyp\.venv`;
- downloads and checks Blender;
- renders 2 test scenes at low quality and runs a short training test.

It must end with **`CHECK PASSED`**. If it does not, send `C:\fyp\member\job_log.txt` to Usman.

## 3. Run the job

Double-click **`C:\fyp\member\RUN_ME.bat`** and leave the window open. Plug in the charger, keep the lid open and set
Windows' power mode to **Best performance**. The job keeps Windows awake by itself.

**Part 1: rendering (about 3-4 hours).** It renders scenes 200-599 and prints the time left. When it finishes it shows

    SEND THIS FILE TO USMAN:  C:\fyp\out\m1\m1_part_B.zip      (about 1.8 GB)

Upload that file to Google Drive (or OneDrive) and send Usman the link. The window then says

    WAITING for Usman's file  m1_part_A.zip

**Part 2: training (about 3 hours).** When Usman sends you `m1_part_A.zip`, download it into your **Downloads**
folder and double-click `RUN_ME.bat` again. It checks every file of Usman's scenes, builds the training frames and
trains four models (`rgb_lwir`, `teacher`, `student`, `student_ctrl`). When the window says **ALL DONE**, send Usman:
- `C:\fyp\out\m1\det2_results_B_<your-PC>_<date>.zip` (a few MB);
- `C:\fyp\member\job_log.txt`.

**If anything stops it** (a restart, Windows update, crash or a closed window), double-click `RUN_ME.bat` again.
Finished work is skipped; rendering resumes at the next unfinished scene and training from its last saved point
(every 250 steps).

## Please do not

- Edit any file in `C:\fyp` while the job runs.
- Run games, video editing or other heavy programs at the same time: they slow it down a lot.
- Delete `C:\fyp\out\m1`. It holds the scenes, the training frames and the runs the job resumes from.

## If something goes wrong

| Message | What to do |
|---|---|
| `Python 3.11 is not installed` | Do setup step 1, then double-click again |
| `This folder path is too long` | Move the `fyp` folder to `C:\fyp` |
| `Could not download Blender` | Check the internet connection, then run again (the download resumes) |
| `Not enough free disk space` | Free 30 GB, then run again |
| `WAITING for Usman's file` | Normal: put `m1_part_A.zip` in Downloads when it arrives and run again |
| `damaged` / `download it again` | The zip was cut off: download it again |
| Any other error | Send `C:\fyp\member\job_log.txt` to Usman |

## For Usman (laptop A)

1. Make the code bundle for laptop B: `python tools/m1_det2.py export-code` (writes `out/m1/kal_code_<date>.zip`; it
   holds every file a commit would, including `docs/sources` data, so laptop B needs no git) and send it.
2. Run **`member\RUN_LAPTOP_A.bat`** (or `python tools/member_job.py --role A`). It renders scenes 0-199, writes
   `out\m1\m1_part_A.zip` to send to laptop B, waits for `m1_part_B.zip`, trains `lwir_only`, `lwir_cooled_only` and
   `mwir_only`, waits for laptop B's results zip, imports it and evaluates once: `out\m1\det2\results.md`.

The plan, the timings and what each model shows are in `docs/m1_v2.md`.
