# Pilot 2 on a free cloud GPU

Section 6d of `docs/privileged_teacher_case.md` needs 20–40 seeds, which a cloud GPU trains in hours rather than days.
The laptop builds the inputs and keeps rendering the fresh test scenes; the cloud trains and evaluates; the laptop then
merges the results and runs the pre-registered decision. The code and settings are the laptop's; only the device
differs.

## What to upload

- **Now:** `out/m1/cloud/kal_pilot2_train.zip` (2.2 GB): the code, pilot 2's training crops, the validation and test
  frames, and the scenes' metadata. No checkpoints or results.
- **Later**, once the laptop has rendered and cached them: `out/m1/cloud/kal_pilot2_cohorts.zip`, the thermal-arm and
  fresh test frames. Build it with `python tools/cloud_pilot2.py package-cohorts`; I will tell you when.

## Kaggle (recommended: 30 GPU hours a week, runs in the background)

1. Create a free account at kaggle.com and verify your phone number; GPUs need it.
2. **Datasets → New Dataset:** upload `kal_pilot2_train.zip`, name it `kal-pilot2-train`, keep it private, then
   **Create**. Kaggle unpacks it.
3. **Code → New Notebook → File → Import Notebook:** upload `cloud/pilot2_cloud.ipynb`.
4. In the notebook, **Add Input → Datasets → Your Datasets →** `kal-pilot2-train`.
5. **Settings:** Accelerator **GPU T4 ×2** (or P100). Internet can stay off.
6. **Save Version → Save & Run All (Commit).** It runs in the background for up to 12 hours; you can close the browser.
7. When it has finished, open the version, go to **Output**, and download every `kal_pilot2_seed_*.zip` and the latest
   `kal_pilot2_results_*.zip`.
8. Put them in `out/m1/cloud/results/` on the laptop. To merge them yourself, run
   `python tools/cloud_pilot2.py merge out/m1/cloud/results/*.zip`.

**If a session stops before every seed is done:** add that version's output to the notebook as an input (**Add Input →
Your Work → Notebooks**) and commit again. Finished seeds are skipped.

**More seeds after the interim:** in cell 4, set `SEEDS` to the number in
`out/m1/privileged_ablation/pilot2_power.json` (for example `'101-130'`) and commit again, with the earlier output
added as an input.

**The cohorts:** upload `kal_pilot2_cohorts.zip` as a second dataset, add it as an input, and commit again. The trained
seeds are then evaluated on the new cohorts.

## Colab (alternative)

1. In Google Drive, make a folder `kal_pilot2` and upload `kal_pilot2_train.zip` into it (later also
   `kal_pilot2_cohorts.zip`).
2. Open `cloud/pilot2_cloud.ipynb` in Colab (**File → Upload notebook**). Choose **Runtime → Change runtime type →
   GPU**, then **Run all**, and allow Drive access.
3. Each finished seed is written to `MyDrive/kal_pilot2/` straight away. After a disconnect, run all again: finished
   seeds are skipped.

## What to expect

- The first model's log shows its speed in seconds per step (the laptop does 0.66).
- Three workers share the GPU, because the crop loading runs on the CPU.
- Twenty seeds of the primary arms, plus the one-extra-camera arms for fifteen of them, are about 160 models.
- On a T4 that should take several hours. The first session's log gives the real figure.
