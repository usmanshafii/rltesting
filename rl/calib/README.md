# Calibration and sensor model (M2 / WP2)

Implemented on branch ibad-m2 from the course plan and team guide. The shared
contract, frozen configuration, plan, M1 environment and source census are unchanged.
WP2 is **not ready for sign-off**: see REPORT.md and CHANGE_REQUESTS.md.

## Reproduce

From the repository root with Python 3.11 and rl/requirements-rl.txt installed:

~~~text
python -m rl.calib.calibrate
python -m pytest tests/rl -q
python -m rl.scripts.check wp2
~~~

The build regenerates 126 non-test scene specs and proves the join at <0.1% range
and unblurred-box size error. It takes a few minutes because the repository sampler
also computes mesh visibility. It writes joined_census.json, join_sources.json and
params.json. Reuse an unchanged verified join with:

~~~text
python -m rl.calib.calibrate --reuse-join
~~~

That option refuses changed source hashes. Source hashes normalize CRLF to LF.
The WP2 command writes rl/results/checks/wp2.json and returns **nonzero** while any
required evidence or integration is incomplete. It never treats PENDING as a pass.

## Use the handoff

~~~python
from rl.calib import CalibratedSensorModel
from rl.env import SkyEnv

env = SkyEnv(sensor_model=CalibratedSensorModel())
obs, info = env.reset(seed=11)
~~~

The default SkyEnv remains M1's stub pending M1-08 review. The calibrated object
implements every existing protocol method and is automatically covered by the
contract tests. Student observations remain unchanged when privileged sensing is
turned off. Low-heat episodes raise a clear error because no low-heat evidence is
available; they never inherit an invented reduction factor.

## Statistical model

Training only: 107 census scenes; validation only: 19. Positive/seen d-prime values
are fitted in log space by band using log size, day/night, their size interaction,
log visibility, cloud, aspect groups and power setting. Five folds are assigned by
scene, shared by every band. Paired thermal-to-LWIR median correction is fitted
only to jointly seen training rows. Coverage zeros are never fitted as log signal.
The 60/120-degree aspect boundaries are fixed grade-C grouping decisions, not measured sensor thresholds.
The paired residual bank retains cross-camera dependence and per-scene noise shares.
Validation values cannot change fitted coefficients or residual spread.

A seeded episode bootstrap supplies persistent nuisance state and joint residuals
from training scenes matching day and cloud. Missing per-band residuals are sampled
from available matched training residuals. This is a grade-C missing-data assumption,
not a complete physical flight/aspect model. Subpixel area scaling follows the
existing stub assumption. The model retains census uncertainty; it is not a trained CNN.

score_model.py provides Gaussian scores reproducing the frozen historical logistic,
an explicit null threshold, and exact whole-frame false-alarm partitioning.
clutter.py separates measured clutter/sensor spread, provides persistent shared LWIR
clutter with fresh independent camera noise, and imports the repository d_after
formula verbatim by delegation. SkyEnv v0 does not yet consume these score/clutter
samplers or dwell integration; see CR-M2-01.

## Recover missing pixel checks

pixels.py provides a local SceneDataset adapter whose --no-plume flag removes only
the object named plume from every thermal band. It leaves the source loader and
source pixels untouched; tests check all four sensors. It uses the exact M1 noise
salts, preprocessing, coverage and detectability. Once complete four-band scenes and
the pinned PyTorch runtime are present:

~~~text
python -m rl.calib.pixels --dataset out/m1/dataset --verify-five --output rl/calib/five_scene_check.json
python -m rl.calib.pixels --dataset out/m1/dataset --output rl/calib/pixel_census.json
python -m rl.calib.pixels --dataset out/m1/dataset --no-plume --output rl/calib/noplume_census.json
python tools/m1_det2.py render --start 0 --count 180
python tools/m1_det2.py render --start 0 --count 180 --overlay configs/m1_low_heat.yaml --source out/m1/dataset_low_heat
python -m rl.calib.pixels --dataset out/m1/dataset_low_heat --overlay configs/m1_low_heat.yaml --output rl/calib/low_heat_census.json
~~~

Rendering requires the repository's Blender 4.2 installation. The recovery path was
unit-tested for layer filtering; actual pixel reproduction could not run without the
missing scenes. Do not promote its output to params.json without validating the
paired measurements and reviewing the model/version change.

## Evidence and limitations

params.json records normalized SHA-256 hashes of the original census, detector data,
configs, sampler, geometry, climate and physical derivations. Each named calibration
value has a source and evidence grade. Grade B is derived synthetic-census evidence;
it does not claim an instrument measurement. Grade C marks the analytic plume bound,
cloud false-alarm factors, cross-band efficiency assumption and other surrogates.

The original plume/stub ratio .1-.7 is not a plume/whole-target fraction. The fallback
uses the conservative removal d/(1+r), applies it only to MWIR shared signal, and
does not claim to have reproduced the missing plume-free census. No low-heat result
or privileged-transfer benefit is fabricated.
