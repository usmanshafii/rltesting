# Proposal notes (M1-01)

Facts and references for the proposal the brief asks for. The team writes the proposal in its own words from these
notes (the brief forbids AI-generated project ideas; the idea is the team's, and these notes only collect the plan's
points). Log the use in `docs/ai_use/`.

## 1. The real-world problem and its significance
- Small fixed-wing drones approach ground sites mostly at night; a camera post must find and confirm them early.
- RGB is cheap and widespread but blind at night; uncooled LWIR works day and night but is noisy and less sensitive.
- Cooled LWIR (lower noise, same physics as uncooled) and MWIR (engine and exhaust heat) see better but are expensive
  and not always available at deployment.

## 2. Limits of existing approaches / what makes it hard
- Fixed search patterns (raster scans) spend time on empty sky; a narrow zoom sees far but covers little.
- Targets are 2-64 px; clouds produce persistent false detections (one cloudy scene set the strict threshold in 12 of
  18 evaluations in the repo's pilot: `docs/privileged_teacher_case.md`).
- No simulator exists for this decision problem (Theme D).

## 3. Why RL, not a fixed rule or a supervised model
- No labels exist for the best place to look; the value of a look arrives steps later (sequential, delayed reward).
- Trade-offs: search new sky against dwell; wide against narrow zoom; early confirmation against false confirmation.
- Baselines that make the case: random, raster scan, Bayesian search on the same evidence.

## 4. The agent's objective
- Confirm an incoming drone as early as possible with RGB + uncooled LWIR only, at a fixed false-confirmation rate.
- Research question: how much of the cooled LWIR's and MWIR's benefit, used only in training (privileged critic or
  teacher), survives when they are absent at deployment; and does cooled LWIR transfer better than MWIR?

## References to cite (all in the team's doc, Sources section)
- Pinto et al., RSS 2018 (asymmetric actor-critic); Baisero and Amato, AAMAS 2022 (unbiased asymmetric critic).
- Chen et al., CoRL 2019 (Learning by Cheating); Ross et al., AISTATS 2011 (DAgger).
- Zhong et al., ICML 2021 (privileged teacher for active tracking); Hu et al., ICLR 2024 (Scaffolder).
- Page 1954 and Lorden 1971 (the sequential confirmation test).
