# sleepwear

Sleep staging from an Apple Watch: accelerometer and PPG heart rate in, sleep stages out,
scored against polysomnography (Walch et al. 2019, PhysioNet `sleep-accel`, 31 subjects,
one lab night each).

**Question:** with the paper's three features and subject-wise evaluation, does a small
transformer that sees neighbouring epochs beat the paper's per-epoch classifiers?

**Answer, on these 31 nights:**
- **Wake / NREM / REM: yes.** κ 0.380 against 0.299–0.321 for the paper's four classifiers.
  Per subject, it beats each of them in 24–27 of 31 nights, and every paired κ difference has
  a 95% CI above zero (macro-F1 gains too, though against logistic regression that CI includes 0).
- **Wake / sleep: not on κ or macro-F1** (a tie). It ranks epochs better: its per-subject AUC is
  higher than all four classifiers'.

## Results

Every metric is computed per subject from leave-one-subject-out predictions, then averaged
over the 31 subjects. Brackets are 95% bootstrap CIs over subjects (2,000 resamples).
Predictions are the argmax class, with no threshold tuned on test data.

Two columns are ranking measures, also computed per subject and never pooled across folds:
- AUC.
- Wake correct at 93% sleep correct (the paper's operating point). This uses each subject's
  own labels to set the threshold, so it is for comparison only.

The transformer row is the mean over 3 seeds. Source: `results/tables/main.csv`.

**Wake / Sleep (2-class)**

| model | kappa | macro_f1 | recall_W | recall_Sleep | auc | wake_at_0.93 |
|---|---|---|---|---|---|---|
| majority baseline | 0.000 [0.000, 0.000] | 0.475 [0.468, 0.480] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.500 [0.500, 0.500] | n/a |
| logistic regression | 0.336 [0.278, 0.394] | 0.647 [0.610, 0.683] | 0.787 [0.722, 0.844] | 0.809 [0.754, 0.850] | 0.879 [0.851, 0.906] | 0.667 [0.595, 0.738] |
| k-NN | 0.386 [0.331, 0.440] | 0.681 [0.650, 0.712] | 0.775 [0.710, 0.832] | 0.853 [0.818, 0.881] | 0.889 [0.860, 0.916] | 0.693 [0.622, 0.759] |
| random forest | 0.365 [0.310, 0.421] | 0.667 [0.635, 0.699] | 0.811 [0.747, 0.866] | 0.832 [0.806, 0.856] | 0.903 [0.875, 0.926] | 0.694 [0.627, 0.759] |
| MLP | 0.373 [0.316, 0.431] | 0.670 [0.636, 0.703] | 0.818 [0.755, 0.873] | 0.830 [0.800, 0.857] | 0.901 [0.874, 0.925] | 0.714 [0.650, 0.776] |
| transformer (mean of 3 seeds) | 0.374 [0.311, 0.436] | 0.665 [0.629, 0.703] | 0.820 [0.754, 0.880] | 0.815 [0.768, 0.855] | 0.917 [0.893, 0.939] | 0.736 [0.673, 0.793] |

**Wake / NREM / REM (3-class)**

| model | kappa | macro_f1 | recall_W | recall_NREM | recall_REM |
|---|---|---|---|---|---|
| majority baseline | 0.000 [0.000, 0.000] | 0.283 [0.270, 0.301] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] |
| logistic regression | 0.321 [0.282, 0.361] | 0.549 [0.522, 0.578] | 0.658 [0.587, 0.731] | 0.641 [0.598, 0.685] | 0.612 [0.522, 0.693] |
| k-NN | 0.308 [0.274, 0.343] | 0.539 [0.512, 0.568] | 0.737 [0.665, 0.802] | 0.522 [0.484, 0.558] | 0.738 [0.647, 0.816] |
| random forest | 0.299 [0.262, 0.340] | 0.533 [0.504, 0.565] | 0.769 [0.704, 0.827] | 0.543 [0.507, 0.579] | 0.660 [0.583, 0.730] |
| MLP | 0.307 [0.275, 0.340] | 0.531 [0.505, 0.559] | 0.778 [0.711, 0.835] | 0.522 [0.483, 0.560] | 0.705 [0.622, 0.778] |
| transformer (mean of 3 seeds) | 0.380 [0.337, 0.422] | 0.570 [0.540, 0.601] | 0.797 [0.725, 0.862] | 0.594 [0.550, 0.640] | 0.721 [0.621, 0.806] |

5132496 and 7749105 have no REM epochs. Their REM recall is undefined, so the REM column
averages 29 subjects.

**Transformer minus each paper model, paired by subject** (`results/tables/transformer_vs_paper.csv`):

| classes | metric | vs | diff [95% CI] | transformer better in |
|---|---|---|---|---|
| 2 | kappa | logistic regression | +0.037 [-0.000, +0.075] | 19/31 |
| 2 | kappa | k-NN | -0.012 [-0.037, +0.014] | 10/31 |
| 2 | kappa | random forest | +0.008 [-0.021, +0.037] | 16/31 |
| 2 | kappa | MLP | +0.001 [-0.031, +0.031] | 15/31 |
| 2 | macro_f1 | logistic regression | +0.018 [-0.004, +0.041] | 18/31 |
| 2 | macro_f1 | k-NN | -0.016 [-0.030, -0.001] | 10/31 |
| 2 | macro_f1 | random forest | -0.002 [-0.021, +0.017] | 13/31 |
| 2 | macro_f1 | MLP | -0.004 [-0.025, +0.014] | 15/31 |
| 2 | auc | logistic regression | +0.038 [+0.018, +0.060] | 24/31 |
| 2 | auc | k-NN | +0.028 [+0.010, +0.047] | 22/31 |
| 2 | auc | random forest | +0.014 [+0.001, +0.029] | 15/31 |
| 2 | auc | MLP | +0.016 [+0.002, +0.032] | 15/31 |
| 3 | kappa | logistic regression | +0.059 [+0.022, +0.096] | 24/31 |
| 3 | kappa | k-NN | +0.072 [+0.050, +0.092] | 27/31 |
| 3 | kappa | random forest | +0.081 [+0.043, +0.113] | 27/31 |
| 3 | kappa | MLP | +0.073 [+0.051, +0.094] | 27/31 |
| 3 | macro_f1 | logistic regression | +0.021 [-0.006, +0.046] | 21/31 |
| 3 | macro_f1 | k-NN | +0.031 [+0.012, +0.048] | 26/31 |
| 3 | macro_f1 | random forest | +0.037 [+0.011, +0.060] | 26/31 |
| 3 | macro_f1 | MLP | +0.039 [+0.021, +0.056] | 26/31 |

- **The 3-class gain isn't a seed effect.** Each seed alone scores κ 0.369–0.393, and the
  best paper model scores 0.321.
- **What it buys at 3 classes:**
  - NREM recall: 0.594, against 0.52–0.54 for k-NN, RF and the MLP.
  - Wake recall: 0.797, the highest of any model.
  - REM recall: 0.721, close to k-NN's 0.738.
  - The paper models trade NREM against REM. The transformer gives up less of either.
- **At 2 classes,** k-NN has the highest κ (0.386) and macro-F1, and the transformer is level
  with RF and the MLP.

**Figures** (`results/figures/`):
- `kappa_per_subject.png`: per-subject κ for every model, both tasks.
- `confusion_3class_best.png`: row-normalised 3-class confusion of the best model (the
  transformer), averaged over subjects and seeds.
- `hypnogram_3class_4018081.png`: scored vs MLP vs transformer (seed 0) for 4018081, the
  median subject by transformer κ among nights with an NREM→REM transition.
- `attention_3class_4018081.png`: last-layer attention, averaged over heads, in the
  128-epoch window around that night's first NREM→REM transition.
  - Attention isn't local to the transition. It concentrates on a few fixed epochs in the
    window.
  - It is a picture of one model on one window, not an explanation of the gain.

### Reproduction gate

Before the transformer was trained, the four classifiers were rerun under the paper's own
protocol: 50 random 70/30 subject splits, each split scored on its pooled test subjects.
Their sleep/wake AUC was then compared with the paper's Tables 2–5 (`results/tables/gate.csv`,
`gate_ordering.csv`).

| features | logistic regression | k-NN | random forest | MLP |
|---|---|---|---|---|
| motion | 0.841 (0.819) | 0.772 (0.803) | 0.828 (0.810) | 0.841 (0.815) |
| HR | 0.718 (0.743) | 0.677 (0.682) | 0.686 (0.708) | 0.716 (0.737) |
| motion + HR | 0.808 (0.830) | 0.814 (0.810) | 0.827 (0.816) | 0.830 (0.828) |
| motion + HR + clock | 0.853 (0.854) | 0.860 (0.868) | 0.877 (0.871) | 0.870 (0.878) |

Ours first, the paper's in brackets.

- **11 of the 12 gated rows** (motion, HR, motion + HR) are within ±0.03.
  - **The exception is k-NN on motion alone,** 0.031 low. Half of all activity values are
    exactly zero. Distance-weighted k-NN with 1,000 neighbours handles such ties badly. With
    a second feature, k-NN is within 0.005.
- **All 19 required orderings of feature sets hold:** HR is the weakest set and the clock set
  the strongest, for every classifier.
- **The clock rows** aren't gated, because the paper used a circadian model there and we use a
  cosine. They are still within 0.01.

The full write-up, built from the same tables, is `notebooks/report.ipynb`.

## Method

### Features

These are the paper's three features, following its reference code
(`github.com/ojwalch/sleep_classifiers`):

- **Activity:** counts from raw z-axis acceleration (resample to 50 Hz, 3–11 Hz band-pass,
  te Lindert-style binning per 15 s). They are then Gaussian-smoothed (σ = 50 s) over ±9
  epochs within the subject.
- **HR standard deviation:**
  1. Heart rate is interpolated to 1 Hz and filtered with a difference of Gaussians
     (σ 120 s and 600 s).
  2. It is divided by the night's 90th percentile of |filtered HR|.
  3. The feature is the standard deviation over (t − 285 s, t + 315 s).
- **Clock proxy:** −cos(2π(t − 5 h)/24 h), with t measured from PSG start.
  - **Deviation from the paper:** the paper's main tables use a circadian-clock model driven
    by step counts. Steps are missing for many nights, so we use the cosine that the
    reference code falls back to.

### Models

- **Shared by every model:**
  - Median imputation and standard scaling, fitted on the training subjects only.
  - The same leave-one-subject-out folds: 31 folds, each subject's night predicted by a model
    that never saw it.
- **Paper models:** `LogisticRegression` (liblinear, L1/L2 × C), `KNeighborsClassifier`
  (distance-weighted, k ∈ {500, 1000}), `RandomForestClassifier` (100 trees,
  max_depth ∈ {10, 50, 100}) and `MLPClassifier` (15-15-15, α ∈ {1e-5 … 0.1}).
  - Settings and grids are the reference code's.
- **Majority baseline:** predicts the training fold's most common class.
- **Transformer:** a linear projection of the 3 features, sinusoidal positions, 2 encoder
  layers with 4 heads, and a per-epoch classification head.
  - Trained on windows of W consecutive epochs (stride W/4). Whole nights are predicted with
    the same windows, averaging overlapping probabilities.
  - Loss is weighted cross-entropy with AdamW. Training stops early on 3 held-out training
    subjects.
  - Tuned: d_model {32, 64}, W {32, 64, 128}, dropout {0.1, 0.3}, learning rate {3e-4, 1e-3}.
  - Reported as the mean over 3 seeds.
- **Tuning, the same for every model:**
  - Inside each outer fold, on 3 grouped inner folds of the training subjects (the same folds
    for every model).
  - Selection criterion: class-balanced log loss on the held-out inner subjects.
  - Chosen settings per fold: `results/tables/tuning.csv`.
  - The transformer most often chose the largest window, W = 128 (39 of 62 folds across both
    tasks).
- **Class balance:**
  - LR and RF use balanced class weights.
  - The MLP uses balanced sample weights.
  - k-NN votes are divided by training class counts.
  - The transformer's loss is class-weighted.
  - This puts every model's argmax on the same footing.
  - **Deviations from the reference:**
    - The reference's MLP ignored its class weights.
    - The reference had no scaler.
    - The reference tuned the 2-class models on AUC, with ungrouped inner folds.

### Limitations

- **Small sample:** 31 nights, one per person, from one lab.
- **Offline scoring:** both the features and the transformer use epochs after the one being
  scored, so this is whole-night offline scoring, not real-time staging.
- **Unequal training data:** the transformer trains on 27 of the 30 training nights (3 are
  held out for early stopping), the paper models on all 30.
- **Grid edge:** the transformer's chosen window sits at the edge of its grid. Longer windows
  were not tried, so its result may be conservative.

### Next step

- **Context or transformer?** The comparison above changes two things at once: the
  transformer sees neighbouring epochs, and it is a different model. The next run gives the
  MLP the same context, as the features of the epochs around each one, with the same folds
  and inner-fold tuning.
  - If that MLP matches the transformer at 3 classes, the gain comes from context.
  - If it doesn't, the transformer adds something beyond context.

## Data and handling

Per-subject figures below come from `results/tables/subjects.csv`, written by `make data`.

### Source

- **Subjects:** 31, one in-lab polysomnography (PSG) night each, wearing an Apple Watch
  (Walch et al. 2019, *Sleep*, doi:10.1093/sleep/zsz180; PhysioNet `sleep-accel` 1.0.0).
- **Signals:** triaxial acceleration (g) and PPG heart rate (bpm) from the watch. PSG
  stages come in 30 s epochs.
- **Integrity:** all raw files are checked against PhysioNet's `SHA256SUMS.txt`.

### Epoch grid

- Every file's timestamps are seconds from PSG start. Label files start at 0 and step by
  exactly 30 s, so the grid is `t = 30k` and a sample at time s belongs to epoch
  `floor(s / 30)`.
- Epochs are kept only if they lie wholly inside the window where PSG, accelerometer and
  heart rate all have data. This trims the final epoch for 7 subjects.

### Epochs kept

- **Kept:** labeled epochs (codes 0–5) that contain at least one accelerometer sample.
  Unscored epochs (code −1, 438 in total) are dropped.
- **No-motion epochs dropped (1,240 in total), following the reference implementation.**
  Per subject:

  | subject | no-motion epochs dropped | epochs kept |
  |---|---|---|
  | 7749105 | 829 | 119 |
  | 1066528 | 124 | 828 |
  | 2598705 | 75 | 879 |
  | 1449548 | 67 | 886 |
  | 8530312 | 59 | 888 |
  | 1360686 | 41 | 925 |
  | 759667 | 21 | 453 |
  | 4018081 | 12 | 497 |
  | 8000685 | 7 | 950 |
  | 781756 | 2 | 979 |
  | 844359 | 2 | 896 |
  | 5498603 | 1 | 743 |
  | other 19 subjects | 0 | |

- **7749105:** the accelerometer stops about 67 minutes into the night and only resumes
  after PSG ends, so 119 epochs remain.
- **Total:** 25,548 of 27,204 in-window epochs are kept.

### Heart rate

- Heart rate is linearly interpolated to 1 s across gaps of up to 5 minutes. Longer gaps
  are left out of the HR feature rather than bridged.
- **This departs from the reference implementation**, which interpolates heart rate across
  gaps of any length. For 7749105 it would bridge a gap of about 25,000 s from its two
  endpoints, inventing most of a night's heart rate.

### Motion

- **Activity counts** follow the reference: z axis, linearly interpolated to 50 Hz,
  3–11 Hz band-pass. Interpolation bridges gaps between samples; 303 kept epochs in 15
  subjects have at least one second with no raw sample (`acc_gap_epochs` in
  `results/tables/subjects.csv`), most of them in 2598705 (118), 1066528 (72) and 1449548 (43).
- **5383425 and 8258170 record at 10 Hz,** so 11 Hz is above their 5 Hz Nyquist limit.
  As in the reference, their counts are still computed from the signal resampled to 50 Hz,
  which in effect measures a 3–5 Hz band (`counts_full_band` is false for them). Their
  count distribution is close to everyone else's (63% and 56% of epochs at exactly zero
  activity, against 51% overall).
- **Sampling rate** is recorded per subject in `results/tables/subjects.csv` as device
  metadata. It is not a model feature, and nor are per-epoch sample counts, which track it.

### Steps

- Not used. Step-count bins cover none of the kept epochs in 7 subjects and only part of
  them in 10 more.

### Stage labels

- **Code 4 appears in three subjects:** 190 epochs in 5383425, 83 in 8258170, 83 in 9618981.
  - The PhysioNet description lists only 0, 1, 2, 3 and 5.
  - The epochs behave like Rechtschaffen & Kales (R&K) stage 4: almost every transition
    into them comes from stage 3, 68–100 % of them fall in the first third of the night, and these subjects have
    unusually little stage 3.
  - These three nights therefore appear to be scored with R&K rules rather than AASM.
  - Code 4 counts as sleep (2-class) and NREM (3-class).
- **Class definitions:**
  - 2 classes: Wake / Sleep
  - 3 classes: Wake / NREM / REM
- **Stages absent from a subject's kept epochs:**
  - 5132496 has no REM.
  - 7749105 has no REM in the 119 epochs that remain.
  - For these subjects, REM recall is undefined, not zero, and their 3-class macro-F1 is the
    mean over W and NREM.

## Reproduce

```
uv sync
make data      # downloads PhysioNet sleep-accel into data/raw, writes epochs
make features
make gate      # reproduction gate, ~10 min on 24 cores
make train     # paper models (~8 min CPU) and transformer (~45 min, one GPU)
make eval      # results/tables/main.csv and figures
make test
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/report.ipynb  # write-up
```
