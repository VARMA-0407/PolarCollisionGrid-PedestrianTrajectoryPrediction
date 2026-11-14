# Master Project File

## Real-Time AI-Based Pedestrian Safety Prediction

**Project title:** Real-Time AI-Based Pedestrian Safety Prediction:
Interaction-Aware Multimodal Trajectory and Conflict Forecasting from
Video\
**Last updated:** 11 November 2026 (updated with Phase 2 Tasks 1–7 findings and Tasks 8–9 execution handoff)\
**Purpose:** Single source of truth for project architecture, technical
decisions, implementation, completed work, errors, experiments, status,
and next steps.

------------------------------------------------------------------------

## 1. Project objective

Develop a real-time video-based pedestrian safety system that detects
and tracks pedestrians and vehicles, models their interactions, predicts
multiple possible future trajectories, estimates conflict risk, and
produces calibrated early warnings.

**Intended input:** Roadside-camera or dashcam video, targeting
approximately 25--30 FPS.

**Intended outputs:** - Detected and tracked agents with persistent
IDs. - Historical trajectories and kinematic states. - Multiple possible
future trajectories. - Pedestrian-pedestrian (P-P) and
pedestrian-vehicle (P-V) conflict estimates. - TTC, PET, minimum
predicted separation, conflict probability, risk category, and
early-warning time. - Alerts for an ITS dashboard, alert unit, or
autonomous-driving controller.

The final system is intended to be more than an object detector or a
trajectory predictor: it combines perception, interaction-aware
forecasting, conflict reasoning, and calibrated safety outputs.

------------------------------------------------------------------------

## 2. Target architecture

The architecture below is the planned final system. Only the PCG
baseline work described in Section 5 has been implemented so far.

### Stage 1 --- Perception frontend

1.  Video ingestion at approximately 25--30 FPS.
2.  YOLO object detection.
3.  Classes: pedestrian, car, truck, bus, and bike.
4.  ByteTrack multi-object tracking.
5.  Persistent track IDs and bounding boxes.

**Output:** Per-frame detections and tracked identities.\
**Status:** Not implemented yet.

### Stage 2 --- Geometric and kinematic preprocessing

1.  Camera calibration / road-plane homography.
2.  Pixel-to-road-coordinate projection.
3.  Coordinate-unit verification.
4.  Trajectory smoothing.
5.  Velocity, acceleration, and heading estimation.
6.  Handling of missing/noisy/short tracks.

**Output:** Time-indexed agent states in a consistent road coordinate
system.\
**Status:** Not implemented in the proposed video pipeline. The PCG
baseline's own data processing still needs source-code inspection.

**Open issue:** HBS coordinate scale and physical units have not been
verified. Do not label ADE/FDE as metres until this is established.

### Stage 3 --- Interaction-aware graph and temporal encoder

-   Heterogeneous pedestrian and vehicle graph.
-   P-P, V-V, and P-V edges.
-   Heterogeneous GATv2 spatial interaction encoder.
-   Dilated Temporal Convolutional Network (TCN) for temporal encoding.

**Output:** Context-aware agent embeddings.\
**Status:** Proposed only; not implemented.

### Stage 4 --- Multimodal forecasting and conflict prediction

-   Multiple future trajectory hypotheses.
-   Mode probabilities and trajectory uncertainty.
-   Conflict reasoning for P-P and P-V pairs.
-   TTC, PET, and minimum predicted separation.
-   Conflict prediction head.

**Output:** Multiple possible futures and conflict-related
measurements.\
**Status:** Not implemented in the proposed architecture. PCG currently
serves as the trajectory-prediction baseline.

### Stage 5 --- Calibration and safety alert dispatcher

-   Probability calibration; temperature scaling is an initial
    candidate.
-   Expected Calibration Error (ECE).
-   Risk categories / alert levels.
-   Early-warning-time estimation.
-   Dispatch to ITS dashboard, alert unit, or autonomous-driving
    controller.

**Status:** Planned; not implemented.

------------------------------------------------------------------------

## 3. Baseline selection and decisions

### Selected baseline: Polar Collision Grid (PCG)

Repository:
https://github.com/Golchoubian/PolarCollisionGrid-PedestrianTrajectoryPrediction

PCG was selected as the initial trajectory-prediction baseline before
implementing the proposed interaction-aware architecture.

Other repositories discussed as references: -
https://github.com/kaktusracing/pedestrian_trajectories -
https://github.com/StanfordASL/Trajectron-plus-plus -
https://github.com/abduallahmohamed/Social-STGCNN

These are references considered during planning; the experiment
completed in this workstream is PCG.

### PCG configuration used for the 5-epoch run

  Parameter                              Value
  ----------------------- --------------------
  input_size                                 2
  output_size                                5
  rnn_size                                 128
  batch_size                                10
  seq_length                                12
  pred_length                                6
  obs_length                                 6
  num_epochs                                 5
  learning_rate                          0.001
  dropout                                  0.5
  embedding_size                            64
  embedding_size_action                     32
  neighborhood_size                          8
  grid_size                                  4
  maxNumPeds                                27
  lambda_param                          0.0005
  store_grid                              True
  num_sector                                 8
  method                    4 (Collision Grid)

Other observed values: `grad_clip=10.0`, `decay_rate=0.95`, `gru=False`,
`drive=False`, `num_validation=2`, `freq_validation=1`,
`freq_optimizer=8`, `neighborhood_size_veh_in_ped=64`,
`grid_size_veh_in_ped=4`, `TTC=[9]`, `D_min=0.7`, `TTC_veh=[8]`, and
`D_min_veh=1.0`.

**Configuration caveat:** An older `CollisionGrid/config.pkl` contained
`num_epochs=200` and `neighborhood_size=32`. The new run's root config
contained `num_epochs=5` and `neighborhood_size=8`. The test directory
was staged with the new config and epoch-4 checkpoint before evaluation.
The original config was separately backed up.

### Why this is an initial baseline, not a final result

-   Only 5 epochs were trained, not the repository's default 200.
-   The test script selected the best of 5 sampled trajectories
    according to ADE.
-   The reported values are best-of-five, not single-sample metrics.
-   HBS coordinate scale and physical units are not verified.
-   PCG is a trajectory-prediction baseline, not the final calibrated
    safety-prediction system.

Do not call this state-of-the-art, production-ready, or validated
real-world road-safety performance.

------------------------------------------------------------------------

## 4. Dataset and preprocessing history

The PCG repository uses the HBS dataset. Files referenced: -
`Data/train/trajectories_train.cpkl` -
`Data/test/trajectories_test.cpkl` - `hbs.csv`

During training: - 98 sequences with no pedestrian throughout were
filtered. - 2,386 sequences remained. - 238 batches per epoch. - Grid
calculation was reported to take approximately 30.57 seconds.

Dataset provenance, license, frame rate, coordinate convention, and
physical scale still need to be verified and documented. No final
dataset selection for the complete video-based pedestrian-and-vehicle
system has been recorded.

------------------------------------------------------------------------

## 5. Implementation log --- completed work

### Repository and compute setup

-   PCG repository cloned in Google Colab.
-   PCG repository also cloned locally on the Windows PC.
-   Colab GPU used for the experiment: NVIDIA Tesla T4.
-   Local VS Code Git status, active branch, and remote output have not
    yet been shared.
-   A local development branch named `pedestrian-safety-development` was
    recommended, but creation has not been confirmed.

### Training

Command:

``` bash
python train.py --method 4 --num_epochs 5
```

Observed result: - 5 epochs completed. - Total training time
approximately 748.46 seconds (about 12.47 minutes). - Epoch 0--4
checkpoints created, each approximately 1.28 MB. - Training loss became
negative, approximately -2 to -2.5 in later epochs.

Negative loss can occur with probabilistic likelihood objectives. It has
not been diagnosed as an error; inspect the loss definition and training
code before drawing a conclusion.

### Checkpoints/configuration backed up to Google Drive

Folder: `MyDrive/PCG_Experiments/PCG_T4_5_Epoch_Run`

Files: - `SOCIALLSTM_lstm_model_0.tar` - `SOCIALLSTM_lstm_model_1.tar` -
`SOCIALLSTM_lstm_model_2.tar` - `SOCIALLSTM_lstm_model_3.tar` -
`SOCIALLSTM_lstm_model_4.tar` - `config.pkl` -
`CollisionGrid_original_config.pkl` - `SOCIALLSTM_lstm_model_181.tar`

Reported sizes: - Epoch 0--4 checkpoints: approximately 1,315.46 KB
each. - Epoch 181 checkpoint: approximately 1,310.40 KB. - Each config:
approximately 0.62 KB.

### Test setup and evaluation

Before testing: - New root config copied to
`Store_Results/model/CollisionGrid/config.pkl`. - Epoch-4 checkpoint
copied to
`Store_Results/model/CollisionGrid/SOCIALLSTM_lstm_model_4.tar`. - Old
epoch-181 checkpoint preserved. - Active test config confirmed as
`num_epochs=5`, `neighborhood_size=8`.

Command:

``` bash
python test.py --method 4 --epoch 4 --sample_size 5
```

Test completed successfully on 1,081 HBS test sequences.

### Reported metrics

  Metric                                           Result
  --------------------------------- ---------------------
  ADE                                  0.5587965606209528
  FDE                                  1.1381949186325073
  MHD                                   3.071232318878174
  Speed error                         0.37326878044215545
  Heading error                         40.81245962143868
  Overall collision percentage                      2.17%
  Pedestrian-pedestrian collision                   5.91%
  Pedestrian-vehicle collision                      0.21%

**Interpretation cautions** 1. These are best-of-five results because
the test script selects the best of 5 sampled trajectories based on ADE.
2. HBS coordinate scale is not verified. Do not call ADE/FDE metres. 3.
Heading error's units and aggregation have not been checked in
`get_velocity_errors` and related code. Do not assume degrees or
radians. 4. Collision percentages are collision counts under configured
distance thresholds; they are not calibrated conflict probabilities or
proof of real-world safety. 5. The test set has already been evaluated
for epoch 4. Do not repeatedly select hyperparameters/epochs using test
results. Use validation data for model selection.

### Generated test artifacts

The test output directory contained:

``` text
Store_Results/plot/test/test_results.pkl
Store_Results/plot/test/SocialLSTM/test_results.pkl
Store_Results/plot/test/VLSTM/test_results.pkl
Store_Results/plot/test/LR/test_results.pkl
Store_Results/plot/test/CollisionGrid/test_results.pkl
Store_Results/plot/test/CollisionGrid/PV-CollisionGrid/test_results.pkl
Store_Results/plot/test/CollisionGrid/P-CollisionGrid/test_results.pkl
Store_Results/plot/test/CollisionGrid/V-CollisionGrid/test_results.pkl
```

Reported file sizes: - Main `test_results.pkl`: approximately 33,724.38
KB. - SocialLSTM: approximately 34,178.06 KB. - VLSTM: approximately
8,314.32 KB. - LR: approximately 6,788.91 KB. - CollisionGrid and
variants: approximately 33,724 KB each.

The main results file was loaded: - Python type: `list`. - Number of
entries: `1081`. - Each entry is a tuple of 12 elements.

First tuple element types/shapes: \| Tuple index \| Observed type /
shape \| \|---\|---\| \| 0 \| array `(12, 3, 9)` \| \| 1 \| array
`(12, 3, 9)` \| \| 2 \| list \| \| 3 \| dict \| \| 4 \| `NoneType` \| \|
5 \| int \| \| 6 \| array `(12, 3, 5)` \| \| 7 \| array `(12, 1, 9)` \|
\| 8 \| list \| \| 9 \| dict \| \| 10 \| list \| \| 11 \| list \|

The semantic meaning of all 12 tuple elements has not yet been mapped to
the source code. Do not infer meanings from shapes alone.

The main result file was copied to Drive as `test_results_epoch4.pkl`;
backup verification printed `Test results backed up: True`.

------------------------------------------------------------------------

## 6. Errors, issues, and resolutions

  -----------------------------------------------------------------------
  Issue                   Observation             Resolution / status
  ----------------------- ----------------------- -----------------------
  Malformed training      A command was           Correct training
  command                 accidentally            command subsequently
                          concatenated with       ran successfully.
                          `%cd`, producing an     
                          invalid integer         
                          argument such as        
                          `5%cd`.                 

  Old/new config mismatch Original CollisionGrid  Test folder staged with
                          config had 200 epochs   new config/checkpoint;
                          and neighborhood size   original config backed
                          32; new run had 5       up separately.
                          epochs and neighborhood 
                          size 8.                 

  Grid memory demand      Repository comment      T4 run completed;
                          estimated               monitor memory in
                          `store_grid=True` could larger runs.
                          require around 12 GB    
                          RAM.                    

  Negative training loss  Later loss values were  Not diagnosed as a bug;
                          approximately -2 to     inspect objective.
                          -2.5.                   

  Large heading error     Reported heading error  Units and calculation
                          approximately 40.81.    remain unverified.

  Unknown coordinate      ADE/FDE numerical       Verify dataset
  scale                   values available,       documentation and
                          physical units not      source preprocessing.
                          confirmed.              

  Best-of-five test       Test script chooses     Label result as
  metric                  best of 5 samples by    best-of-five; use
                          ADE.                    validation split for
                                                  selection.

  Colab runtime           Colab displayed         Important artifacts
  disconnected            "Reconnect T4" after    were copied to Drive
                          backup.                 before stopping.
  -----------------------------------------------------------------------

------------------------------------------------------------------------

## 7. Current status and phase tracking

**Current phase: Phase 1 --- PCG Baseline Reproduction and Validation.**

### Completed

-   [x] PCG repository cloned in Colab.
-   [x] PCG repository cloned locally.
-   [x] HBS training data loaded.
-   [x] 98 empty-pedestrian sequences filtered.
-   [x] Five-epoch Collision Grid training completed on T4.
-   [x] Epoch 0--4 checkpoints saved.
-   [x] Epoch-4 evaluation completed on 1,081 test sequences.
-   [x] Evaluation metrics recorded.
-   [x] Prediction result structure inspected.
-   [x] Checkpoints and configurations backed up to Drive.
-   [x] Test predictions backed up to Drive.

### In progress

-   [ ] Map the 12 elements of each result tuple to source-code
    definitions.
-   [ ] Inspect prediction generation and plotting code.
-   [ ] Verify HBS coordinate units and physical scale.
-   [ ] Inspect ADE/FDE, collision, speed, and heading metric
    implementations.
-   [ ] Review validation procedure and checkpoint-selection logic.
-   [ ] Document baseline strengths and limitations.

### Not started

-   [ ] YOLO detection frontend.
-   [ ] ByteTrack tracking.
-   [ ] Camera calibration and homography.
-   [ ] Metric road-coordinate conversion.
-   [ ] Kinematic feature extraction and smoothing.
-   [ ] Heterogeneous graph construction.
-   [ ] GATv2 interaction encoder.
-   [ ] Dilated TCN temporal encoder.
-   [ ] Multimodal trajectory forecasting head.
-   [ ] Conflict prediction head.
-   [ ] TTC/PET/minimum-separation evaluation for the proposed model.
-   [ ] Probability calibration and ECE.
-   [ ] Risk classification and early-warning dispatcher.
-   [ ] End-to-end real-time video evaluation.

------------------------------------------------------------------------

## 8. Immediate next steps

### Step 1 --- Verify local Git workspace

In the local PCG repository, open the VS Code terminal and run:

``` powershell
git status
git branch --show-current
git remote -v
```

A separate development branch was recommended:

``` powershell
git switch -c pedestrian-safety-development
```

If it already exists:

``` powershell
git switch pedestrian-safety-development
```

The user confirmed the repository is cloned both locally and in Colab,
but local Git outputs and branch creation are not yet confirmed.

### Step 2 --- Inspect PCG result tuple definitions

Search the repository for: - `test_results.pkl` - `write_to_plot_file` -
`get_velocity_errors` - ADE/FDE aggregation. - Collision percentage
calculation. - Code that appends/saves result tuples.

Goal: map all 12 tuple elements to named fields and understand
coordinate conventions.

### Step 3 --- Verify dataset coordinate system

Establish: - Coordinate origin and axes. - Pixel or physical units. -
Any conversion factor. - Frame rate and time interval. - Whether
trajectory positions refer to pedestrian centres, bounding-box centres,
or another point.

Do not convert ADE/FDE to metres until supported by code or dataset
documentation.

### Step 4 --- Establish a defensible baseline protocol

-   Preserve the epoch-4 best-of-five result as the initial reproduction
    record.
-   Do not repeatedly use the test set for model selection.
-   Identify or create a validation protocol.
-   Inspect `train.py` before attempting more training: resume-training
    support has not been verified. Do not assume a fresh run resumes
    from epoch 4.
-   Keep the original epoch-181 checkpoint/config separate from the new
    5-epoch experiment.

### Step 5 --- Implement the proposed model

After baseline analysis: 1. Create a clean project structure. 2.
Implement and validate perception/tracking. 3. Build geometric/kinematic
preprocessing. 4. Construct heterogeneous interaction graphs. 5.
Implement GATv2 + TCN. 6. Add multimodal trajectory prediction. 7. Add
conflict metrics and risk prediction. 8. Calibrate probabilities and
implement alerts.

------------------------------------------------------------------------

## 9. Environment and workflow decisions

### Local development

-   Windows PC and VS Code.
-   Local clone of PCG repository.
-   Previously reported laptop: Intel 12th Gen i5-1235U, 16 GB RAM,
    Intel Iris Xe integrated graphics.

### GPU experimentation

-   Google Colab.
-   Tesla T4 used for the completed 5-epoch run.
-   User has Colab compute credits; current remaining balance is not
    confirmed.
-   Do not switch to A100 without measured need. A faster GPU can reduce
    runtime or provide more memory; it does not inherently improve model
    accuracy.

### Persistence and division of responsibilities

-   **VS Code:** source editing, project structure, debugging, Git,
    documentation.
-   **Colab:** GPU training and expensive evaluation.
-   **Google Drive:** checkpoints, configurations, logs, and experiment
    artifacts.
-   Keep large datasets/checkpoints out of ordinary Git commits.

Drive backup folder: `MyDrive/PCG_Experiments/PCG_T4_5_Epoch_Run`

------------------------------------------------------------------------

## 10. Experiment registry

  -----------------------------------------------------------------------------
  Experiment     Hardware       Method         Configuration     Outcome
  -------------- -------------- -------------- ----------------- --------------
  PCG baseline   Colab Tesla T4 PCG method 4 / 5 epochs, batch   Completed in
  training                      Collision Grid 10, neighborhood  \~748.46
                                               size 8            seconds

  PCG evaluation Colab Tesla T4 Epoch-4        `sample_size=5`   1,081 test
                                checkpoint                       sequences
                                                                 processed

  Original       Google Drive   Existing epoch Original          Backup
  checkpoint                    181            CollisionGrid     completed
  preservation                                 config saved      
                                               separately        
  -----------------------------------------------------------------------------

### PCG 5-epoch run record

-   Training sequences after filtering: 2,386.
-   Filtered sequences: 98.
-   Batches per epoch: 238.
-   Epochs: 5.
-   Total training time: approximately 12.47 minutes.
-   Test sequences: 1,081.
-   Sampling: best-of-five according to ADE.
-   ADE: 0.5587965606209528.
-   FDE: 1.1381949186325073.
-   MHD: 3.071232318878174.
-   Speed error: 0.37326878044215545.
-   Heading error: 40.81245962143868.
-   Overall collision: 2.17%.
-   P-P collision: 5.91%.
-   P-V collision: 0.21%.

Metric units and exact definitions must be verified before scientific
comparison or claims.

------------------------------------------------------------------------

## 11. Reproducibility rules

1.  Never overwrite the original PCG checkpoint/configuration without
    preserving a copy.
2.  Give each experiment a unique folder name.
3.  Save exact config, checkpoint, logs, and evaluation output for every
    experiment.
4.  Record random seeds and software versions in future runs.
5.  Record dataset version/split and preprocessing details.
6.  Do not select epochs or hyperparameters by repeatedly checking test
    results.
7.  Label best-of-N metrics clearly and report N.
8.  Do not report physical units unless verified.
9.  Do not describe collision percentages as calibrated probabilities.
10. Keep baseline and proposed-model implementations distinguishable.

------------------------------------------------------------------------

## 12. Open questions

-   What is the exact meaning of each of the 12 PCG result tuple
    elements?
-   What are the HBS coordinate units and physical scale?
-   What are the precise units and aggregation method for heading error?
-   Does the PCG training script support resuming from a checkpoint?
-   Which dataset will support the complete video-based
    pedestrian/vehicle system?
-   What camera calibration information will be available?
-   What observation and prediction horizons will the proposed model
    use?
-   What hardware and FPS target will be used for final real-time
    evaluation?
-   What conflict-label definition and ground-truth protocol will be
    used?
-   How will probability calibration and early-warning-time ground truth
    be evaluated?

These are unresolved questions, not decisions already made.

------------------------------------------------------------------------

## 13. Handoff summary

The project is **not yet at the proposed-model implementation stage**.
The completed work is a short PCG baseline reproduction and evaluation.

The immediate next action is to inspect PCG's result-writing and
metric-calculation code, map the 12-element prediction tuple, and verify
coordinate units. Then verify the local VS Code Git workspace and
proceed to the perception frontend.

**Last confirmed state:** Colab T4 runtime showed "Reconnect" after
successful backup. PCG clones exist locally and in Colab. Local Git
status/branch/remote output is pending.

------------------------------------------------------------------------

*End of master project file.*



------------------------------------------------------------------------

# 14. Consolidated source-code and dataset audit (10 November 2026)

This section records findings from the additional files inspected in the
continuation chat. Where an earlier section conflicts with this section,
**Section 14 and the updated handoff in Section 15 take precedence**.
The original reproduction artifacts have not been modified.

## 14.1 Local development setup reported by the user

The user reports that:

- The repository has been cloned and opened in VS Code.
- A Python environment named `pcg-baseline` was created with Python 3.9.
- The repository's required/original dependency versions were installed
  rather than upgrading packages indiscriminately.
- Git branches were created for separate testing and modification work.
- The user has since shared this branch output from the local repository:

```text
  baseline-original
  master
* reproduction
```

The active branch at the time of that output was `reproduction`.
The branch names for the separately described testing/modification
branches are not confirmed by that output. Re-check the current branch
state before making changes.

The local laptop is suitable for source inspection, preprocessing
experiments and lightweight debugging. Use Colab for GPU-heavy training.
Do not assume the local Intel Iris Xe environment can run the original
CUDA-default training script without adapting its configuration.

## 14.2 Source files inspected

The following repository files have now been read:

- `Interaction.py`
- `model/model_CollisionGrid.py`
- `train.py`
- `utils/helper.py`
- `test.py`
- `utils/DataLoader.py`
- `utils/grid.py`
- `visualization.py`

### A. `Interaction.py` — TTC-based interaction grids

The file contains:

- `Time2Conflict(...)`: computes a constant-relative-velocity
  closest-approach/conflict-entry calculation using relative position,
  relative velocity and a minimum separation distance.
- `approach_dir_sector(...)` and `approach_angle_sector(...)`:
  angular sector calculations.
- `getInteractionGridMask(...)`: creates pedestrian interaction masks
  and TTC-related masks.
- `getSequenceInteractionGridMask(...)`: applies the interaction-mask
  calculation across a sequence.

Important implementation observations:

1. The calculation assumes constant velocities over the conflict
   calculation horizon. It does not model acceleration or turning.
2. `Time2Conflict` checks zero relative speed before checking whether
   agents are already inside the minimum distance. This ordering may
   classify an already-close, same-velocity pair as no future conflict.
   Verify with a unit test before changing it.
3. `.data.numpy()` requires CPU tensors; CUDA tensors need to be moved
   to CPU before this conversion.
4. `TTC_min.sort()` mutates the supplied list.
5. The occupancy branch should be checked carefully because the
   returned `TTC_mask` may not be initialized in that branch.
6. The TTC-mask value is based on a threshold-minus-conflict-time
   expression. Its interpretation and threshold indexing should be
   unit-tested before reusing it in a new model.
7. Stationary or near-zero velocity vectors can make angular-sector
   interpretation ambiguous.

These are audit observations, not approved code modifications.

### B. `model_CollisionGrid.py` — model architecture

The inspected `CollisionGridModel` uses:

- Position embedding: `Linear(input_size, embedding_size)`.
- Pedestrian interaction embedding: `Linear(num_TTC * num_sector,
  embedding_size)`.
- Vehicle interaction embedding:
  `Linear(num_TTC_veh * num_sector, embedding_size)`.
- Concatenation of the three embeddings.
- An `LSTMCell` by default, with an optional `GRUCell`.
- An output layer with five values, documented as bivariate Gaussian
  distribution parameters.

For the recorded run, each embedding has dimension 64, so the recurrent
cell receives 192 concatenated features and has hidden size 128.

The helper functions `getSocialTensor` and `getSocialTensor2` are
defined in this file, but are not called by the inspected `forward()`
path. Do not assume they are active in method 4's forward pass.

`forward()` uses the position and pedestrian/vehicle TTC-grid features.
The exact mapping of the five output values is defined in `helper.py`
and recorded below.

### C. `utils/helper.py` — probabilistic outputs and metrics

`getCoef(outputs)` maps the five network outputs to:

- `mux`, `muy`: predicted displacement means.
- `sx`, `sy`: positive standard deviations, obtained with exponentials.
- `corr`: correlation constrained to [-1, 1], obtained with `tanh`.

`Gaussian2DLikelihood(...)` computes a bivariate Gaussian likelihood
objective. A negative reported negative-log-likelihood is possible when
the probability density exceeds 1; a negative value alone is not proof
of a training bug.

Other findings:

1. `position_change_seq(...)` converts absolute positions into
   frame-to-frame displacements and records the first observed position.
2. `revert_postion_change_seq2(...)` reconstructs absolute coordinates.
   Its inference-mode boundary (`ind > obs_length`) should be checked:
   the first predicted frame may still use a ground-truth position as
   the previous position.
3. `get_mean_error(...)` and `get_final_error(...)` calculate trajectory
   errors over present agents/timesteps.
4. `get_hausdorff_distance(...)` is not the symmetric standard
   Hausdorff distance: the inspected implementation takes a directed
   maximum from predicted points to ground-truth points.
5. `get_velocity_errors(...)` calculates speed and angular-heading
   differences; the test script applies square roots to the accumulated
   squared errors. The reported heading value is in degrees in the
   inspected implementation, but its aggregation should still be
   documented precisely.
6. `sequence_error(...)` samples from the Gaussian output and reconstructs
   trajectories for training-time error logging. It is not the same as
   the full test-time autoregressive sampling procedure.

### D. `train.py` — training-loop audit

The inspected training script:

- Uses RMSprop with learning rate 0.001.
- Saves a checkpoint after each epoch.
- Uses method 4 for Collision Grid.
- Builds/stores interaction grids when `store_grid=True`.
- Uses 12-frame sequences with 11 shifted input/target steps in the
  training loop.
- Computes Gaussian 2D likelihood loss.
- Accumulates gradients across sequences in a batch before optimizer
  update.

Important caveats:

1. Gradient clipping is configured by `grad_clip=10.0`, but the
   clipping code in the inspected script is commented out.
2. `lambda_param` exists in the arguments, but the corresponding
   weight-decay implementation is commented out.
3. Validation-related arguments exist, but no validation loop was
   found in this `train.py` training loop.
4. `save_every` does not appear to control checkpoint saving here;
   checkpoints are saved after each epoch.
5. The CUDA argument is declared with `store_true` and a default of
   `True`, so the normal parser does not provide a straightforward
   command-line way to disable CUDA.
6. `D_min` and `D_min_veh` are declared with `type=int` despite their
   floating-point defaults. Passing decimal values on the command line
   may fail argument parsing.
7. The script does not show a complete checkpoint-resume implementation.
   Saving optimizer state in a checkpoint does not itself mean the
   training script can resume automatically.

Do not change these behaviours in the reproduction branch. Any fix
should be made in a separate experiment branch and documented.

### E. `test.py` — evaluation protocol and leakage risks

The test script:

- Loads the saved configuration and selected checkpoint.
- Uses a test DataLoader with batch size 1 and inference mode.
- Samples multiple future trajectories.
- For the recorded run, `sample_size=5`.
- Selects the sample with the lowest ADE against ground truth.

This is **best-of-five oracle selection**. The selected trajectory is
chosen using future ground-truth error, which is unavailable in a real
deployment. Label these results best-of-five and do not compare them
directly with single-sample or deployment-time metrics without a
matching protocol.

The inspected prediction loop also uses future ground-truth vehicle
states and future pedestrian presence lists in parts of the interaction
calculation. This makes the evaluation partly oracle-assisted rather
than a fully autonomous forecast. It must be disclosed in any report.

The result tuple has 12 elements, mapped by source-code inspection:

| Index | Meaning |
|---:|---|
| 0 | Original/ground-truth pedestrian sequence |
| 1 | Selected predicted pedestrian sequence |
| 2 | Pedestrian IDs present per frame |
| 3 | Pedestrian ID-to-tensor-index lookup |
| 4 | `None` placeholder |
| 5 | Observation length |
| 6 | Distribution/output parameter sequence |
| 7 | Original/ground-truth vehicle sequence |
| 8 | Vehicle IDs present per frame |
| 9 | Vehicle ID-to-tensor-index lookup |
| 10 | Pedestrian interaction-grid sequence |
| 11 | Vehicle-in-pedestrian interaction-grid sequence |

The stored prediction array contains the full 12-frame sequence, not
only the six future frames. Its feature dimensions are nine, while the
distribution/output array has five values per agent/frame. The
distribution array contains different forms of values in observed and
predicted portions; do not treat every row as an identical parameter
representation without checking the test code.

Additional evaluation caveats:

- Best-of-five sample selection uses ground-truth ADE.
- Future vehicle ground truth and future pedestrian-presence information
  are supplied during parts of prediction.
- Collision percentages are threshold-based geometric counts, not
  calibrated conflict probabilities.
- Metric aggregation differs by metric; document denominators and
  pair-counting rules before comparing results.
- The MHD implementation is a directed variant, not symmetric
  Hausdorff distance.
- The inference reconstruction boundary should be checked for possible
  first-prediction-step ground-truth leakage.

### F. `utils/DataLoader.py` — HBS preprocessing audit

The constructor and preprocessing code assume:

```python
self.downsample_step = 1
fps = 2 / self.downsample_step
self.timestamp = 1 / fps
```

The inspected `hbs.csv` confirms a 0.5-second timestamp interval,
consistent with 2 observations per second in this CSV.

Other findings:

1. The CSV columns used by the loader are
   `frame_id`, `agent_id`, `pos_x`, `pos_y`, `vel_x`, `vel_y`,
   `label`, and `timestamp`. The actual CSV also has `scene_id`.
2. Pedestrian labels accepted by the loader are `pedestrian` and `ped`.
3. Vehicle labels accepted by the loader are `car`, `cart`, and `veh`.
4. The actual HBS CSV contains `bike` labels, which the current loader
   does not include. Therefore, bicycle observations are excluded by
   this baseline preprocessing.
5. The loader calculates acceleration from velocity differences using
   the fixed timestamp interval. A missing observation within an agent
   track can make a fixed-interval derivative assumption inaccurate.
6. `considered_frames` is generated with `np.arange(min_frame,
   max_frame, ...)`, which excludes the maximum frame.
7. The number of sliding windows is calculated as `N - seq_length`;
   the usual stride-one count is `N - seq_length + 1`. One possible
   window is therefore omitted.
8. The train/test split is applied after overlapping windows are
   generated. Adjacent windows at the split boundary can share frames.
   For a 12-frame stride-one window, the boundary windows can overlap
   by 11 frames. This weakens split independence.
9. No genuine validation partition was generated by the inspected
   preprocessing path, despite validation-related paths/arguments.
10. Filtering retains pedestrians present throughout the sequence.
    This removes agents that enter/leave during a window and changes
    the evaluation population. Vehicle filtering is not symmetric.
11. Cached pickle files can prevent preprocessing changes from taking
    effect unless the cache is deliberately regenerated.
12. Batch creation discards a final incomplete batch.

Do not regenerate or overwrite the baseline pickle files. Corrected
preprocessing should be a separately versioned experiment.

### G. `utils/grid.py` — conventional occupancy grids

This file implements conventional spatial occupancy grids:

- `getGridMask(...)`: pedestrian-to-pedestrian occupancy.
- `getSequenceGridMask(...)`: applies the pedestrian grid per frame.
- `getGridMask_heterogeneous(...)`: pedestrian-to-vehicle occupancy.
- `getSequenceGridMask_heterogeneous(...)`: applies heterogeneous
  occupancy grids across a sequence.

For a 4x4 grid, there are 16 cells. With occupancy disabled, the
homogeneous mask has shape `pedestrians x pedestrians x 16`; the
heterogeneous mask has shape `pedestrians x vehicles x 16`.

The homogeneous neighbourhood width is `neighborhood_size`. The
heterogeneous function doubles the supplied neighbourhood value when
constructing its width/height bounds.

These conventional occupancy-grid functions are distinct from the
TTC-based interaction functions in `Interaction.py`. The inspected
method-4 forward path uses TTC-grid features; do not assume the
conventional occupancy-grid helpers are the active method-4 input path.

Implementation observations:

- `.data.numpy()` expects CPU tensors.
- Upper spatial boundaries are excluded by the half-open range checks.
- The sequence functions convert IDs to integers in place.
- Binary occupancy merges multiple agents in the same cell when
  `is_occupancy=True`; the non-occupancy representation retains an
  agent dimension.

### H. `visualization.py` — plotting audit

The visualization file provides:

- `plot_trajecotries(...)`: trajectory comparison plot.
- `Loss_Plot(...)`: training error/loss plots.
- `main()`: loads Collision Grid, Social LSTM, Vanilla LSTM and
  Linear Regression result files and plots every 50th test sequence.

The plot conventions are:

- Blue: Collision Grid.
- Red: Social LSTM.
- Green: Vanilla LSTM.
- Yellow: Linear Regression.
- Black: ground truth.
- Grey: vehicles.

The plotting function also marks an ego pedestrian and indicates
neighbours inside/outside its interaction grid.

Limitations:

1. The main comparison requires all four model result files.
2. The missing-file exception handling prints an error but does not
   reliably stop before attempting to load the missing handle.
3. Axes are labelled in metres even though the HBS physical coordinate
   scale has not been verified.
4. Plot bounds are based on pedestrian predictions and may crop vehicles.
5. Only every 50th test sequence is plotted.
6. The first pedestrian in the frame is hard-coded as the ego agent.
7. The visualization shows grid membership, not calibrated collision
   probability or a complete TTC/risk timeline.

Use this as a reference for understanding the original repository,
not as the final visualization system for the proposed project.

## 14.3 HBS CSV audit

The uploaded `hbs.csv` was directly inspected. It contains:

- 43,459 rows.
- 9 columns:
  `frame_id`, `agent_id`, `pos_x`, `pos_y`, `label`, `scene_id`,
  `timestamp`, `vel_x`, `vel_y`.
- 1,475 unique agent IDs.
- 3,620 unique frame IDs, from 0 through 3,619.
- One scene ID in this file.
- Labels and observation counts:

| Label | Observations | Unique agents |
|---|---:|---:|
| `ped` | 28,123 | 1,115 |
| `car` | 14,414 | 331 |
| `bike` | 922 | 29 |
| **Total** | **43,459** | **1,475** |

The timestamps follow:

```text
timestamp = frame_id * 0.5
```

The CSV therefore has a 0.5-second sampling interval (2 Hz) in the
provided data. This confirms the loader's 2 FPS assumption for this
CSV, but does not prove the native/original camera FPS.

The supplied velocity columns were checked against position differences
for consecutive observations and were numerically consistent to
floating-point precision in the inspected data. This is a consistency
check, not proof of physical units.

Additional checks recorded:

- No duplicate scene/frame/agent records were found.
- No agent had multiple labels.
- One gap was identified within an individual track: car ID 1446
  between frame 1485 and frame 1494.
- The final frame is 3619; the current loader's `arange` upper bound
  excludes it.
- Coordinate units, coordinate origin/axes, camera calibration,
  physical scale, and point-of-person convention remain unverified.

**Dataset conclusion:** HBS is suitable for continuing analysis of the
existing trajectory baseline, but it is one scene and sampled at 2 Hz.
It does not by itself demonstrate a 25–30 FPS end-to-end video
perception system. Bicycle observations are present but excluded by
the current PCG loader.

## 14.4 Configuration files reconciled

Three configuration files were inspected:

| File | `num_epochs` | `neighborhood_size` | Role |
|---|---:|---:|---|
| `CollisionGrid_original_config.pkl` | 200 | 32 | Original/default reference |
| Earlier `config.pkl` | 200 | 32 | Original/default configuration copy |
| `config(1).pkl` | 5 | 8 | Five-epoch experiment configuration |

The five-epoch config also records:

- `input_size=2`, `output_size=5`
- `rnn_size=128`, `batch_size=10`
- `seq_length=12`, `obs_length=6`, `pred_length=6`
- `learning_rate=0.001`, `dropout=0.5`
- `embedding_size=64`, `embedding_size_action=32`
- `grid_size=4`, `maxNumPeds=27`
- `neighborhood_size_veh_in_ped=64`,
  `grid_size_veh_in_ped=4`
- `num_sector=8`, `TTC=[9]`, `D_min=0.7`,
  `TTC_veh=[8]`, `D_min_veh=1.0`
- `method=4`, `store_grid=True`, `use_cuda=True`
- `grad_clip=10.0`, `lambda_param=0.0005`,
  `decay_rate=0.95`, `freq_optimizer=8`,
  `num_validation=2`, `freq_validation=1`.

The values for gradient clipping, weight decay and validation arguments
being present in the config do not mean those mechanisms were active
in the inspected training loop.

## 14.5 Training log and loss plot

The uploaded `log_curve.txt` records five epochs:

| Epoch | Training loss | Training error |
|---:|---:|---:|
| 0 | -0.9362500777 | 0.3653495561 |
| 1 | -1.6138312124 | 0.2589591922 |
| 2 | -1.7974503563 | 0.2507183386 |
| 3 | -1.9391287907 | 0.2424761454 |
| 4 | -2.0328173953 | 0.2373968073 |

The uploaded batch-level plot shows approximately 1,190 training
batches, consistent with 238 batches per epoch x 5 epochs.

The training loss becomes more negative and the training error falls.
This is evidence of optimization progress, not evidence by itself of
generalization or calibrated safety performance.

## 14.6 Checkpoint archive inspected

The uploaded ZIP contains six checkpoint files:

- `SOCIALLSTM_lstm_model_0.tar`
- `SOCIALLSTM_lstm_model_1.tar`
- `SOCIALLSTM_lstm_model_2.tar`
- `SOCIALLSTM_lstm_model_3.tar`
- `SOCIALLSTM_lstm_model_4.tar`
- `SOCIALLSTM_lstm_model_181.tar`

The five epoch-0-to-4 checkpoints are each 1,347,031 bytes. The
epoch-181 checkpoint is 1,341,853 bytes.

The checkpoints were opened with PyTorch. Each contains:

- `epoch`
- `state_dict`
- `optimizer_state_dict`

The stored epoch values match the filenames. Each inspected model
state dictionary contains 12 parameter entries.

**Correction to an earlier chat statement:** the optimizer state is
not `None`. The epoch-4 checkpoint contains an optimizer-state
dictionary with `state` and `param_groups`, including 12 parameter
states. This supports preservation of optimizer information, although
the inspected `train.py` still does not implement automatic resume
loading. Do not claim that exact resume is supported until a resume
path is implemented and tested.

The epoch-181 checkpoint is a separate original/reference artifact.
Do not substitute it for epoch 4.

## 14.7 Correct test-result file identified

Two result files were uploaded:

- `test_results.pkl`
- `test_results_epoch4.pkl`

Both contain 1,081 sequences and share the same ground-truth
pedestrian and vehicle trajectories, but their predicted trajectories
and distribution outputs differ.

The direct ADE/FDE calculation previously performed on the earlier
`test_results.pkl` did not match the logged/reported evaluation
metrics. The corresponding calculation on
`test_results_epoch4.pkl`, over the six future frames, matched the
reported values to the displayed precision:

- ADE approximately 0.5587966.
- FDE approximately 1.1381949.

Therefore, use `test_results_epoch4.pkl` as the authoritative saved
prediction result for the five-epoch epoch-4 experiment. Retain the
other file as a separate result artifact; do not overwrite or delete it.

This establishes consistency among the five-epoch config, epoch-0-to-4
checkpoints, training log, and epoch-4 result metrics. It does not
provide a cryptographic provenance link between the checkpoint and
result file; future experiments should save a manifest with hashes.

## 14.8 Baseline limitations now explicitly recorded

The following must be disclosed in reports and comparisons:

1. Five training epochs are a short reproduction run, not a fully
   tuned or converged reproduction of a 200-epoch default.
2. Test-time best-of-five selection uses ground-truth ADE (oracle
   selection).
3. Future vehicle ground-truth states and future pedestrian presence
   lists are used in parts of the prediction loop.
4. The HBS physical coordinate scale is not verified; do not label
   ADE/FDE as metres.
5. HBS is one scene in the uploaded CSV and has a 2 Hz sample interval.
6. The current loader excludes the `bike` class.
7. Train/test windows are created before splitting and can overlap at
   the split boundary.
8. The inspected preprocessing path does not create a genuine
   validation split.
9. The current evaluation is not a calibrated conflict-probability
   evaluation.
10. The MHD implementation is a directed variant rather than symmetric
    Hausdorff distance.
11. The model's interaction calculation assumes constant relative
    velocity for TTC-style conflict calculations.
12. The original visualization's metre axis labels are not justified
    until coordinate units are verified.

------------------------------------------------------------------------

# 15. Updated project status and next-step execution plan

## 15.1 Current phase

**Phase 1 — PCG baseline reproduction and source/data audit.**

The five-epoch reproduction artifacts are now reconciled. The source
audit and initial CSV audit are complete. The baseline is not yet
scientifically fully validated because coordinate units, data-split
independence, metric definitions and oracle-assisted evaluation
limitations still need to be addressed.

## 15.2 Completed since the previous master-file version

- [x] Inspected `Interaction.py`.
- [x] Inspected `model_CollisionGrid.py`.
- [x] Inspected `train.py`.
- [x] Inspected `utils/helper.py`.
- [x] Inspected `test.py`.
- [x] Inspected `utils/DataLoader.py`.
- [x] Inspected `utils/grid.py`.
- [x] Inspected `visualization.py`.
- [x] Mapped the 12 result tuple fields to source-code definitions.
- [x] Inspected the uploaded HBS CSV and counted labels/agents/frames.
- [x] Confirmed the CSV timestamp interval is 0.5 seconds.
- [x] Checked supplied velocity columns against consecutive position
      differences.
- [x] Identified `bike` observations omitted by the PCG loader.
- [x] Reconciled original/default config against five-epoch config.
- [x] Inspected five-epoch training log and batch-level loss plot.
- [x] Inspected checkpoint archive and epoch metadata.
- [x] Identified `test_results_epoch4.pkl` as the result file matching
      reported ADE/FDE.
- [x] Documented oracle best-of-five and future-ground-truth evaluation
      limitations.
- [x] Recorded preprocessing, metric, visualization and implementation
      caveats.

## 15.3 Remaining baseline audit tasks

Priority order:

1. [ ] Inspect the exact HBS dataset documentation/source and establish
       coordinate units, coordinate origin/axes, and whether positions
       are image-plane or road-plane coordinates.
2. [ ] Trace metric definitions and aggregation end-to-end:
       ADE, FDE, MHD, speed error, heading error, and collision counts.
3. [ ] Write small unit tests for `Time2Conflict`, TTC-mask threshold
       behaviour, zero-relative-velocity cases, and the inference
       reconstruction boundary.
4. [ ] Quantify train/test overlap caused by splitting after sliding
       window creation. Do not silently rewrite the original split.
5. [ ] Design a corrected validation protocol by splitting the raw
       timeline (or independent scenes/sequences) before window creation.
       Keep it separate from the reproduction baseline.
6. [ ] Decide how bicycles should be represented in the proposed
       heterogeneous system. Do not change the baseline label mapping
       without a separate experiment.
7. [ ] Create a manifest linking config, checkpoint, result, dataset
       split, code commit, package versions, seed, and SHA-256 hashes.
8. [ ] Verify all required artifacts are present in Drive and local
       storage.

## 15.4 Git and branch safety before editing

The user reports a local VS Code setup, Python 3.9 environment, and
installed repository-compatible dependency versions. The last shared
branch output was:

```text
  baseline-original
  master
* reproduction
```

Before modifying code, run in the VS Code terminal:

```powershell
git status
git branch -vv
git remote -v
git log -5 --oneline
```

Then:

1. Confirm `baseline-original` is an untouched reference branch.
2. Confirm `reproduction` contains the reproduced baseline state.
3. Create or identify a separate branch for controlled fixes, for
   example `baseline-audit-fixes`.
4. Create a separate branch for the proposed architecture, for
   example `proposed-model`.
5. Do not assume branch names already exist; inspect before creating.
6. Commit code and small documentation/config manifests. Keep large
   datasets, checkpoints and pickle outputs in Drive or other artifact
   storage, not ordinary Git history.
7. Tag or record the exact commit used for the five-epoch experiment.

Suggested commands only after checking current branch/status:

```powershell
git switch reproduction
git status
git switch -c baseline-audit-fixes
```

Do not run these commands blindly if the branch already exists or the
working tree has uncommitted changes.

## 15.5 Recommended implementation sequence after the audit

Do not jump directly into GATv2 + TCN. Build and test the system in
independent, measurable stages.

### Phase 2 — Dataset and geometry foundation

1. Establish dataset provenance, license, coordinate system and units.
2. Choose the dataset(s) for the final video-based system.
3. Define a consistent agent schema:
   `track_id`, `class`, timestamp, image bbox/centre, world position,
   velocity, acceleration, heading, confidence, and visibility.
4. Define how camera calibration/homography will produce road-plane
   coordinates.
5. Define handling of track gaps, entry/exit, occlusion, smoothing and
   noisy derivatives.
6. Create a corrected, leakage-aware train/validation/test split.

**Exit criteria:** reproducible data pipeline, documented units,
valid tracks, and split independence checks.

### Phase 3 — Perception frontend

1. Implement video ingestion.
2. Integrate YOLO detection for pedestrian, car, truck, bus and bike.
3. Integrate ByteTrack.
4. Evaluate detection and tracking quality separately.
5. Measure actual throughput and latency on the target hardware.

**Exit criteria:** stable track IDs, documented detection/tracking
metrics, and measured FPS. Do not claim 25–30 FPS until measured.

### Phase 4 — Geometric and kinematic preprocessing

1. Calibrate camera and estimate road-plane homography.
2. Transform image points to world/road coordinates.
3. Estimate velocity, acceleration and heading with timestamp-aware
   calculations.
4. Handle missing frames and track discontinuities.
5. Validate projected trajectories against known geometry or
   annotated reference points.

**Exit criteria:** consistent world-coordinate agent states with
verified units and uncertainty/quality checks.

### Phase 5 — Interaction-aware forecasting

1. Construct heterogeneous agent graphs with P-P, P-V and, if useful,
   V-V edges.
2. Define edge features (relative position, relative velocity,
   distance, heading, time context and agent classes).
3. Implement and test GATv2 spatial interaction encoding.
4. Implement dilated TCN temporal encoding.
5. Add multimodal trajectory hypotheses and mode probabilities.
6. Compare against the preserved PCG baseline and simpler ablations.

**Exit criteria:** leakage-aware ADE/FDE and multimodal metrics,
per-class/per-scenario results, ablations and inference latency.

### Phase 6 — Conflict prediction and safety evaluation

1. Define conflict labels operationally before training.
2. Compute P-P and P-V minimum predicted separation, TTC and PET where
   the required trajectories and geometry support them.
3. Train a conflict prediction head.
4. Evaluate precision, recall, F1, PR-AUC and false alarms at relevant
   horizons.
5. Calibrate probabilities and report reliability diagrams and ECE.
6. Define risk thresholds and early-warning-time evaluation.

**Exit criteria:** calibrated, independently evaluated conflict
probabilities and a defensible early-warning protocol.

### Phase 7 — Real-time integration

1. Integrate perception, tracking, geometry, forecasting and risk
   modules.
2. Measure end-to-end latency, throughput and memory.
3. Test occlusion, dense traffic, low light, camera movement and
   unusual agent interactions.
4. Produce dashboard/alert outputs.
5. Report failure cases and limitations.

**Exit criteria:** measured end-to-end performance on held-out video,
not only offline trajectory datasets.

## 15.6 Immediate next working session

The next coding/research session should follow this order:

**Task 1 — Freeze and verify Git state.**
Get the current `git status`, branch list, remote and recent commits
from VS Code. Confirm the baseline branch is clean and preserved.

**Task 2 — Resolve the metric audit.**
Read the exact metric functions and test aggregation in `helper.py`
and `test.py`. Create a small independent script that recalculates
metrics from `test_results_epoch4.pkl` using explicitly documented
frames, agent masks and aggregation. The ADE/FDE match already
observed should be retained as a regression check.

**Task 3 — Resolve coordinate provenance.**
Find the original HBS dataset documentation and determine whether
`pos_x`/`pos_y` are pixels, metres or another coordinate system. Until
then, all position-error metrics remain in dataset coordinate units.

**Task 4 — Create a baseline audit report and experiment manifest.**
Record the code commit, Python/package versions, config, checkpoint,
result file, data split, seed (if known), metrics and limitations.
Use SHA-256 hashes for artifact identity.

**Task 5 — Only then make controlled fixes.**
First create unit tests and a separate audit-fixes branch. Do not
change the original reproduction artifacts. Once the baseline protocol
is frozen, begin Phase 2 dataset/geometry work.

## 15.7 Evidence and artifact naming

Use the following names when organizing the verified five-epoch run:

- `config_5epoch_neighborhood8.pkl` — copy of the five-epoch config
  currently uploaded as `config(1).pkl`.
- `config_original_default_neighborhood32.pkl` — original/default
  config reference.
- `checkpoint_epoch0.tar` through `checkpoint_epoch4.tar` — preserve
  original filenames in the actual archive; the aliases are for
  documentation only.
- `checkpoint_epoch181_original.tar` — preserve separately.
- `training_log_5epoch.txt` — copy of `log_curve.txt`.
- `training_loss_plot_5epoch.png` — copy of the uploaded batch plot.
- `test_results_epoch4.pkl` — authoritative saved epoch-4 predictions
  matching the reported ADE/FDE.
- `test_results_other_run.pkl` — retain the earlier `test_results.pkl`
  separately because its predictions differ.

Do not rename or move the only copies. Keep an artifact manifest that
records original filenames and hashes.

------------------------------------------------------------------------

## 16. Current handoff summary

The project has completed a five-epoch PCG Collision Grid baseline
run on HBS and has now completed a substantial source-code and dataset
audit. The five-epoch configuration, training log, epoch 0–4
checkpoints, and `test_results_epoch4.pkl` are mutually consistent
with the recorded experiment. The earlier `test_results.pkl` is a
different prediction artifact and should not be used for the reported
epoch-4 metrics.

The baseline is **reproduced at a short-run level, but not yet fully
validated**. Important limitations remain: best-of-five oracle sample
selection, future ground-truth information used in parts of testing,
potential train/test window overlap, no genuine validation split in
the inspected preprocessing path, unverified physical coordinate
units, and the HBS loader's exclusion of bicycles.

The proposed GATv2 + TCN architecture, YOLO/ByteTrack frontend,
calibrated conflict head, risk levels and early-warning dispatcher
remain unimplemented.

**Next action:** verify and freeze the local Git branches, then finish
the metric and coordinate-system audit. After that, establish a
leakage-aware data protocol and begin the perception/geometry
foundation before implementing the proposed forecasting architecture.

*End of updated master project file.*

---

# 17. Phase 2 continuation — baseline audit, archival, and validation handoff (11 November 2026)

This section supersedes earlier immediate-next-step descriptions where they conflict with the Phase 2 progress recorded below. Historical sections are retained for traceability. The original PCG implementation, original dataset/preprocessed files, checkpoints, and authoritative prediction artifact remain preserved.

## 17.1 Phase 2 scope and operating rules

Phase 2 is a controlled audit of the reproduced PCG baseline. It is not the implementation phase for the proposed YOLO/ByteTrack + heterogeneous GATv2 + dilated TCN architecture.

- Do not implement YOLO, ByteTrack, GATv2, or TCN during this phase.
- Do not alter original PCG source on `reproduction`.
- Do not overwrite original data, preprocessed pickles, checkpoints, or the authoritative result artifact.
- Keep audit scripts, tests, reports, and derived outputs in new files.
- Preserve superseded reports as historical evidence and identify the authoritative report.
- Do not rerun training/testing unless a separately approved experiment requires it.
- Do not claim a task complete until its evidence and acceptance criteria have been reviewed.

## 17.2 Confirmed local repository and Git setup

Repository: `D:\BTech-Research\PCG-Baseline`

Environment: `pcg-baseline`

Working branch used for Phase 2: `reproduction`

User fork: `https://github.com/VARMA-0407/PolarCollisionGrid-PedestrianTrajectoryPrediction`

The original repository is configured as `upstream`; the user's fork is `origin`.

Artifact-publication commit:

`7a6f56f Add PCG T4 five-epoch reproduction artifacts`

The commit was pushed to `origin/reproduction`. No pull request to upstream was created.

Before further editing, re-check `git status`, `git branch -vv`, `git remote -v`, and `git log -5 --oneline`. Do not assume the working tree is clean based on an earlier task.

## 17.3 Experiment archive and authoritative artifact

Archive:

`experiments/PCG_T4_5_Epoch_Run/`

```text
experiments/PCG_T4_5_Epoch_Run/
  checkpoints/
    SOCIALLSTM_lstm_model_0.tar
    SOCIALLSTM_lstm_model_1.tar
    SOCIALLSTM_lstm_model_2.tar
    SOCIALLSTM_lstm_model_3.tar
    SOCIALLSTM_lstm_model_4.tar
  configs/
    experiment_config.pkl
    original_config.pkl
  logs/
    log_curve.txt
    loss_plot_batch.png
  results/
    test_results_epoch4.pkl
  manifest.json
```

The archive is approximately 39.4 MiB. Ten artifact SHA-256 hashes are recorded in `manifest.json`. The archive and manifest were committed and pushed in commit `7a6f56f`.

**Authoritative prediction artifact:**

`experiments/PCG_T4_5_Epoch_Run/results/test_results_epoch4.pkl`

Do not substitute the other `test_results.pkl`; it contains different predictions and does not reproduce the reported epoch-4 ADE/FDE.

The authoritative artifact contains 1,081 entries, each a 12-element tuple:

| Index | Meaning |
|---:|---|
| 0 | Original/ground-truth pedestrian sequence |
| 1 | Selected predicted pedestrian sequence |
| 2 | Pedestrian IDs present per frame |
| 3 | Pedestrian ID-to-tensor-index lookup |
| 4 | `None` placeholder |
| 5 | Observation length |
| 6 | Distribution/output parameter sequence |
| 7 | Original/ground-truth vehicle sequence |
| 8 | Vehicle IDs present per frame |
| 9 | Vehicle ID-to-index lookup |
| 10 | Pedestrian interaction-grid sequence |
| 11 | Vehicle-in-pedestrian interaction-grid sequence |

## 17.4 Phase 2 task register

### Task 1 — Git verification and baseline freeze

**Status: COMPLETE**

Repository/fork and branch context were checked. The original remote is `upstream`; the fork is `origin`. Baseline reproduction artifacts were preserved. The `reproduction` branch is the controlled baseline branch; do not modify original implementation there.

### Task 2 — Artifact archival and GitHub publication

**Status: COMPLETE**

The experiment archive was assembled under `experiments/PCG_T4_5_Epoch_Run/`. Artifact hashes were recorded in `manifest.json`. Commit `7a6f56f Add PCG T4 five-epoch reproduction artifacts` was created and successfully pushed to the user's fork at `origin/reproduction`.

### Task 3 — Independent ADE/FDE reconstruction

**Status: COMPLETE**

Files:
- `scripts/inspect_prediction_results.py`
- `scripts/reconstruct_ade_fde.py`
- `experiments/PCG_T4_5_Epoch_Run/results/ade_fde_reconstruction.json`

The reconstruction report records status `MATCH`. Independently reconstructed ADE and FDE match the reported epoch-4 metrics within `1e-5`. Original source and the authoritative prediction pickle were not modified.

### Task 4 — Oracle selection and future-information audit

**Status: COMPLETE — findings reviewed**

Evidence:
- `experiments/PCG_T4_5_Epoch_Run/reports/task4_oracle_future_information_audit.json`
- `experiments/PCG_T4_5_Epoch_Run/logs/task4_sample_function.txt`

Findings:
1. Test-time evaluation uses `sample_size=5`.
2. The test procedure calculates ADE/FDE for sampled trajectories, chooses the sample with minimum ADE against future ground truth, and reports FDE from that same sample.
3. This is best-of-five oracle selection. Future ground truth is unavailable to a deployed predictor, so these are not single-sample deployment metrics.
4. The stored result artifact preserves the selected sample, not all five original candidates.
5. During autoregressive rollout, future pedestrian ground-truth presence/positions are used in constructing interaction inputs.
6. For method 4, future vehicle ground-truth presence/states are also used in parts of the rollout to construct interaction inputs.
7. Future ground truth used only for final evaluation is valid; future ground truth used as rollout input or to select the prediction is not deployment-feasible.

Always label the historical metrics as best-of-five and oracle-assisted. Do not describe them as leakage-free deployment performance.

### Task 5 — Sliding-window overlap audit

**Status: COMPLETE — authoritative audit established**

Authoritative report:

`experiments/PCG_T4_5_Epoch_Run/reports/task5_window_overlap_authoritative.json`

Source evidence established:
- `downsample_step = 1`.
- The DataLoader constructs frames from pedestrian-containing frame IDs.
- `num_seq_in_dataset = len(all_frame_data) - seq_length`.
- Sequence length is 12.
- HBS pre-filter total windows: 3,599.
- Test windows before filtering: 1,115.
- Training windows before filtering: 2,484.
- Filtering removed 34 test and 98 train windows.
- Retained test windows: 1,081.
- Retained training windows: 2,386.

Post-filter overlap:

| Measure | Result |
|---|---:|
| Cross-split overlapping window pairs | 66 |
| Retained test windows affected | 11 |
| Shared temporal frame indices | 11 |
| Shared pedestrian-frame observations | 33 |
| Affected retained test windows | 1.017576% |
| Shared retained test pedestrian-frame observations | 0.387051% |

The 11 shared temporal frame indices are `[1115,1116,1117,1118,1119,1120,1121,1122,1123,1124,1125]`.

Earlier Task 5 reports based on raw CSV reconstruction assumptions are superseded, not deleted:
- `task5_window_overlap_audit.json`
- `task5_window_overlap_post_filter_audit.json`

The authoritative report is `task5_window_overlap_authoritative.json`.

Interpretation: the split is not fully temporally independent. The measured overlap affects a small fraction of retained test windows, but must be disclosed. A corrected protocol should split chronologically with a purge gap before window generation, or use independent scenes where available.

### Task 6 — Corrected validation protocol

**Status: PENDING REVIEW**

The protocol design is to be documented and reviewed; no corrected leakage-free metrics have been produced.

Required principles:
- Only observed pedestrian and vehicle states within the observation horizon may be used as model inputs.
- Future ground-truth positions, IDs, actor presence, and vehicle states must not be used to construct rollout interaction features.
- Future actor entry/exit must be handled through a deployment-feasible tracking/presence mechanism, with explicit birth/death and uncertainty handling.
- Avoid best-of-five oracle selection. Use deterministic single-sample output or a documented non-oracle stochastic evaluation. Oracle metrics may be retained only as a clearly labelled upper bound.
- Create chronological splits with a purge boundary sufficient to prevent 12-frame windows crossing train/test boundaries; independent scenes are preferable where available.
- Report ADE/FDE and relevant conflict/safety metrics, and add calibration and warning-time metrics when the proposed conflict model exists.
- Preserve original historical metrics; do not overwrite or relabel them as corrected results.

### Task 7 — HBS coordinate and provenance audit

**Status: PARTIAL — source behaviour verified; external provenance/physical coordinate interpretation unresolved**

A local metadata inventory search found the repository README and an empty metadata-inventory output file. This alone does not prove that the README or original dataset has no relevant metadata.

Verified from the inspected DataLoader source:
- The loader assumes a 2 Hz sampling rate and a 0.5-second interval.
- `add_features` derives acceleration from velocity differences divided by the loader timestamp.
- Code behaviour does not establish physical units of position or velocity.
- A loader assumption does not prove native recording FPS.

Previously inspected HBS CSV evidence records a 0.5-second timestamp interval and consistency between supplied velocity columns and consecutive position differences. This is a data-consistency check, not proof of metres, pixels, or another physical unit.

Still unresolved:
- Physical units and coordinate origin/axes for `pos_x` and `pos_y`.
- Whether coordinates are image-plane, road-plane, or another coordinate system.
- Whether a homography/calibration file exists for the original data.
- Original recording FPS.
- Original dataset expansion, collection methodology, camera/scene provenance, and license.

Do not report ADE/FDE in metres or claim calibrated world coordinates until authoritative dataset documentation or calibration evidence supports it. Distinguish observed code/CSV facts from unresolved provenance claims.

## 17.5 Historical baseline results — preserve as reported

Training configuration: PCG method 4 (Collision Grid), five epochs, batch size 10, sequence length 12, observation length 6, prediction length 6, learning rate 0.001, RNN size 128, embedding size 64, pedestrian neighbourhood 8, grid size 4, vehicle-in-pedestrian neighbourhood 64, `TTC=[9]`, and `TTC_veh=[8]`.

Training log:

| Epoch | Training loss | Training error |
|---:|---:|---:|
| 0 | -0.936250 | 0.365350 |
| 1 | -1.613831 | 0.258959 |
| 2 | -1.797450 | 0.250718 |
| 3 | -1.939129 | 0.242476 |
| 4 | -2.032817 | 0.237397 |

Historical test metrics:

| Metric | Result |
|---|---:|
| ADE | 0.5587965606209528 |
| FDE | 1.1381949186325073 |
| MHD | 3.0712323189 |
| Speed error | 0.3732687804 |
| Heading error | 40.8124596214 |
| Overall collision | 2.17% |
| P-P collision | 5.91% |
| P-V collision | 0.21% |

These are historical best-of-five, oracle-assisted results. HBS coordinate units are not established; do not append metre units. Reported collision percentages are threshold-based counts, not calibrated conflict probabilities.

## 17.6 Tasks 8 and 9 — next execution package

The next working session is assigned to Tasks 8 and 9 together. The user intends to execute them in a separate ChatGPT conversation using the detailed execution prompt supplied in the preceding project conversation. This section records the handoff and acceptance criteria; it does not claim the work has already been run.

### Task 8 — Unit tests

**Status: NOT STARTED — execution pending in separate chat**

Create an isolated test suite under `tests/phase2/` covering:

1. TTC calculation behaviour from the actual `Interaction.py` implementation, including approaching, separating, zero-relative-velocity, same-position, invalid-input (where supported), and threshold-boundary cases.
2. TTC mask behaviour, including exact `<` versus `<=` threshold semantics, valid/invalid values, empty actors, mixed cases, dimensions, and data types.
3. ADE/FDE reconstruction boundaries using the actual metric functions and original eligibility/aggregation semantics, including zero error, known displacement, multiple pedestrians, missing actors, ineligible actors, final-frame boundary, and no-eligible-agent behaviour where supported.

Expected new files:

```text
tests/phase2/
  conftest.py                         # only if fixtures are needed
  test_ttc.py
  test_ttc_masks.py
  test_ade_fde_boundaries.py

experiments/PCG_T4_5_Epoch_Run/
  reports/task8_unit_test_report.json
  logs/task8_pytest_output.txt
```

Run:

```powershell
conda activate pcg-baseline
python -m pytest tests/phase2 -v
```

Tests must use small synthetic inputs, require neither network access nor a GPU, and must not modify original source, data, checkpoints, or the authoritative result artifact. Any test wrapper or limitation must be disclosed; tests must not silently redefine original functions.

**Task 8 acceptance:** all required tests executed, results recorded, failures/skips explained, and preservation checks satisfied.

### Task 9 — Reproducibility report and manifest

**Status: NOT STARTED — execution pending in separate chat**

Create:
- `experiments/PCG_T4_5_Epoch_Run/reports/PHASE2_REPRODUCIBILITY_REPORT.md`
- `experiments/PCG_T4_5_Epoch_Run/manifest_v2.json`
- `scripts/verify_phase2_manifest.py`
- `experiments/PCG_T4_5_Epoch_Run/logs/task9_manifest_verification.txt`

The report must consolidate experiment identity, source/fork, environment, data/preprocessing, configuration, commands, training log, historical metrics, ADE/FDE reconstruction, oracle and future-information findings, overlap findings, Task 6/7 status, Task 8 results, limitations, artifact inventory, hash verification, and reproduction instructions.

`manifest_v2.json` must include experiment identity, Git branch/commit, artifact paths/sizes/SHA-256 hashes, configuration summary, dataset/preprocessing references, commands, test/audit report references, timestamp, and verification status. It must not contain a self-referential hash for itself.

The verification script must check file existence, byte size, and SHA-256 for each listed artifact; print PASS/FAIL per file; return a non-zero exit code on any failure; and never modify artifacts.

Run:

```powershell
python scripts\verify_phase2_manifest.py
```

Compare existing archived-artifact hashes with the original `manifest.json`. Do not silently replace a mismatch. Do not automatically commit or push Task 8/9 outputs before review.

**Task 9 acceptance:** reproducibility report and v2 manifest exist, all required artifact hashes are verified or mismatches are explicitly documented, missing evidence is identified, and scientific limitations are clearly stated.

### Task 8/9 execution review checklist

After execution, review:
- Full pytest output.
- `task8_unit_test_report.json`.
- `PHASE2_REPRODUCIBILITY_REPORT.md`.
- `manifest_v2.json`.
- `task9_manifest_verification.txt`.
- `git status --short` and latest commit.

Do not mark Tasks 8 or 9 complete based solely on a coding agent's summary. Inspect test logic, expected values, report claims, and hash verification output.

## 17.7 Updated Phase 2 tracker

| Task | Description | Status |
|---:|---|---|
| 1 | Git verification and baseline freeze | Complete |
| 2 | Artifact archival and GitHub publication | Complete |
| 3 | Independent ADE/FDE reconstruction | Complete |
| 4 | Oracle and future-information audit | Complete |
| 5 | Sliding-window overlap audit | Complete |
| 6 | Corrected validation protocol | Pending review |
| 7 | HBS coordinate and provenance audit | Partial |
| 8 | TTC, TTC-mask, and ADE/FDE boundary unit tests | Not started |
| 9 | Reproducibility report and manifest v2 | Not started |
| 10 | Final master project file review/update after Tasks 8–9 | Pending |

## 17.8 Handoff to the next ChatGPT conversation

Continue from this master file and the Phase 2 Tasks 8–9 execution prompt already prepared in the preceding project conversation.

The user wants to execute Tasks 8 and 9 in a separate chat. Do not repeat the Task 1–7 investigation or restart the baseline. Start by helping execute the combined Task 8/9 prompt in the existing repository, then review actual outputs before changing the tracker.

Preserve these distinctions:
- Tasks 1–5 are complete.
- Task 6 is pending review, not a completed corrected evaluation.
- Task 7 is partial; physical coordinate units and original dataset provenance remain unresolved.
- Tasks 8–9 are pending execution.
- Task 10 is the later consolidation step after evidence review.

**Current technical boundary:** the original PCG baseline has been archived and audited, but a corrected leakage-free evaluation has not yet been run. The proposed end-to-end video system remains unimplemented.

*End of Phase 2 continuation and handoff.*
