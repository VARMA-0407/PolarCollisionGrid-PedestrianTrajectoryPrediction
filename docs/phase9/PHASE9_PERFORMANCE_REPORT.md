# Phase 9 — Performance report (engineering benchmark, precomputed-artifact mode)

Source: `experiments/phase9_end_to_end_pie_video_0004/run_manifest.json`, `performance.json`, `performance_attribution.json` (run `pie_video_0004_full`, exit code 0, all 9 stages COMPLETED).
**Scope: Phases 4–8 on precomputed Phase 3 artifacts.** Video decoding and Phase 3 perception were not executed here and are reported **UNAVAILABLE** (not 0 ms). Nothing below is a raw-video
end-to-end FPS, a YOLO inference time or a real-time claim.

## Environment (measured on the benchmark machine)
CPU 12th Gen Intel Core i5-1235U (12 logical CPUs); RAM 16,890,978,304 B (15.7 GiB); Windows 11 (10.0.26300); Python 3.9.25; PyTorch 2.2.1+cpu, 1 thread for inference (the configured bitwise-reproducible setting);
no CUDA device (GPU time and GPU memory: NOT_APPLICABLE); `psutil`, OpenCV, Ultralytics: not installed. CPU name and RAM were read after the run on the same machine (`performance_attribution.json`, `measured_after_run_on_the_same_machine`).

## Workload represented
9,840 video frames (30 fps, 328 s of video, recorded by Phase 3), 65,388 agent states (65,388 valid positions, 62,212 valid velocities, 0 quarantined), 1,500 tracks, 2,327 accepted windows (6,371 candidates rejected:
reasons in the manifest), 8,233 pedestrian–vehicle pairs (8,090 valid, 143 invalid), 1,231 INSUFFICIENT_EVIDENCE risk records, 28 alerts, 1,200 events, 473 transitions into INSUFFICIENT_EVIDENCE. Inputs: `agent_states.jsonl` 67.6 MB, causal kinematics 105.9 MB.

## Per-component time (seconds; mean / median over n repetitions)
| Component | Mean | Median | n | Notes |
|---|---|---|---|---|
| Video decoding | UNAVAILABLE | – | 0 | no raw video, no OpenCV |
| Phase 3 perception (YOLO + ByteTrack) | UNAVAILABLE | – | 0 | **recorded, not measured here**: Colab T4, detect 20.39 ms + track 2.53 ms mean per frame, 609.2 s wall for 9,840 frames (16.15 frames/s including decode and I/O), no warm-up/stage isolation, different hardware |
| Phase 3 artifact validation | 0.52 | 0.52 | 1 | reading and validating 65,388 rows; **not** perception |
| Phase 4 Task 2 image kinematics (incl. its file I/O) | 151.37 | 158.02 | 3 | runs 158.0 / 130.4 / 165.7 (cold, no warm-up) |
| Phase 4 Task 3 interaction (incl. its file I/O) | 104.91 | 106.76 | 3 | runs 106.8 / 99.9 / 108.1 |
| Phase 5 P5.3 window generation (incl. I/O) | 8.46 | 8.10 | 3 | runs 9.2 / 8.1 / 8.1 |
| Phase 5 model load + hash verification | 0.36 | 0.36 | 1 | first pass |
| Phase 5 inference: model forward (2,327 windows, batch 64) | 9.92 | 9.96 | 5 | 4.26 ms per window; steady state after 1 warm-up (20.6 s) |
| Phase 5 inference: collate and other | 7.80 | 7.84 | 5 | dataset item construction and batching |
| Phase 6 conflict processing | 0.330 | 0.327 | 5 | |
| Phase 7 risk processing | 0.183 | 0.180 | 5 | |
| Phase 8 alert manager (no file output) | 1.616 | 1.629 | 5 | the state machine alone |
| Inference chain total (forward → alerts) | 19.87 | 20.03 | 5 | CV 3.7%; 8.54 ms per window |
| File I/O and serialization | PARTIALLY MEASURED | – | – | predictions write 0.28 s measured; reads/writes of Phase 3/4/5/8 are inside their stage times and cannot be separated without modifying those phases |
First-pass stage wall times (the one recorded pass): S3 0.52, S4A 158.0, S4B 106.8, S5A 9.2, S5B 19.9 (includes load, history-only and future-isolation checks, write), S6 0.46, S7 0.27, S8 1.91, S9 0.59.
Phase 4–8 first-pass total: **296.6 s** (S4A + S4B = 89%). Total run wall time 938.0 s includes the benchmark repetitions.

## Throughput statements that are allowed
- Phase 4–8 offline-batch throughput on this CPU: 9,840 frames / 296.6 s = 33.2 frames of video per second of processing, i.e. 30.1 ms per frame — **excluding decoding and perception, processing the whole recording as one batch,
  and requiring each target's future frames to form a window. It is not a raw-video end-to-end rate and must not be compared with a 30 FPS requirement.**
- The Phase 8 alert manager alone handled ≈2×10⁴ observations/s (Phase 8 microbenchmark; 1.6 s for 8,233 records here). This is a component number, not system throughput.
- Real-time assessment rule (`realtime_assessment`): a claim needs perception executed inside the measurement, a causal streaming implementation, a documented configuration and a measured rate ≥ 30 frames/s.
  Unmet here: perception not measured; offline-batch design. **`claim_supported = false`.**

## Memory
Process peak working set 1,541,705,728 B (1.44 GiB) via the Windows `GetProcessMemoryInfo` API (`psutil` is unavailable; the OS counter is reliable at process level). It is monotone and process-wide: 1.175 GB after the first Phase 4
stage and 1.54 GB at the end including the benchmark repetitions; it cannot attribute memory to a stage. GPU memory and utilisation: not applicable. CPU use is reported as mean cores busy per stage (`cpu_s / wall_s`); no OS utilisation counter was sampled.

## Variability
Steady-state inference chain CV 3.7% (n=5); preprocessing stages were repeated cold (n=3) with spread of 130–166 s for Task 2 (CV ≈ 12%); one machine, one thread for inference, other applications not controlled. Tests were deliberately not run
during the benchmark.
