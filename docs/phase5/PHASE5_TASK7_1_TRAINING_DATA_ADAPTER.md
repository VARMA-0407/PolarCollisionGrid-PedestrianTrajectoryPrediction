# P5.7.1 Training Dataset and Batch Adapter

## Status

Implemented as a deterministic, read-only adapter over the existing P5.3
`windows.npz`, `graph_bank.npz`, and `run_manifest.json` artifacts.

This task does not implement training, optimizers, checkpointing, scaling
fitting, or evaluation.

## API

`pedsafety.phase5.training_data.TrainingWindowDataset`:

```python
dataset = TrainingWindowDataset(run_dir)
sample = dataset[index]
batch = dataset.collate([dataset[0], dataset[1]])
encoder_input = dataset.encoder_input(batch)
```

Hash verification is enabled by default. The explicit
`TrainingWindowDataset(run_dir, verify_hashes=False)` escape hatch is intended
only for controlled inspection of tampered artifacts; batches produced through
that path carry `artifact_hashes_verified: false` in their provenance and must
not be treated as verified datasets.

The adapter delegates graph construction to P5.3 `collate_windows` and dataset
validation to `validate_window_dataset`. It does not interpolate, forward-fill,
merge tracks, or synthesize observations.

## Batch contract

Model/history fields remain at the top level and are compatible with
`EncoderInput.from_collated_batch`:

| Field | Shape | Dtype |
|---|---:|---|
| `node_features` | `[B,6,A,10]` | float32 |
| `node_masks` | `[B,6,A,4]` | bool |
| `present` | `[B,6,A]` | bool |
| `node_type` | `[B,A]` | integer |
| `track_index` | `[B,A]` | integer |
| `slot_valid` | `[B,A]` | bool |
| `edges[r].bhsd` | `[E_r,4]` | int64 |
| `edges[r].edge_attr` | `[E_r,14]` | float32 |
| `edges[r].edge_mask` | `[E_r,14]` | bool |
| `edges[r].flags` | `[E_r,6]` | bool |

Supervision and decoder fields are separate:

| Field | Shape | Dtype |
|---|---:|---|
| `target_future_uv_px` | `[B,15,2]` | float32 |
| `target_future_valid` | `[B,15]` | bool |
| `anchor_uv_px` | `[B,2]` | float32 |
| `target_valid` | `[B]` | bool |

Metadata is stored under `batch["metadata"]` and includes window, video,
group, track, frame-reference, source-artifact, manifest-hash, and dataset
label fields. `metadata["provenance"]` preserves the manifest's exact
`sources`, `outputs`, `git`, and `label` structures without placing them in
model tensors.

The raw anchor is obtained from the final target history position before any
future model processing. It is not derived from normalized encoder features.

## Leakage safeguards

`TrainingWindowDataset.encoder_input` removes:

- `target_future_uv_px`;
- `target_future_valid`;
- `future_frames`;
- `anchor_uv_px`;
- `target_valid`;
- metadata.

It then calls the existing `EncoderInput.from_collated_batch`, which copies
only the P5.5 history keys. The decoder compatibility test calls the decoder
with only a synthetic `target_summary`, raw anchor, and history-derived
`target_valid`; future targets are not passed.

The graph bank contains only P5.3 history frames. Future targets remain
supervision-only arrays.

## Validation

The adapter verifies:

- required artifact files and run manifest;
- every output declared by the run manifest, including `smoke_manifest.json`
  when declared, and its recorded SHA-256;
- manifest configuration consistency;
- P5.3 window and graph-bank validation checks;
- accepted-window count consistency;
- required video/group metadata;
- finite final anchors;
- valid target slot and history presence;
- valid relation-specific edge indices and masks.

## Pilot smoke result

The committed pilot artifact was loaded without regeneration:

- 2,327 accepted windows;
- 84 target tracks;
- one video/group: `pie:video_0004`;
- `H=6`, `F=15`, 10 Hz;
- all accepted windows have 15/15 valid future samples;
- deterministic single- and multi-window collation passed;
- variable agent counts and multiple relation types were exercised.
- combined multi-window edge arrays were compared against independently
  collated windows for batch index, history index, slot endpoints, attributes,
  masks, flags, and relation identity;
- the actual P5.5 encoder and P5.6 decoder completed a CPU interface smoke
  pass using history-only inference inputs.

The artifact remains labelled
`PILOT_DEVELOPMENT_ONLY_NON_INDEPENDENT`.

## Limitations

- This adapter does not perform scaling fitting.
- It does not implement a PyTorch training loop or `DataLoader` worker policy.
- It does not create train/validation/test splits.
- Malformed artifacts are rejected through the existing P5.3 validator; this
  adapter does not repair malformed arrays.
- The pilot target source remains the P5.3 perceived-track smoke source.
- Provenance is preserved from the run manifest but is not independently
  upgraded or reinterpreted by this adapter.
