# Phase 4 Task 3 interaction representation run

Image space only (captured_1920x1080; px, px/s, s, image-plane rad). Every feature physical: false. Not a safety measure.

| Item | Value |
|---|---|
| causal rows | 177 |
| frames with nodes | 30 |
| nodes | 177 (pedestrian 57, vehicle 60, bicyclist 60) |
| edges | 750 |
| edges pedestrian__pedestrian | 54 |
| edges vehicle__pedestrian | 114 |
| edges pedestrian__vehicle | 114 |
| edges bicyclist__pedestrian | 114 |
| edges pedestrian__bicyclist | 114 |
| edges bicyclist__vehicle | 120 |
| edges vehicle__bicyclist | 120 |
| pairs not represented | bicyclist__bicyclist 60, vehicle__vehicle 60 |

All edges equal the complete-graph bound for their relation: True.

See interaction_diagnostics.json, feature_dictionary.json and run_manifest.json.
