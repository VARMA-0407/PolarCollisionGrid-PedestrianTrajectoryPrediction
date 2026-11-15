# Phase 4 Task 3 interaction representation run

Image space only (captured_1920x1080; px, px/s, s, image-plane rad). Every feature physical: false. Not a safety measure.

| Item | Value |
|---|---|
| causal rows | 65388 |
| frames with nodes | 9326 |
| nodes | 65388 (pedestrian 23661, vehicle 34678, bicyclist 7049) |
| edges | 404710 |
| edges pedestrian__pedestrian | 105634 |
| edges vehicle__pedestrian | 81618 |
| edges pedestrian__vehicle | 81618 |
| edges bicyclist__pedestrian | 42205 |
| edges pedestrian__bicyclist | 42205 |
| edges bicyclist__vehicle | 25715 |
| edges vehicle__bicyclist | 25715 |
| pairs not represented | bicyclist__bicyclist 18878, vehicle__vehicle 141544 |

All edges equal the complete-graph bound for their relation: True.

See interaction_diagnostics.json, feature_dictionary.json and run_manifest.json.
