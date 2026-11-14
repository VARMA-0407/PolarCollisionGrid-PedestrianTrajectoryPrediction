# ==============================================================================================
# PART C - ARTIFACT INTEGRITY MANIFEST  ->  artifact_manifest.json   (hashes only; nothing is written to or replaced in the artifacts)
# ==============================================================================================
# Previously recorded hashes. Source A = archived experiment manifest (experiments/PCG_T4_5_Epoch_Run/manifest.json in the project repository).
# Source B = hash of the git-tracked local copy of the historical files, computed on 2026-11-14 (no earlier recorded hash existed for these two).
_A = "archived experiment manifest.json (project repository)"
_B = "git-tracked local copy hashed 2026-11-14 (no earlier recorded hash)"
PREV = {
    "ckpt0": ("876D017CD6D381FBBDD1A9976A5A2843FDCEAEBA0FB45560DB406DF410C825CE", _A),
    "ckpt1": ("EBEE25A064E6DA5D235C5B23ECF8C1E55489EF8A5EEA654A279A6008D90D43F8", _A),
    "ckpt2": ("3ACF2412362C342E3426A7BE4C8B44558241ED3302970CF60554D832E5D0EC74", _A),
    "ckpt3": ("3A4EBD8BC4BFF5B960B9090ACB8EBD16640A01E5CB834B7DBFAA07D42F06B0EF", _A),
    "ckpt4": ("AC9F5462EC741B91541FA6BE74F07167060D63A469F88936C9D5FBC833387D48", _A),
    "experiment_config": ("9C994A14A5F6519B2C78256B3603C2E26975DDB647AC455BF13AECCD18485BA5", _A),
    "original_config": ("763E68C787524E836DD20DC833F6BF949DF488591F01BC74BDAB375ECB8E43D9", _A),
    "ckpt181": ("A01B3C59C425E070E7E25DB47C692226304E83C1F285C35D00AFA4BF1101775F", _B),
    "test_results": ("119E9BB665E2168035062FA075343ABA46CC9B2F2F8810054C16A996579192DE", _A),
    "log_curve": ("89BD93363B9D829B9876ACCCCC242951DB1E00ED6D91561304226466535D84C1", _A),
    "loss_plot_batch": ("7EC8BC95896EC75AA1746BCFD8C7BAD45DB7957E992FA32D81C57E19F9C77205", _A),
}

def _entry(label, path, key=None, note=None):
    h, src = PREV.get(key, (None, None))
    rec = ml.file_record(path, label=label, expected_sha256=h, expected_source=src)
    if note:
        rec["note"] = note
    return rec

ENTRIES = []
for e in range(5):                                   # epoch 0-4 checkpoints: working copy on the Colab disk/Drive link, and the Drive backup
    ENTRIES.append(_entry("checkpoint epoch %d (Store_Results/model)" % e, os.path.join(STORE, "model", "SOCIALLSTM_lstm_model_%d.tar" % e), "ckpt%d" % e))
    ENTRIES.append(_entry("checkpoint epoch %d (Drive backup)" % e, os.path.join(DRIVE_BACKUP, "SOCIALLSTM_lstm_model_%d.tar" % e), "ckpt%d" % e))
ENTRIES += [
    _entry("training configuration (Store_Results/model/config.pkl)", os.path.join(STORE, "model", "config.pkl"), "experiment_config"),
    _entry("training configuration (Drive backup config.pkl)", os.path.join(DRIVE_BACKUP, "config.pkl"), "experiment_config"),
    _entry("CollisionGrid config in repo (STAGED copy of the new training config)", os.path.join(STORE, "model", "CollisionGrid", "config.pkl"), "experiment_config",
           "the evaluation cell copied the new run's config over this file; it is EXPECTED to equal the training config, not the original"),
    _entry("ORIGINAL CollisionGrid config (Drive backup)", os.path.join(DRIVE_BACKUP, "CollisionGrid_original_config.pkl"), "original_config"),
    _entry("historical epoch-181 checkpoint (Store_Results/model/CollisionGrid)", os.path.join(STORE, "model", "CollisionGrid", "SOCIALLSTM_lstm_model_181.tar"), "ckpt181"),
    _entry("historical epoch-181 checkpoint (Drive backup)", os.path.join(DRIVE_BACKUP, "SOCIALLSTM_lstm_model_181.tar"), "ckpt181"),
    _entry("staged epoch-4 checkpoint (Store_Results/model/CollisionGrid)", os.path.join(STORE, "model", "CollisionGrid", "SOCIALLSTM_lstm_model_4.tar"), "ckpt4"),
    _entry("test prediction pickle (Store_Results/plot/test)", os.path.join(STORE, "plot", "test", "test_results.pkl"), "test_results",
           "test.py rewrites this file on every evaluation run; a mismatch can mean it was overwritten after the archived copy was made"),
    _entry("test prediction pickle (Drive backup test_results_epoch4.pkl)", os.path.join(DRIVE_BACKUP, "test_results_epoch4.pkl"), "test_results"),
    _entry("training log (Store_Results/log/log_curve.txt)", os.path.join(STORE, "log", "log_curve.txt"), "log_curve"),
    _entry("batch loss plot (Store_Results/plot/train)", os.path.join(STORE, "plot", "train", "loss_plot_batch.png"), "loss_plot_batch"),
    _entry("epoch loss plot (Store_Results/plot/train)", os.path.join(STORE, "plot", "train", "loss_plot_epoch.png")),
]
# consistency between a working copy and its Drive backup (same content?)
PAIRS = []
for e in range(5):
    a, b = ENTRIES[2 * e], ENTRIES[2 * e + 1]
    PAIRS.append({"pair": "epoch %d checkpoint: working copy vs Drive backup" % e, "both_exist": a["exists"] and b["exists"],
                  "identical": (a["sha256"] == b["sha256"]) if (a["exists"] and b["exists"]) else None})
MANIFEST = {
    "manifest_version": "pcg-artifact-manifest-1.0", "measurement_run_id": MEAS_RUN_ID, "created_utc": ml.utc_now_iso(),
    "policy": "read-only: files are hashed, never modified, moved or replaced",
    "previous_hash_sources": {"A": _A, "B": _B},
    "counts": {"entries": len(ENTRIES), "exist": sum(1 for r in ENTRIES if r["exists"]), "missing": sum(1 for r in ENTRIES if not r["exists"]),
               "match_previous": sum(1 for r in ENTRIES if r["matches_previous_hash"] is True),
               "mismatch_previous": sum(1 for r in ENTRIES if r["matches_previous_hash"] is False),
               "no_previous_hash": sum(1 for r in ENTRIES if r["exists"] and r["matches_previous_hash"] is None)},
    "pair_consistency": PAIRS, "artifacts": ENTRIES}
_sha = ml.write_new_json(os.path.join(MEAS_DIR, "artifact_manifest.json"), MANIFEST)
print("wrote artifact_manifest.json  sha256 =", _sha)
print(json.dumps(MANIFEST["counts"], indent=1))
for r in ENTRIES:
    flag = "MISSING" if not r["exists"] else ("MATCH" if r["matches_previous_hash"] is True else ("MISMATCH" if r["matches_previous_hash"] is False else "no previous hash"))
    print("%-9s %-72s %s" % (flag, r["label"][:72], (r["sha256"] or "")[:16]))
