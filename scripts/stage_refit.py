#!/usr/bin/env python3
"""Stage a new same-data refit directory on a server; never launch training."""
from pathlib import Path
import argparse
import datetime
import json
import re
import shutil
import sys

from verify_export import REPO, read, require, sha, verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--gpus", nargs=2, required=True, metavar=("UUID_A", "UUID_B"))
    args = parser.parse_args()
    require(sys.platform.startswith("linux"), "Run on the Linux experiment server")
    verify()
    base = args.prepared_root.resolve(strict=True)
    run = args.run_dir.absolute()
    require(not run.exists() and not run.is_symlink(), "Run directory already exists")
    require(len(set(args.gpus)) == 2 and all(re.fullmatch(r"GPU-[0-9a-fA-F-]{36}", v) for v in args.gpus),
            "Provide two distinct full GPU UUIDs")
    # All originals are only read. Check every training input before creating the run.
    inputs = read(REPO / "provenance/expected_inputs.json")
    checked = {}
    for relative, value in inputs.items():
        if not relative.startswith("data/episodes/"):
            continue
        path = base / "dataset" / relative
        require(path.is_file() and path.stat().st_size == value["bytes"], "Input size: " + relative)
        require(sha(path) == value["sha256"], "Input SHA256: " + relative)
        checked["dataset/" + relative] = value["sha256"]
    require(len(checked) == 100, "Expected 50 RGB/action TRAIN pairs")
    prepared = read(REPO / "provenance/prepared_training_files.json")["files"]
    for relative, value in prepared.items():
        path = base / relative
        require(path.is_file() and path.stat().st_size == value["bytes"], "Prepared size: " + relative)
        require(sha(path) == value["SHA256"], "Prepared SHA256: " + relative)
        checked[relative] = value["SHA256"]

    run.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(REPO / "experiment", run, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for relative in checked:
        destination = run / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if relative == "epoch_selection.json":
            shutil.copyfile(base / relative, destination)
        else:
            destination.symlink_to(base / relative)
    assignment = {"GPU_UUIDs": args.gpus, "provided_for_future_launch": True,
                  "allocation_or_GPU_idle_check_performed": False, "training_started": False}
    with (run / "GPU_assignment.json").open("x", encoding="utf-8") as stream:
        json.dump(assignment, stream, indent=2)
        stream.write("\n")
    receipt = {
        "status": "staged_not_launched", "UTC": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "original_prepared_root": str(base), "new_run": str(run),
        "checked_files": checked, "original_scientific_source_and_contract_unchanged": True,
        "original_files_not_written": True, "TEST_GT_staged": False, "training_started": False,
        "linked_inputs_must_remain_immutable": True,
    }
    with (run / "staging_receipt.json").open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"status": receipt["status"], "new_run": str(run),
                      "verified_input_files": len(checked), "training_started": False}, indent=2))


if __name__ == "__main__":
    main()
