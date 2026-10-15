#!/usr/bin/env python3
"""Check source bytes and syntax without importing numerical libraries."""
from pathlib import Path
import argparse
import ast
import hashlib
import json

REPO = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify():
    original = read(REPO / "experiment/source_manifest.json")
    root = REPO / "experiment"
    for relative, expected in original["files"].items():
        require(sha(root / relative) == expected, "Scientific source changed: " + relative)
    require(sha(root / "contract.json") == original["contract_SHA256"], "Contract changed")

    copies = read(REPO / "provenance/source_copy_manifest.json")["files"]
    for relative, record in copies.items():
        require(sha(REPO / relative) == record["SHA256"], "Original copy changed: " + relative)

    package = read(REPO / "provenance/model_package_manifest.json")
    mapping = {
        "contract.json": "experiment/contract.json",
        "source/common.py": "experiment/source/common.py",
        "public_predict.py": "experiment/analysis/public_predict.py",
        **{
            "reference_source/" + name: "experiment/reference_source/" + name
            for name in ("model03.py", "pls03.py", "keypoints03.py")
        },
    }
    for member, relative in mapping.items():
        require(sha(REPO / relative) == package["files"][member], "Bundle source mismatch: " + member)

    contract = read(root / "contract.json")
    require(len(contract["expected_initial_fingerprint"]["records"]) == 187, "Initial state schema")
    require(len(contract["train_ids"]) == 50 and len(contract["test_ids"]) == 10, "Split sizes")
    require(set(contract["train_ids"]).isdisjoint(contract["test_ids"]), "TRAIN/TEST overlap")
    require(contract["selected_epoch_fixed_before_new_training"] == 60, "Epoch decision")

    sources = [p for folder in ("scripts", "experiment", "preparation_reference")
               for p in (REPO / folder).rglob("*.py")]
    for path in sources:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        compile(tree, str(path), "exec")

    return {
        "passed": True,
        "original_scientific_manifest_files": len(original["files"]),
        "preserved_files_verified": len(copies),
        "bundle_source_files_verified": len(mapping),
        "python_files_syntax_checked": len(sources),
        "scientific_source_unchanged": True,
        "contract_SHA256": sha(root / "contract.json"),
        "original_source_manifest_SHA256": sha(root / "source_manifest.json"),
        "numerical_model_or_training_executed": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, help="Optional new JSON receipt")
    args = parser.parse_args()
    result = verify()
    if args.out:
        with args.out.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
