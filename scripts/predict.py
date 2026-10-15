#!/usr/bin/env python3
"""Verify this exact full model bundle and call its unchanged RGB predictor."""
from pathlib import Path, PurePosixPath
import argparse
import datetime
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile

from verify_export import REPO, read, require, sha, verify


def validate_cache(cache, manifest):
    require(cache.is_dir() and not cache.is_symlink(), "Cache must be a regular directory")
    actual = read(cache / "package_manifest.json")
    require(actual == manifest, "Cached package manifest differs from this version")
    for relative, expected in manifest["files"].items():
        path = cache / relative
        require(path.is_file() and not path.is_symlink(), "Missing/linked bundle member: " + relative)
        require(sha(path) == expected, "Cached member SHA256 mismatch: " + relative)


def unpack(bundle, cache):
    expected = read(REPO / "provenance/artifacts.json")["artifacts"]["complete_joint_IDM_model.zip"]
    require(bundle.is_file(), "Model bundle does not exist")
    require(bundle.stat().st_size == expected["bytes"], "Model bundle size mismatch")
    require(sha(bundle) == expected["SHA256"], "Model bundle SHA256 mismatch")
    manifest = read(REPO / "provenance/model_package_manifest.json")
    if cache.exists():
        validate_cache(cache, manifest)
        return
    with zipfile.ZipFile(bundle) as archive:
        members = archive.infolist()
        names = [item.filename for item in members]
        require(len(names) == len(set(names)), "Duplicate ZIP members")
        require(set(names) == set(manifest["files"]) | {"package_manifest.json"}, "Unexpected ZIP members")
        for item in members:
            path = PurePosixPath(item.filename)
            require(not path.is_absolute() and ".." not in path.parts, "Unsafe ZIP path")
            require(not item.is_dir() and item.file_size <= 100 * 1024 * 1024, "Unexpected ZIP entry")
        require(json.loads(archive.read("package_manifest.json")) == manifest, "ZIP manifest mismatch")
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.mkdir(exist_ok=False)
        for item in members:
            destination = cache / item.filename
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(item) as source, destination.open("xb") as output:
                shutil.copyfileobj(source, output, length=8 * 1024 * 1024)
    validate_cache(cache, manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-bundle", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--rgb", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--unpack-only", action="store_true")
    args = parser.parse_args()
    require(sys.platform.startswith("linux"), "Run on the Linux experiment server")
    verify()
    bundle = args.model_bundle.resolve()
    # Preserve the supplied path for symlink checks.
    cache = args.cache_dir.absolute()
    if not args.unpack_only:
        require(args.rgb is not None and args.rgb.is_file(), "--rgb must name an RGB NPY file")
        require(args.out is not None and args.out.suffix == ".npz", "--out must end in .npz")
        require(not args.out.exists() and not args.out.with_suffix(".receipt.json").exists(), "Output already exists")
        require(re.fullmatch(r"cuda:[0-9]+", args.device) is not None, "Use a CUDA device, e.g. cuda:0")
    unpack(bundle, cache)
    if args.unpack_only:
        print(json.dumps({"status": "verified_unpacked", "model_bundle_SHA256": sha(bundle),
                          "members_verified": 11, "model_loaded": False, "GPU_used": False}))
        return

    args.out.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    command = [sys.executable, str(cache / "public_predict.py"), "--rgb", str(args.rgb.resolve()),
               "--out", str(args.out.resolve()), "--device", args.device]
    subprocess.run(command, check=True, env=env)
    require(args.out.is_file(), "Predictor exited without the expected output")
    receipt = {
        "status": "complete", "UTC": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "RGB_SHA256": sha(args.rgb), "prediction_SHA256": sha(args.out),
        "model_bundle_SHA256": sha(bundle), "device": args.device,
        "unchanged_public_predict_SHA256": sha(cache / "public_predict.py"),
        "GT_input": False, "command": command,
    }
    with args.out.with_suffix(".receipt.json").open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
