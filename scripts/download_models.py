"""Download the public release, safely extract it, and verify every model file."""

import argparse
import hashlib
import json
import shutil
import stat
import sys
import time
import urllib.error
import urllib.request
import uuid
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify_models(folder=None):
    folder = Path(folder or ROOT / "models")
    expected = json.loads((ROOT / "scripts/model_manifest.json").read_text())
    problems = []
    for name, spec in expected.items():
        path = (folder / name).resolve()
        if not path.is_relative_to(folder.resolve()) or not path.is_file():
            problems.append(name + ": missing")
            continue
        if path.stat().st_size != spec["bytes"] or sha256(path) != spec["sha256"]:
            problems.append(name + ": checksum mismatch")
    return problems


def install_models(archive_path=None):
    release = json.loads((ROOT / "model_release.json").read_text())
    if archive_path is not None:
        supplied = Path(archive_path).expanduser().resolve()
        if not supplied.is_file():
            raise RuntimeError("Archive not found: " + str(supplied))
        if (
            supplied.stat().st_size != release["bytes"]
            or sha256(supplied) != release["sha256"]
        ):
            raise RuntimeError(
                "The supplied archive does not match the published Release size and SHA-256. Download the exact Portable_Lung_Model.zip from the repository Releases page."
            )
        cache = ROOT / ".downloads"
        cache.mkdir(exist_ok=True)
        cached = cache / "Portable_Lung_Model.zip"
        if supplied != cached.resolve():
            copied = cache / "manual-archive.tmp"
            shutil.copy2(supplied, copied)
            copied.replace(cached)
        print(
            "Supplied archive size and SHA-256 verified. No network download is needed."
        )
    if not verify_models():
        print("Model files are already verified.")
        return
    cache = ROOT / ".downloads"
    cache.mkdir(exist_ok=True)
    archive = cache / "Portable_Lung_Model.zip"
    partial = archive.with_suffix(".zip.part")
    if partial.exists() and partial.stat().st_size >= release["bytes"]:
        if (
            partial.stat().st_size == release["bytes"]
            and sha256(partial) == release["sha256"]
        ):
            partial.replace(archive)
        else:
            partial.unlink()
    if not archive.exists() or sha256(archive) != release["sha256"]:
        if shutil.disk_usage(ROOT).free < release["bytes"] * 2 + 100 * 1024**2:
            raise RuntimeError(
                "Free at least 1 GB of disk space for the model download and extraction."
            )
        for attempt in range(3):
            try:
                offset = partial.stat().st_size if partial.exists() else 0
                headers = {"User-Agent": "ReadingRoom-Setup/1.0"}
                if offset:
                    headers["Range"] = "bytes=" + str(offset) + "-"
                request = urllib.request.Request(release["url"], headers=headers)
                print(
                    "Downloading public model release (about 326 MB). This happens once.",
                    flush=True,
                )
                with urllib.request.urlopen(request, timeout=60) as response:
                    append = offset > 0 and response.status == 206
                    if not append:
                        offset = 0
                    last = offset
                    with partial.open("ab" if append else "wb") as out:
                        while True:
                            block = response.read(1024 * 1024)
                            if not block:
                                break
                            out.write(block)
                            offset += len(block)
                            if offset - last >= 10 * 1024**2:
                                print(
                                    f"  {offset / 1024**2:.0f} / {release['bytes'] / 1024**2:.0f} MB",
                                    flush=True,
                                )
                                last = offset
                if sha256(partial) != release["sha256"]:
                    partial.unlink()
                    raise RuntimeError(
                        "Downloaded file failed its checksum; retrying a clean download."
                    )
                partial.replace(archive)
                break
            except urllib.error.HTTPError as exc:
                if exc.code == 404:
                    raise RuntimeError(
                        "The public model release is unavailable. Check the repository Releases page or try again later."
                    ) from exc
                if attempt == 2:
                    raise RuntimeError(
                        "Could not download the model. Check your internet connection and rerun setup; partial downloads resume."
                    ) from exc
                time.sleep(2)
            except (OSError, RuntimeError) as exc:
                if attempt == 2:
                    raise RuntimeError(
                        "Model download did not finish. Rerun setup to retry or resume. "
                        + str(exc)
                    ) from exc
                time.sleep(2)
    temporary = cache / ("extract-" + uuid.uuid4().hex)
    temporary.mkdir()
    try:
        with zipfile.ZipFile(archive) as zipped:
            if (
                len(zipped.infolist()) > 500
                or sum(i.file_size for i in zipped.infolist()) > 2 * 1024**3
            ):
                raise RuntimeError("Unexpected model archive size.")
            for info in zipped.infolist():
                target = (temporary / info.filename).resolve()
                mode = info.external_attr >> 16
                if (
                    "\\" in info.filename
                    or not target.is_relative_to(temporary.resolve())
                    or stat.S_ISLNK(mode)
                ):
                    raise RuntimeError("Unsafe path in model archive.")
            zipped.extractall(temporary)
        problems = verify_models(temporary)
        if problems:
            raise RuntimeError(
                "Extracted model verification failed: " + ", ".join(problems[:3])
            )
        destination = ROOT / "models"
        if destination.exists():
            backup = ROOT / ("models.previous-" + uuid.uuid4().hex[:8])
            destination.rename(backup)
            print("Previous model folder preserved at " + backup.name)
        temporary.rename(destination)
        print("All model files verified. The demo can now run offline.")
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    flags = parser.add_mutually_exclusive_group()
    flags.add_argument(
        "--archive",
        type=Path,
        metavar="PATH",
        help="Install a downloaded Release ZIP after size/SHA-256 verification; no HTTP download.",
    )
    flags.add_argument(
        "--verify-only",
        action="store_true",
        help="Check installed model files without downloading or modifying anything.",
    )
    args = parser.parse_args()
    if args.verify_only:
        problems = verify_models()
        if problems:
            print(
                "Model verification failed: "
                + str(len(problems))
                + " missing or changed file(s).",
                file=sys.stderr,
            )
            for problem in problems[:5]:
                print("  " + problem, file=sys.stderr)
            print(
                "Run setup_demo.py or pass --archive PATH to install the public model.",
                file=sys.stderr,
            )
            return 1
        print(
            "All installed model files match the pinned manifest. No download performed."
        )
        return 0
    install_models(args.archive)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001 - friendly errors at the CLI boundary
        print("Setup could not finish: " + str(exc), file=sys.stderr)
        raise SystemExit(1)
