"""Command-line entry point for the reproducible Spotify popularity study."""

import argparse
import hashlib
import importlib.metadata
import os
from pathlib import Path
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parent
EXPECTED_HASH = "8a08c199bd93438783e915470ec3ea1142c1267f2da48b7d1315d2b9f5d39cf9"
STAGES = {"audit": "01_audit.py", "train": "02_train.py", "stability": "03_stability.py"}
PRIOR_FILES = (
    "split_manifest.csv.gz", "experiment_metadata.json", "test_results.json",
    "split_summary.json", "overlap_checks.json",
)


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def check_environment():
    required = {"numpy": "2.3.5", "pandas": "2.3.3", "scikit-learn": "1.7.2",
                "matplotlib": "3.10.6", "joblib": "1.5.2"}
    missing = []
    for name, expected in required.items():
        try:
            actual = importlib.metadata.version(name)
            print(f"{name}: {actual}", flush=True)
            if actual != expected:
                print(f"  Note: reported experiments used {expected}.", flush=True)
        except importlib.metadata.PackageNotFoundError:
            missing.append(name)
    if missing:
        raise ValueError("Missing dependencies: " + ", ".join(missing)
                         + ". Run: python -m pip install -r requirements.txt")


def check_data(path, require_original=True):
    if not path.is_file():
        raise ValueError(f"Dataset not found: {path}\nDownload the source archive using "
                         "data/README.md, then supply --data /path/to/archive.zip.")
    try:
        with zipfile.ZipFile(path) as archive:
            with archive.open("tracks.csv") as source:
                digest = hashlib.file_digest(source, "sha256").hexdigest()
    except (zipfile.BadZipFile, KeyError) as error:
        raise ValueError("Expected a ZIP archive with tracks.csv at its root.") from error
    print(f"tracks.csv SHA-256: {digest}", flush=True)
    if digest != EXPECTED_HASH and require_original:
        raise ValueError("This dataset differs from the reported experiment. Use the original "
                         "archive, or run the audit command to inspect the new data. "
                         "Training intentionally requires the original source.")
    if digest != EXPECTED_HASH:
        print("This is a different dataset version; recorded project results do not apply.", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["check", "audit", "train", "stability", "all"],
                        help="Check setup, run one analysis stage, or run all stages in order")
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "archive.zip",
                        help="ZIP containing tracks.csv (default: data/archive.zip)")
    parser.add_argument("--output", type=Path, default=ROOT / "runs",
                        help="Directory for new results (default: runs/; matching files are replaced)")
    parser.add_argument("--jobs", type=positive_int, default=2,
                        help="Parallel random-forest workers (default: 2)")
    args = parser.parse_args(argv)
    data = args.data.expanduser().resolve()
    output = args.output.expanduser().resolve()
    try:
        check_environment()
        check_data(data, require_original=args.command != "audit")
        if args.command == "check":
            print("Environment and source data are ready.")
            return 0
        if args.command == "stability":
            missing = [name for name in PRIOR_FILES if not (output / "model_results" / name).is_file()]
            if missing:
                raise ValueError("Run train first with the same --output directory. Missing: "
                                 + ", ".join(missing))
        output.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, SPOTIFY_DATA_PATH=str(data), SPOTIFY_OUTPUT_DIR=str(output),
                   SPOTIFY_JOBS=str(args.jobs), PYTHONUNBUFFERED="1", MPLBACKEND="Agg")
        stages = list(STAGES) if args.command == "all" else [args.command]
        for stage in stages:
            script = ROOT / "scripts" / STAGES[stage]
            print(f"\nRunning {stage}; results: {output}", flush=True)
            result = subprocess.run([sys.executable, str(script)], cwd=ROOT, env=env)
            if result.returncode:
                return result.returncode
        print(f"\nCompleted. Results are in {output}", flush=True)
        return 0
    except (ValueError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
