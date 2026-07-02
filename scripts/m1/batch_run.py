"""
Batch runner for M1 animation generation (local).

Runs one or all animation models across all headshots in the headshots/ directory.

Usage examples:
    # Run all three models (LivePortrait requires --template)
    python scripts/m1/batch_run.py --model all --template path/to/template.mp4

    # Run a single model
    python scripts/m1/batch_run.py --model liveportrait --template path/to/template.mp4
    python scripts/m1/batch_run.py --model sadtalker
    python scripts/m1/batch_run.py --model wav2lip

    # Run on a single headshot instead of the full batch
    python scripts/m1/batch_run.py --model sadtalker --headshot headshots/person_01.png

    # Force CPU (no GPU)
    python scripts/m1/batch_run.py --model sadtalker --cpu
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
HEADSHOTS_DIR = PROJECT_ROOT / "headshots"
OUTPUTS_BASE = PROJECT_ROOT / "outputs" / "m1"

sys.path.insert(0, str(PROJECT_ROOT))


def _run_model(model: str, headshots: list[str], args: argparse.Namespace) -> list[dict]:
    results = []
    failures = []

    if model == "liveportrait":
        from scripts.m1.liveportrait import generate_animation
        output_dir = str(OUTPUTS_BASE / "liveportrait")
        for hs in headshots:
            name = Path(hs).stem
            print(f"\n[LivePortrait] {name}")
            try:
                r = generate_animation(hs, output_dir, args.template)
                results.append(r)
            except Exception as e:
                print(f"  FAILED: {e}")
                failures.append({"headshot": hs, "model": model, "error": str(e)})

    elif model == "sadtalker":
        from scripts.m1.sadtalker import generate_animation
        output_dir = str(OUTPUTS_BASE / "sadtalker")
        for hs in headshots:
            name = Path(hs).stem
            print(f"\n[SadTalker] {name}")
            try:
                r = generate_animation(
                    hs, output_dir,
                    cpu_only=args.cpu,
                    use_enhancer=not args.no_enhancer,
                )
                results.append(r)
            except Exception as e:
                print(f"  FAILED: {e}")
                failures.append({"headshot": hs, "model": model, "error": str(e)})

    elif model == "wav2lip":
        from scripts.m1.wav2lip import generate_animation
        output_dir = str(OUTPUTS_BASE / "wav2lip")
        for hs in headshots:
            name = Path(hs).stem
            print(f"\n[Wav2Lip] {name}")
            try:
                r = generate_animation(hs, output_dir)
                results.append(r)
            except Exception as e:
                print(f"  FAILED: {e}")
                failures.append({"headshot": hs, "model": model, "error": str(e)})

    return results, failures


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch generate M1 talking animations locally")
    parser.add_argument(
        "--model",
        choices=["liveportrait", "sadtalker", "wav2lip", "all"],
        required=True,
        help="Which model(s) to run",
    )
    parser.add_argument(
        "--template",
        default="",
        help="Path to driving template video (required for LivePortrait)",
    )
    parser.add_argument(
        "--headshot",
        default="",
        help="Run on a single headshot instead of the full batch",
    )
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference (SadTalker)")
    parser.add_argument("--no_enhancer", action="store_true", help="Skip GFPGAN (SadTalker)")
    args = parser.parse_args()

    # Resolve headshot list
    if args.headshot:
        headshots = [str(Path(args.headshot).resolve())]
    else:
        headshots = sorted(glob.glob(str(HEADSHOTS_DIR / "person_*.png")))

    if not headshots:
        print(f"No headshots found in {HEADSHOTS_DIR}")
        print("Expected files named person_01.png … person_08.png")
        sys.exit(1)

    print(f"Found {len(headshots)} headshot(s): {[Path(h).name for h in headshots]}")

    # LivePortrait needs a template
    models_to_run = (
        ["liveportrait", "sadtalker", "wav2lip"] if args.model == "all" else [args.model]
    )
    if "liveportrait" in models_to_run and not args.template:
        print("ERROR: --template is required for LivePortrait.")
        print("Provide a short (~5-10s) video of any person talking as the motion driver.")
        sys.exit(1)

    # Run each model
    all_results = []
    all_failures = []

    for model in models_to_run:
        print(f"\n{'='*60}")
        print(f"  Model: {model.upper()}")
        print(f"{'='*60}")
        results, failures = _run_model(model, headshots, args)
        all_results.extend(results)
        all_failures.extend(failures)

    # Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  Completed: {len(all_results)}")
    print(f"  Failed:    {len(all_failures)}")

    for r in all_results:
        size_kb = os.path.getsize(r["output_path"]) // 1024 if os.path.exists(r["output_path"]) else 0
        print(f"  [OK] {Path(r['output_path']).name}  ({size_kb} KB)  engine={r['engine']}")

    if all_failures:
        print("\nFailed:")
        for f in all_failures:
            print(f"  [FAIL] {Path(f['headshot']).name} ({f['model']}): {f['error'][:120]}")

    # Save results manifest
    manifest_path = OUTPUTS_BASE / "batch_results.json"
    with open(manifest_path, "w") as fp:
        json.dump({"completed": all_results, "failed": all_failures}, fp, indent=2)
    print(f"\nResults manifest saved to: {manifest_path}")


if __name__ == "__main__":
    main()
