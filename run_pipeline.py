#!/usr/bin/env python3
"""
Replication Runner: Asymmetric effects of real exchange rate movements
on domestic value added in Mexican exports (Acosta Bautista, 2026).

Usage:
  python run_pipeline.py             # Run default replication (Stages 2 & 3)
  python run_pipeline.py --stage all # Run entire pipeline including API extraction
  python run_pipeline.py --check-env # Verify Python environment and dependencies
  python run_pipeline.py --summary   # Print summary of core empirical findings
"""

import sys
import argparse
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

def check_environment():
    """Verify required Python packages are installed."""
    print("[*] Checking Python environment...")
    required = ["torch", "statsmodels", "sklearn", "scipy", "pandas", "numpy", "matplotlib"]
    missing = []
    for pkg in required:
        try:
            __import__(pkg)
            print(f"  [✓] {pkg}")
        except ImportError:
            missing.append(pkg)
            print(f"  [✗] {pkg} (MISSING)")
    if missing:
        print(f"\n[!] Please install missing packages: pip install -r requirements.txt")
        return False
    print("\n[+] All core dependencies are installed and verified.")
    return True

def show_summary():
    """Display core empirical findings from the thesis."""
    print("=" * 70)
    print("CORE EMPIRICAL FINDINGS: NARDL ESTIMATION (2012–2022)")
    print("=" * 70)
    print("  Long-Run Multipliers:")
    print("    * Real Appreciation (REER+): +1.1002 (p < 0.0001)")
    print("    * Real Depreciation (REER-): +0.3846 (p = 0.0106)")
    print("    * Asymmetry Ratio:           2.86 : 1 (~3:1)")
    print("\n  Statistical Tests:")
    print("    * Wald Test for Symmetry:    F = 28.171 (p = 5.06e-06) -> REJECT SYMMETRY")
    print("    * Pesaran Bounds Test:       F = 10.344 (p = 4.10e-05) -> COINTEGRATION CONFIRMED")
    print("    * Error Correction (Lambda): -1.1170    (p < 0.0001)")
    print("\n  Economic Mechanism:")
    print("    Imported intermediate inputs shared in GVCs and Dominant Currency")
    print("    Pricing (DCP) generate cost-push pressures offsetting standard")
    print("    expenditure-switching benefits of currency depreciation.")
    print("=" * 70)

def run_script(script_path, description):
    print(f"\n========================================================")
    print(f"[*] Running: {description}")
    print(f"[*] Script:  {script_path.relative_to(BASE_DIR)}")
    print(f"========================================================")
    result = subprocess.run([sys.executable, str(script_path)], cwd=BASE_DIR)
    if result.returncode != 0:
        print(f"[!] Warning: Script exited with code {result.returncode}")
    else:
        print(f"[+] Completed successfully: {description}")

def main():
    parser = argparse.ArgumentParser(
        description="Replication Pipeline for CIDE Master's Thesis (Acosta Bautista, 2026)"
    )
    parser.add_argument(
        "--stage",
        choices=["1", "2", "3", "all", "default"],
        default="default",
        help="Specify which stage to run (default: Stages 2 and 3)"
    )
    parser.add_argument("--check-env", action="store_true", help="Check installed dependencies")
    parser.add_argument("--summary", action="store_true", help="Display core econometric summary")
    
    args = parser.parse_args()

    if args.check_env:
        check_environment()
        return

    if args.summary:
        show_summary()
        return

    print("===================================================================")
    print("Replication Package: Mexican Manufacturing DVA & Asymmetric REER")
    print("Author: Bertin Yair Acosta Bautista (CIDE, 2026)")
    print("Repository Handle: https://repositorio-digital.cide.edu/handle/11651/6699")
    print("===================================================================")

    stage_map = {
        "1": [("scripts/stage1_data_acquisition/data_extraction_mt.py", "Stage 1: API Data Acquisition")],
        "2": [("scripts/stage2_ann_midas/ann_midas.py", "Stage 2: ANN-MIDAS Temporal Disaggregation")],
        "3": [
            ("scripts/stage3_econometrics/nardl.py", "Stage 3: Non-Linear ARDL Estimation"),
            ("scripts/stage3_econometrics/toda_yamamoto.py", "Stage 3: Toda-Yamamoto Causality Analysis")
        ]
    }

    if args.stage == "all":
        to_run = stage_map["1"] + stage_map["2"] + stage_map["3"]
    elif args.stage == "default":
        to_run = stage_map["2"] + stage_map["3"]
    else:
        to_run = stage_map[args.stage]

    for script_rel, desc in to_run:
        script_path = BASE_DIR / script_rel
        if script_path.exists():
            run_script(script_path, desc)
        else:
            print(f"[!] File not found: {script_rel}")

    print("\n[+] Execution finished. Outputs saved in 'results/'.")

if __name__ == "__main__":
    main()
