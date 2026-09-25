#!/usr/bin/env python3
"""
Replication Runner: Asymmetric effects of real exchange rate movements
on domestic value added in Mexican exports (Acosta Bautista, 2026).
"""

import sys
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

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
    print("===================================================================")
    print("Replication Package: Mexican Manufacturing DVA & Asymmetric REER")
    print("Author: Bertin Yair Acosta Bautista (CIDE, 2026)")
    print("Repository Handle: https://repositorio-digital.cide.edu/handle/11651/6699")
    print("===================================================================")
    
    stages = [
        ("scripts/stage2_ann_midas/ann_midas.py", "Stage 2: ANN-MIDAS Temporal Disaggregation"),
        ("scripts/stage3_econometrics/nardl.py", "Stage 3: Non-Linear ARDL Estimation & Multipliers"),
        ("scripts/stage3_econometrics/toda_yamamoto.py", "Stage 3: Toda-Yamamoto Granger Causality Analysis")
    ]
    
    for script_rel, desc in stages:
        script_path = BASE_DIR / script_rel
        if script_path.exists():
            run_script(script_path, desc)
        else:
            print(f"[!] Not found: {script_rel}")

    print("\n[+] Full replication execution finished. Check 'results/' for generated figures and tables.")

if __name__ == "__main__":
    main()
