"""
Pre-flight check. RUN THIS FIRST.

    python -m src.verify_setup

Catches the mistakes that otherwise surface 20 minutes into preprocessing or
half an hour into training:
  * DATASET_DIR pointing at the wrong folder (or at a parent of the real one)
  * missing Python packages
  * .wav files with no matching .txt (or vice versa)
  * annotation files that parse to zero usable cycles
  * the official split list being absent (which silently changes your numbers)
  * not enough free disk space for the feature archive

Exits with code 0 if you are good to go, 1 if something must be fixed.
"""

from __future__ import annotations

import glob
import shutil
import sys
from pathlib import Path

OK = "  [ OK ]"
BAD = "  [FAIL]"
WARN = "  [WARN]"


def _hr(title):
    print(f"\n{title}\n" + "-" * 66)


def check_packages():
    _hr("1. Python packages")
    required = {
        "numpy": "numpy",
        "scipy": "scipy",
        "sklearn": "scikit-learn",
        "torch": "torch",
        "librosa": "librosa",
        "tqdm": "tqdm",
    }
    optional = {"torchvision": "torchvision (only for --backbone resnet34)"}

    missing = []
    for mod, pipname in required.items():
        try:
            m = __import__(mod)
            ver = getattr(m, "__version__", "?")
            print(f"{OK} {pipname:<14} {ver}")
        except ImportError:
            print(f"{BAD} {pipname:<14} NOT INSTALLED")
            missing.append(pipname)

    for mod, label in optional.items():
        try:
            m = __import__(mod)
            print(f"{OK} {label:<14} {getattr(m, '__version__', '?')}")
        except ImportError:
            print(f"{WARN} {label} not installed (optional)")

    if missing:
        print(f"\n  Fix with:  pip install {' '.join(missing)}")
        return False

    try:
        import torch
        if torch.cuda.is_available():
            print(f"{OK} CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print(f"{WARN} No CUDA detected — training will run on CPU "
                  f"(expect many hours). This is not an error.")
    except Exception:
        pass
    return True


def check_dataset(cfg):
    _hr("2. Dataset folder (PATH 1 in src/config.py)")
    d = Path(cfg.paths.dataset_dir)
    print(f"  DATASET_DIR = {d}")

    if not d.exists():
        print(f"{BAD} This folder does not exist.")
        print(f"\n  Open src/config.py and set DATASET_DIR to the folder that")
        print(f"  holds your ICBHI .wav and .txt files.")
        print(f"  On Windows remember the r prefix:  r\"E:\\ICBHI_final_database\"")
        return False, 0
    print(f"{OK} Folder exists")

    wavs = glob.glob(str(d / "*.wav"))
    txts = [t for t in glob.glob(str(d / "*.txt"))
            if "train_test" not in Path(t).name.lower()]

    print(f"  .wav files found : {len(wavs)}")
    print(f"  .txt files found : {len(txts)}   (excluding the split list)")

    if not wavs and not txts:
        print(f"{BAD} No audio or annotation files directly inside this folder.")
        subs = [p for p in d.iterdir() if p.is_dir()] if d.exists() else []
        hits = [s for s in subs if list(s.glob("*.wav"))]
        if hits:
            print(f"\n  But .wav files ARE present in a subfolder. "
                  f"Point DATASET_DIR at one of these instead:")
            for s in hits[:5]:
                print(f"      {s}")
        return False, 0

    if not wavs:
        print(f"{BAD} Annotation files present but no .wav files.")
        return False, 0
    if not txts:
        print(f"{BAD} Audio files present but no .txt annotations.")
        return False, 0

    wav_stems = {Path(w).stem for w in wavs}
    txt_stems = {Path(t).stem for t in txts}
    paired = wav_stems & txt_stems

    print(f"  matched pairs    : {len(paired)}")
    if len(paired) == 0:
        print(f"{BAD} No .wav/.txt pairs share a filename.")
        return False, 0

    orphan_wav = len(wav_stems - txt_stems)
    orphan_txt = len(txt_stems - wav_stems)
    if orphan_wav or orphan_txt:
        print(f"{WARN} {orphan_wav} .wav without .txt, "
              f"{orphan_txt} .txt without .wav — these will be skipped.")

    if len(paired) < 900:
        print(f"{WARN} The full ICBHI database has 920 recordings; "
              f"you have {len(paired)}. Partial download?")
    else:
        print(f"{OK} Recording count looks right")

    return True, len(paired)


def check_annotations(cfg):
    _hr("3. Annotation parsing")
    d = Path(cfg.paths.dataset_dir)
    txts = [t for t in glob.glob(str(d / "*.txt"))
            if "train_test" not in Path(t).name.lower()]
    if not txts:
        print(f"{BAD} No annotation files to check.")
        return False

    total, bad_lines, counts = 0, 0, [0, 0, 0, 0]
    patients = set()

    for t in txts[:120]:                # sample is enough to validate format
        try:
            patients.add(int(Path(t).stem.split("_")[0]))
        except ValueError:
            pass
        with open(t) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 4:
                    if line.strip():
                        bad_lines += 1
                    continue
                try:
                    s, e = float(parts[0]), float(parts[1])
                    c, w = int(parts[2]), int(parts[3])
                except ValueError:
                    bad_lines += 1
                    continue
                if e <= s:
                    bad_lines += 1
                    continue
                total += 1
                counts[(0 if (c == 0 and w == 0) else
                        1 if (c == 1 and w == 0) else
                        2 if (c == 0 and w == 1) else 3)] += 1

    print(f"  sampled {min(len(txts),120)} annotation files")
    print(f"  parsed cycles : {total}   (malformed lines: {bad_lines})")

    if total == 0:
        print(f"{BAD} Zero cycles parsed. The annotation format is not what "
              f"this pipeline expects (4 whitespace-separated columns: "
              f"start, end, crackle, wheeze).")
        return False

    names = ["Normal", "Crackle", "Wheeze", "Both"]
    for i, n in enumerate(names):
        pct = 100 * counts[i] / total
        print(f"    {n:<8}: {counts[i]:5d}  ({pct:5.1f}%)")
    print(f"  patient IDs parsed from filenames: {len(patients)}")

    if len(patients) < 2:
        print(f"{BAD} Could not parse patient IDs. Filenames must start with "
              f"the patient number, e.g. 101_1b1_Al_sc_Meditron.wav")
        return False

    print(f"{OK} Annotations parse correctly")
    return True


def check_official_split(cfg):
    _hr("4. Official ICBHI split list")
    f = Path(cfg.paths.official_split_file)
    print(f"  looking for: {f}")
    if f.exists():
        n = sum(1 for line in open(f) if line.strip())
        print(f"{OK} Found ({n} entries). The official patient-wise split will "
              f"be used, so your results are comparable to published papers.")
    else:
        print(f"{WARN} Not found. The pipeline will generate its own "
              f"patient-wise split instead.")
        print(f"       That is still leak-free and perfectly valid, but your")
        print(f"       numbers will NOT be directly comparable to published")
        print(f"       ICBHI results. The file ships with the dataset as")
        print(f"       'ICBHI_challenge_train_test.txt'.")
    return True


def check_output_paths(cfg, n_recordings):
    _hr("5. Output paths and disk space (PATHS 2 and 3 in src/config.py)")
    proc = Path(cfg.paths.processed_file)
    out = Path(cfg.paths.output_dir)
    print(f"  PROCESSED_FILE = {proc}")
    print(f"  OUTPUT_DIR     = {out}")

    ok = True
    for label, p in (("PROCESSED_FILE", proc.parent), ("OUTPUT_DIR", out)):
        try:
            p.mkdir(parents=True, exist_ok=True)
            probe = p / ".write_test"
            probe.write_text("x")
            probe.unlink()
            print(f"{OK} {label} directory is writable: {p.resolve()}")
        except Exception as e:
            print(f"{BAD} Cannot write to {p}: {e}")
            ok = False

    if proc.exists():
        gb = proc.stat().st_size / 1e9
        print(f"{WARN} {proc} already exists ({gb:.2f} GB). "
              f"Re-running preprocess will overwrite it.")

    try:
        free_gb = shutil.disk_usage(proc.parent.resolve()).free / 1e9
        # ~6900 cycles x (128 mel + 120 mfcc) x ~251 frames x 4 bytes,
        # before compression
        est = 6900 * 248 * 251 * 4 / 1e9
        print(f"  free space: {free_gb:.1f} GB "
              f"(feature archive needs roughly {est:.1f} GB uncompressed)")
        if free_gb < est:
            print(f"{BAD} Not enough free space. Point PROCESSED_FILE at a "
                  f"drive with more room.")
            ok = False
        else:
            print(f"{OK} Sufficient disk space")
    except Exception:
        pass

    return ok


def main():
    print("=" * 66)
    print("  ICBHI DUAL-BRANCH — SETUP VERIFICATION")
    print("=" * 66)

    try:
        from .config import get_config
    except ImportError:
        print("\nRun this as a module from the repository root:")
        print("    python -m src.verify_setup")
        sys.exit(1)

    cfg = get_config()
    results = []

    results.append(check_packages())
    ds_ok, n_rec = check_dataset(cfg)
    results.append(ds_ok)

    if ds_ok:
        results.append(check_annotations(cfg))
        check_official_split(cfg)
    results.append(check_output_paths(cfg, n_rec))

    print("\n" + "=" * 66)
    if all(results):
        print("  ALL CHECKS PASSED — you are ready to run:")
        print("      python -m src.preprocess")
        print("      python -m src.train --branch signal")
        print("      python -m src.train --branch image")
        print("      python -m src.ensemble")
        print("=" * 66)
        sys.exit(0)
    else:
        print("  SOME CHECKS FAILED — fix the [FAIL] items above first.")
        print("  Paths are set at the top of src/config.py.")
        print("=" * 66)
        sys.exit(1)


if __name__ == "__main__":
    main()
