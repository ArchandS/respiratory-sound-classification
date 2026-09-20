"""
Preprocess the ICBHI 2017 database into a single .npz feature archive.

PATHS USED BY THIS SCRIPT (all set at the top of src/config.py):
    reads  <- DATASET_DIR       your ICBHI .wav + .txt folder   *** SET THIS ***
    writes -> PROCESSED_FILE    the feature archive (.npz), ~1-3 GB

Run once, before any training:
    python -m src.verify_setup     # check paths first
    python -m src.preprocess

WHAT CHANGED vs. the original pipeline, and why
-----------------------------------------------
1. PATIENT IDs ARE NOW SAVED.
   The original script discarded the filename, which made a patient-wise split
   impossible. Cycles from one patient could land in both train and validation,
   letting the model score by recognising a patient's microphone and room tone
   instead of their crackles. We now store patient_ids and filenames so
   `splits.py` can group by patient.

2. MEL/MFCC FILTERBANKS ARE BAND-LIMITED (fmin=100, fmax=2000).
   We bandpass the audio to 100-2000 Hz but the original code then extracted
   MFCCs across the full 0-8000 Hz range. Most filterbanks were measuring the
   silence we had just created. Restricting fmin/fmax concentrates all the
   frequency resolution on the band that actually carries crackles and wheezes.

3. SHORTER CLIPS + VALIDITY MASKS.
   Typical cycles are ~2-3s; padding to 8s meant most frames were zeros. We use
   a 4s window and additionally record how many frames are real, so models can
   mask out padding rather than processing it as signal.

4. DELTA AND DELTA-DELTA MFCCs.
   Crackles are transient events - defined by how the spectrum *changes*, not
   its static value. First and second derivatives make that explicit rather than
   asking the CNN to infer it.

5. FEATURES ARE SAVED UNNORMALISED, ON PURPOSE.
   Normalisation statistics must be computed from the TRAINING SPLIT ONLY,
   otherwise validation information leaks into training. Since the split is
   decided later, normalisation happens at train time in `dataset.py`.

6. LOG-MEL USES A FIXED dB REFERENCE.
   The original used `ref=np.max`, normalising each clip to its own loudest
   point and discarding absolute intensity. We use a fixed reference so loud
   and quiet recordings stay distinguishable.
"""

from __future__ import annotations

import glob
import os
import sys
from pathlib import Path

import librosa
import numpy as np
import scipy.signal as sps
from tqdm import tqdm

from .config import get_config


def build_bandpass(sr: int, lowcut: float, highcut: float, order: int):
    nyq = 0.5 * sr
    low = max(lowcut / nyq, 1e-6)
    high = min(highcut / nyq, 0.999)
    return sps.butter(order, [low, high], btype="bandpass", output="sos")


def parse_patient_id(wav_path: str) -> int:
    """
    ICBHI filenames look like:  101_1b1_Al_sc_Meditron.wav
    The leading number is the patient ID.
    """
    stem = Path(wav_path).stem
    return int(stem.split("_")[0])


def label_from_flags(crackle: int, wheeze: int) -> int:
    if crackle == 0 and wheeze == 0:
        return 0  # Normal
    if crackle == 1 and wheeze == 0:
        return 1  # Crackle
    if crackle == 0 and wheeze == 1:
        return 2  # Wheeze
    return 3      # Both


def fix_length(audio: np.ndarray, max_samples: int):
    """
    Pad or crop to a fixed length.

    Returns (fixed_audio, n_valid_samples) so downstream code knows how much of
    the clip is real signal rather than padding.
    """
    n_valid = min(len(audio), max_samples)
    if len(audio) < max_samples:
        audio = np.pad(audio, (0, max_samples - len(audio)), mode="constant")
    else:
        # Centre crop: a long cycle's informative part is rarely at the very start
        start = (len(audio) - max_samples) // 2
        audio = audio[start:start + max_samples]
    return audio, n_valid


def process_dataset():
    cfg = get_config()
    audio_cfg = cfg.audio
    dataset_dir = Path(cfg.paths.dataset_dir)

    if not dataset_dir.exists():
        sys.exit(
            f"ERROR: dataset directory not found: {dataset_dir}\n"
            f"Edit Paths.dataset_dir in src/config.py to point at your "
            f"ICBHI_final_database folder."
        )

    txt_files = sorted(glob.glob(str(dataset_dir / "*.txt")))
    # The official split list is not an annotation file - skip it
    txt_files = [t for t in txt_files
                 if "train_test" not in Path(t).name.lower()]

    if not txt_files:
        sys.exit(f"ERROR: no annotation .txt files found in {dataset_dir}")

    sos = build_bandpass(audio_cfg.target_sr, audio_cfg.lowcut,
                         audio_cfg.highcut, audio_cfg.filter_order)

    mfccs, mels, labels, patients, files, valid_frames = [], [], [], [], [], []
    skipped = 0

    print(f"Found {len(txt_files)} annotation files. Preprocessing...")
    print(f"  clip length : {audio_cfg.duration}s "
          f"({audio_cfg.max_samples} samples @ {audio_cfg.target_sr} Hz)")
    print(f"  band        : {audio_cfg.fmin}-{audio_cfg.fmax} Hz")
    print(f"  signal feats: {audio_cfg.n_mfcc} MFCC x3 (+delta,+delta2) "
          f"= {audio_cfg.signal_channels} channels")
    print(f"  image feats : {audio_cfg.n_mels} mel bands")

    for txt_path in tqdm(txt_files):
        wav_path = txt_path[:-4] + ".wav"
        if not os.path.exists(wav_path):
            continue

        try:
            y, sr = librosa.load(wav_path, sr=audio_cfg.target_sr, mono=True)
        except Exception as e:
            print(f"  ! could not load {Path(wav_path).name}: {e}")
            continue

        if len(y) < audio_cfg.target_sr * 0.1:
            continue

        y = sps.sosfiltfilt(sos, y).astype(np.float32)

        # Per-recording peak normalisation. Different recordings used different
        # microphones at different gains; this removes that nuisance variation
        # while preserving relative dynamics WITHIN a recording.
        peak = np.max(np.abs(y))
        if peak > 0:
            y = y / peak

        patient_id = parse_patient_id(wav_path)
        fname = Path(wav_path).name

        with open(txt_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            try:
                start_sec, end_sec = float(parts[0]), float(parts[1])
                crackle, wheeze = int(parts[2]), int(parts[3])
            except ValueError:
                continue

            if end_sec <= start_sec:
                skipped += 1
                continue

            s = int(start_sec * sr)
            e = int(end_sec * sr)
            cycle = y[s:e]

            if len(cycle) < int(0.1 * sr):   # shorter than 100ms -> unusable
                skipped += 1
                continue

            cycle, n_valid_samples = fix_length(cycle, audio_cfg.max_samples)

            # ---- Image branch: log-Mel spectrogram ----
            mel = librosa.feature.melspectrogram(
                y=cycle, sr=sr,
                n_fft=audio_cfg.n_fft, hop_length=audio_cfg.hop_length,
                n_mels=audio_cfg.n_mels,
                fmin=audio_cfg.fmin, fmax=audio_cfg.fmax,
                power=2.0,
            )
            # Fixed reference (1.0) rather than ref=np.max - keeps absolute level
            log_mel = librosa.power_to_db(mel, ref=1.0, top_db=None)

            # ---- Signal branch: MFCC + deltas ----
            mf = librosa.feature.mfcc(
                S=log_mel, sr=sr, n_mfcc=audio_cfg.n_mfcc
            )
            d1 = librosa.feature.delta(mf, order=1)
            d2 = librosa.feature.delta(mf, order=2)
            sig = np.concatenate([mf, d1, d2], axis=0)

            n_valid_fr = int(np.ceil(n_valid_samples / audio_cfg.hop_length))
            n_valid_fr = max(1, min(n_valid_fr, log_mel.shape[1]))

            mfccs.append(sig.astype(np.float32))
            mels.append(log_mel.astype(np.float32))
            labels.append(label_from_flags(crackle, wheeze))
            patients.append(patient_id)
            files.append(fname)
            valid_frames.append(n_valid_fr)

    X_sig = np.stack(mfccs)
    X_img = np.stack(mels)
    Y = np.array(labels, dtype=np.int64)
    P = np.array(patients, dtype=np.int64)
    F = np.array(files)
    V = np.array(valid_frames, dtype=np.int64)

    print("\nPreprocessing complete.")
    print(f"  cycles extracted : {len(Y)}  (skipped {skipped} malformed)")
    print(f"  unique patients  : {len(np.unique(P))}")
    print(f"  signal features  : {X_sig.shape}")
    print(f"  image features   : {X_img.shape}")
    uniq, cnt = np.unique(Y, return_counts=True)
    for u, c in zip(uniq, cnt):
        print(f"    class {u} ({get_config().train.class_names[u]:<8}): "
              f"{c:5d}  ({100*c/len(Y):.1f}%)")

    out = Path(cfg.paths.processed_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        signal_features=X_sig,
        image_features=X_img,
        labels=Y,
        patient_ids=P,
        filenames=F,
        valid_frames=V,
    )
    print(f"\nSaved -> {out.resolve()}")


if __name__ == "__main__":
    process_dataset()
