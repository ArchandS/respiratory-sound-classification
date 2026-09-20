"""
Central configuration.

╔══════════════════════════════════════════════════════════════════════════╗
║                                                                          ║
║   >>>  IF YOU EDIT ONLY ONE THING IN THIS PROJECT, EDIT DATASET_DIR <<<  ║
║                                                                          ║
║   Set DATASET_DIR below to the folder holding your ICBHI .wav/.txt       ║
║   files. Everything else has a working default.                          ║
║                                                                          ║
║   Verify your setup BEFORE a long run:                                   ║
║       python -m src.verify_setup                                         ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List
import json


# ═══════════════════════════════════════════════════════════════════════════
#  ███  EDIT YOUR PATHS HERE  ███
# ═══════════════════════════════════════════════════════════════════════════

# ---------------------------------------------------------------------------
# PATH 1 of 3  —  DATASET_DIR        *** YOU MUST SET THIS ***
# ---------------------------------------------------------------------------
# The folder containing the ICBHI audio + annotation files. It should hold
# files that look like:
#
#     101_1b1_Al_sc_Meditron.wav
#     101_1b1_Al_sc_Meditron.txt
#     102_1b1_Ar_sc_Meditron.wav
#     ...
#     ICBHI_challenge_train_test.txt      <- optional, strongly recommended
#
# Windows: use a raw string (the r"" prefix) so backslashes are not treated
# as escape characters:
#     DATASET_DIR = r"E:\ICBHI_final_database"
#     DATASET_DIR = r"C:\Users\karthik\Downloads\ICBHI_final_database"
#
# Linux / macOS:
#     DATASET_DIR = "/home/user/data/ICBHI_final_database"
#
# IMPORTANT: if your .wav files live in a SUBFOLDER such as
#     ICBHI_final_database/audio_and_txt_files/
# then point at that subfolder, not its parent. This script looks for .txt
# files sitting directly inside DATASET_DIR.
#
DATASET_DIR = r"E:\ICBHI_final_database"


# ---------------------------------------------------------------------------
# PATH 2 of 3  —  PROCESSED_FILE        (default is fine)
# ---------------------------------------------------------------------------
# Where `python -m src.preprocess` writes the extracted features.
#
# This file is large (roughly 1-3 GB). If your system drive is tight on space,
# put it elsewhere, e.g.:
#     PROCESSED_FILE = r"E:\processed_icbhi.npz"
#
# The default keeps it under data/ inside the repo. .gitignore already excludes
# that folder, so you cannot accidentally commit a multi-GB file to GitHub.
#
PROCESSED_FILE = "data/processed_icbhi.npz"


# ---------------------------------------------------------------------------
# PATH 3 of 3  —  OUTPUT_DIR            (default is fine)
# ---------------------------------------------------------------------------
# Where checkpoints, reports and cached logits go. Created automatically,
# and gitignored.
#
#     outputs/checkpoints/   trained weights (.pth)
#     outputs/reports/       classification reports, confusion matrices
#     outputs/logits/        cached validation logits, consumed by the ensemble
#
OUTPUT_DIR = "outputs"

# ═══════════════════════════════════════════════════════════════════════════
#  ███  END OF PATHS — nothing below needs editing to get started  ███
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class Paths:
    dataset_dir: Path = Path(DATASET_DIR)
    processed_file: Path = Path(PROCESSED_FILE)
    output_dir: Path = Path(OUTPUT_DIR)

    @property
    def ckpt_dir(self) -> Path:
        return self.output_dir / "checkpoints"

    @property
    def report_dir(self) -> Path:
        return self.output_dir / "reports"

    @property
    def logits_dir(self) -> Path:
        return self.output_dir / "logits"

    @property
    def official_split_file(self) -> Path:
        """
        The ICBHI challenge ships an official patient-wise train/test list
        named ICBHI_challenge_train_test.txt. If it sits inside DATASET_DIR we
        pick it up automatically, which makes results comparable to published
        work.

        If yours lives somewhere else, replace the line below with the full
        path, for example:
            return Path(r"E:\\downloads\\ICBHI_challenge_train_test.txt")
        """
        return self.dataset_dir / "ICBHI_challenge_train_test.txt"


# ----------------------------------------------------------------------------
# Audio / feature extraction
# ----------------------------------------------------------------------------
@dataclass
class AudioConfig:
    target_sr: int = 16000

    # Fixed clip length in seconds.
    #
    # Typical ICBHI breathing cycles are ~2-3s. Padding to 8s (as the original
    # pipeline did) meant 60-70% of a typical sample was zero padding that the
    # LSTM still had to process. We use a shorter window AND emit a validity
    # mask so models can ignore padded frames entirely.
    duration: float = 4.0

    # Bandpass: crackles and wheezes live roughly in this band. Outside it is
    # mostly heartbeat rumble, microphone handling noise, and room tone.
    lowcut: float = 100.0
    highcut: float = 2000.0
    filter_order: int = 4

    n_fft: int = 1024
    hop_length: int = 256

    # Mel/MFCC filterbanks restricted to the SAME band we filtered to.
    # Without this, librosa spreads filters over 0-8000 Hz, so most of them
    # measure the silence the bandpass just created.
    fmin: float = 100.0
    fmax: float = 2000.0

    n_mels: int = 128      # image branch
    n_mfcc: int = 40       # signal branch (before deltas)

    @property
    def max_samples(self) -> int:
        return int(self.target_sr * self.duration)

    @property
    def n_frames(self) -> int:
        return self.max_samples // self.hop_length + 1

    @property
    def signal_channels(self) -> int:
        return self.n_mfcc * 3          # MFCC + delta + delta-delta


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------
@dataclass
class TrainConfig:
    seed: int = 42
    num_classes: int = 4
    class_names: List[str] = field(
        default_factory=lambda: ["Normal", "Crackle", "Wheeze", "Both"]
    )

    batch_size: int = 64
    epochs: int = 60
    warmup_epochs: int = 5
    lr: float = 3e-4
    min_lr_factor: float = 0.02
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    label_smoothing: float = 0.05

    # Imbalance handling.
    #
    # The correction goes in the SAMPLER (so every batch has a stable class
    # mix) rather than stacking heavy loss weights on top of unbalanced
    # batches. Stacking both caused the gradient oscillation in earlier runs.
    #
    # sampler_power: 0.0 = natural distribution, 1.0 = fully balanced.
    # 0.5 keeps some of the real prior while lifting the rare classes.
    sampler_power: float = 0.5
    focal_gamma: float = 1.0

    # Augmentation
    mixup_alpha: float = 0.4
    noise_prob: float = 0.5
    noise_std: float = 0.05        # applied to NORMALISED features
    time_mask_prob: float = 0.5
    time_mask_frac: float = 0.10
    freq_mask_prob: float = 0.5
    freq_mask_frac: float = 0.10

    # Checkpoint selection metric: "icbhi_score" | "macro_f1" | "accuracy"
    monitor: str = "icbhi_score"
    patience: int = 15

    # Used only when the official test list is unavailable.
    # The split is always PATIENT-WISE, never cycle-wise.
    val_fraction: float = 0.2

    # num_workers: keep 0 on Windows. Values > 0 require the entry point to be
    # guarded by `if __name__ == "__main__":` (ours is) but can still be flaky.
    # On Linux, 4 is usually a solid speedup.
    num_workers: int = 0
    amp: bool = True               # mixed precision; automatically off on CPU


@dataclass
class Config:
    paths: Paths = field(default_factory=Paths)
    audio: AudioConfig = field(default_factory=AudioConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        def _enc(o):
            if isinstance(o, Path):
                return str(o)
            raise TypeError(type(o))

        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2, default=_enc)


def get_config() -> Config:
    return Config()
