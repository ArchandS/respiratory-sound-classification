# Dual-Branch Ensemble Networks for Respiratory Sound Analysis

Four-class classification of respiratory sounds (Normal / Crackle / Wheeze / Both)
on the **ICBHI 2017 Respiratory Sound Database**, using two complementary deep
learning branches whose predictions are fused into an ensemble.

| Branch | Input representation | Architecture |
|---|---|---|
| **Signal** | MFCC + Δ + ΔΔ sequence `(120, T)` | Multi-scale 1D CNN → BiLSTM → residual self-attention → masked pooling |
| **Image** | Log-Mel spectrogram `(1, 128, T)` | 2D residual CNN with squeeze-excitation (or ImageNet-pretrained ResNet) |
| **Ensemble** | Both branches' probabilities | Blend-weight sweep + logistic stacker |

---

## Read this before you interpret any result

**Expect an ICBHI Score in the 0.50–0.65 range.** That is what four-class ICBHI
classification on the official patient-wise split actually looks like in the
published literature. This task is genuinely hard, and it is hard for reasons
that no amount of model tuning removes:

- Only ~126 patients, so the effective sample size is far smaller than the 6,898
  cycle count suggests.
- Severe class imbalance — 52.8% Normal, 7.3% Both.
- Substantial label noise. Human annotators disagree about whether a given cycle
  contains a crackle more often than you would hope.

**If you see 90%+ accuracy, you have a bug, not a breakthrough.** The usual
culprit is cycle-level rather than patient-level splitting — see
[Patient-wise splitting](#patient-wise-splitting-the-most-important-detail) below.

Also note: **raw accuracy is a misleading metric here.** Predicting "Normal" for
every sample scores **52.8% accuracy** while detecting zero disease. Any accuracy
figure below that is worse than a constant predictor. This repo therefore
reports and selects on the **ICBHI Score**, the official challenge metric.

---

## Setting your paths

**There is exactly one path you must set.** Open `src/config.py` — all three
paths are at the top of the file in a clearly marked block.

| # | Constant | Must edit? | What it is |
|---|---|---|---|
| 1 | `DATASET_DIR` | **YES** | Folder holding your ICBHI `.wav` + `.txt` files |
| 2 | `PROCESSED_FILE` | No | Where extracted features are written (~1–3 GB) |
| 3 | `OUTPUT_DIR` | No | Checkpoints, reports, cached logits |

```python
# src/config.py  — Windows (note the r"" prefix, required for backslashes)
DATASET_DIR = r"E:\ICBHI_final_database"

# Linux / macOS
DATASET_DIR = "/home/user/data/ICBHI_final_database"
```

> **Most common mistake:** pointing at a parent folder. If your audio lives in
> `ICBHI_final_database/audio_and_txt_files/`, point at **that subfolder**.
> `verify_setup` detects this and prints the exact path to use.

You can also override paths per-run without editing anything:

```bash
python -m src.train --branch signal --data "E:/processed_icbhi.npz"
python -m src.ensemble --logits-dir "E:/outputs/logits"
```

---

## Quick start

```bash
git clone <your-repo-url>
cd respiratory-sound-classification
pip install -r requirements.txt

# 1. Set DATASET_DIR at the top of src/config.py

# 2. Check everything before committing to a long run
python -m src.verify_setup

# 3. Preprocess once (~10-20 min)
python -m src.preprocess

# 4. Train both branches
python -m src.train --branch signal
python -m src.train --branch image

# 5. Fuse
python -m src.ensemble
```

Or run everything: `bash scripts/run_all.sh` (Linux/macOS) · `scripts\run_all.bat` (Windows)

`verify_setup` checks installed packages, CUDA, that your dataset folder holds
matched `.wav`/`.txt` pairs, that annotations parse into cycles, whether the
official split list is present, and that you have writable paths and enough
disk space. Run it first — it turns a failure 20 minutes into preprocessing
into a five-second message.

### Dataset

Download the ICBHI 2017 database separately (it is not redistributable here).
If `ICBHI_challenge_train_test.txt` sits inside `DATASET_DIR`, the official
patient-wise split is used automatically.

---

## Repository layout

```
├── src/
│   ├── config.py          All paths + hyperparameters  ← EDIT PATHS HERE
│   ├── verify_setup.py    Pre-flight check — run this first
│   ├── preprocess.py      Audio → features (.npz), run once
│   ├── splits.py          Patient-wise splitting  ← read this one
│   ├── dataset.py         Normalisation, augmentation, balanced sampling
│   ├── losses.py          Focal loss, mixup, class weights
│   ├── metrics.py         ICBHI Score, macro-F1, reports
│   ├── engine.py          Shared training loop (EMA, warmup, clipping)
│   ├── train.py           Entry point for either branch
│   ├── ensemble.py        Fusion of branch predictions
│   └── models/
│       ├── signal_branch.py
│       └── image_branch.py
├── scripts/run_all.sh|.bat
└── requirements.txt
```

---

## Design decisions

This section documents what changed from a naive implementation and **why**,
since most of these were diagnosed from actual failures.

### Patient-wise splitting (the most important detail)

Each recording contributes many breathing cycles. Split at the cycle level and
cycles from the same patient — same microphone, same room tone, same chest —
land in both train and validation. The model then scores well by recognising
*the recording*, not the pathology.

This repo splits by **patient**, and prefers the official ICBHI train/test list
when available. `describe_split()` prints a patient-overlap check on every run
and warns loudly if leakage reappears.

> Switching from cycle-level to patient-level splitting will make your numbers
> **drop**. That drop is leakage being removed. The lower number is the real one.

### Feature normalisation

MFCC coefficient 0 encodes overall loudness and has values in the hundreds;
the remaining coefficients sit in the tens. Fed in raw, one channel is orders of
magnitude larger than the rest, dominates the gradients, and the network spends
its capacity fighting the scale mismatch instead of learning acoustics.

Measured on real MFCCs extracted by this pipeline, channel 0 came out **865×
the magnitude** of channel 5.

Every channel is standardised to zero mean / unit variance using statistics
computed from the **training split only** (computing them over the full dataset
leaks validation information).

### Band-limited filterbanks

The audio is bandpassed to 100–2000 Hz, where crackles and wheezes live. The
mel and MFCC filterbanks are given matching `fmin`/`fmax`. Without this, librosa
spreads filters across 0–8000 Hz and most of them measure the silence the
bandpass just created.

### Padding masks

Typical cycles run 2–3 s. Padding to a fixed window means many frames are zeros
— measured on real extracted features, a 1.3 s cycle yields only 82 real frames
out of 251, so **67% of the tensor is padding**.
Rather than letting the LSTM and pooling treat padding as signal, `preprocess.py`
records how many frames are real, and the signal branch masks padded frames out
of attention and of both mean and max pooling.

### Imbalance handling — applied once, not twice

The correction lives in the **sampler** (`WeightedRandomSampler`, `sampler_power=0.5`),
so every batch has a stable class mix. Loss-side class weights are deliberately
mild.

Stacking heavy loss weights on top of *unbalanced* batches was the cause of a
real failure in this project: with `Both` at ~7% of data, batches contained
wildly varying numbers of minority samples, so a heavily-weighted loss produced
gradients that swung direction batch to batch. Validation accuracy oscillated
40% → 15% → 22% → 42% across consecutive epochs. Balancing the batches fixes the
cause; reweighting the loss only rescales the symptom.

### Training stability

- **GroupNorm over BatchNorm** — BatchNorm statistics shift with batch class
  composition, adding variance to an already imbalanced problem.
- **Pre-norm residual attention** — `x = x + Attn(LN(x))`. Without a residual
  path or normalisation, the attention block's output scale drifts with its
  input and feeds straight into pooling.
- **LR warmup → cosine decay** — the attention/LSTM stack is fragile while Q/K/V
  projections are still random.
- **Gradient clipping** at 1.0.
- **Weight EMA** — evaluating a moving average of recent weights instead of the
  exact current point noticeably damps epoch-to-epoch metric noise.
- **Mixup + SpecAugment-style masking** — on a small, noisily-labelled dataset
  these discourage overconfidence about patient-specific artefacts.

### Model selection

Checkpoints are selected on **ICBHI Score**, never raw accuracy. Selecting on
accuracy actively rewards collapsing toward the majority class, which fights the
entire point of the imbalance handling.

---

## Metrics

**ICBHI Score** = (Sensitivity + Specificity) / 2

- *Specificity* — Normal cycles correctly identified as Normal
- *Sensitivity* — abnormal cycles assigned their correct abnormal class

Reference points: a majority-class predictor scores **0.50**; a perfect one
scores **1.00**. Macro-F1, balanced accuracy, raw accuracy, per-class
precision/recall, and the confusion matrix are all reported alongside.

---

## Configuration

Everything tunable is in `src/config.py`. Common overrides:

```bash
python -m src.train --branch signal --epochs 80 --lr 2e-4 --batch-size 32
python -m src.train --branch image --backbone resnet34   # pretrained, needs torchvision
python -m src.train --branch signal --force-patient-split  # ignore the official list
```

| Knob | Default | Effect |
|---|---|---|
| `sampler_power` | `0.5` | 0 = natural distribution, 1 = fully balanced |
| `focal_gamma` | `1.0` | Higher focuses harder on difficult examples |
| `mixup_alpha` | `0.4` | 0 disables mixup |
| `duration` | `4.0` s | Clip window length |
| `monitor` | `icbhi_score` | Checkpoint selection metric |

---

## Suggested tuning order

If you want to push the score up, these are roughly ordered by expected payoff:

1. `--backbone resnet34` on the image branch. Pretrained features usually help
   most on a dataset this small.
2. Sweep `sampler_power` over {0.3, 0.5, 0.7} — this trades sensitivity against
   specificity and directly moves the ICBHI Score.
3. Sweep `mixup_alpha` over {0.2, 0.4, 0.8}.
4. Average several seeds per branch before fusing. Seed variance on this dataset
   is large enough that a single run's number is not very meaningful.
5. Try `duration` of 3 s vs 4 s vs 6 s.

**Report the mean and standard deviation across at least 3 seeds.** Single-run
numbers on ICBHI are noisy enough to be actively misleading.

---

## Known limitations

- The ensemble's blend weight and fusion strategy are selected on the same
  validation set that gets reported, so the ensemble figure is optimistically
  biased. For a defensible number, hold out a third split or cross-validate.
- The image branch does not mask padded frames (the signal branch does). A 2D
  CNN with global pooling is fairly tolerant of this, but it is not free.
- No test-time augmentation.
- Cycle-level classification only; no recording-level or patient-level
  aggregation.

---

## Citation

```bibtex
@article{rocha2019icbhi,
  title   = {An open access database for the evaluation of respiratory sound
             classification algorithms},
  author  = {Rocha, B M and Filos, D and Mendes, L and others},
  journal = {Physiological Measurement},
  volume  = {40}, number = {3}, pages = {035001}, year = {2019}
}
```

## License

MIT (see `LICENSE`). The ICBHI database itself is distributed under its own terms
by its authors and is not covered by this license.
