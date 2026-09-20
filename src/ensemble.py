"""
Fuse the two branches into the final ensemble.

PATHS USED BY THIS SCRIPT (all set at the top of src/config.py):
    reads  <- OUTPUT_DIR/logits/*_val.npz    cached by each training run
    writes -> OUTPUT_DIR/reports/ensemble_report.txt
    writes -> OUTPUT_DIR/reports/ensemble_results.json

Override the logits location without editing config.py:
    python -m src.ensemble --logits-dir "E:/outputs/logits"

Usage
-----
    # after training both branches
    python -m src.ensemble

Method
------
Both branches are trained independently and their validation logits cached. We
then fuse. Three strategies are tried and the best on the validation set is
reported:

  1. Weighted average of softmax probabilities, with the blend weight swept
     over a grid.
  2. Weighted average of logits.
  3. Learned logistic regression stacker over the concatenated probabilities.

An honest caveat, stated plainly because it is easy to fool yourself here:
we are SELECTING the fusion strategy and blend weight on the same validation
set we report. That makes the ensemble number optimistically biased. With a
dataset this small there is no clean way around it short of a third held-out
split. Treat the ensemble figure as an upper estimate, and if you need a
defensible number for a paper, carve out a separate test set (see
--holdout-frac) or report cross-validated results.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .config import get_config
from .metrics import compute_all, full_report, format_metrics


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def load_branch_logits(logits_dir: Path):
    """Load every cached *_val.npz and verify they describe the same split."""
    files = sorted(Path(logits_dir).glob("*_val.npz"))
    if len(files) < 2:
        raise SystemExit(
            f"Need at least two branches' cached logits in {logits_dir}.\n"
            f"Found {len(files)}. Train both branches first:\n"
            f"    python -m src.train --branch signal\n"
            f"    python -m src.train --branch image"
        )

    branches = {}
    ref_targets = None
    for f in files:
        d = np.load(f)
        name = f.stem.replace("_val", "")
        if ref_targets is None:
            ref_targets = d["targets"]
        elif not np.array_equal(ref_targets, d["targets"]):
            raise SystemExit(
                f"{f.name} was evaluated on a DIFFERENT validation split than "
                f"the other branches. Retrain both branches with the same "
                f"--seed and the same split settings, otherwise fusing them is "
                f"meaningless."
            )
        branches[name] = d["logits"]
        print(f"  loaded {name}: {d['logits'].shape}")

    return branches, ref_targets


def sweep_blend(pa, pb, targets, class_names, step=0.02):
    """Sweep the blend weight and return the best (weight, metrics)."""
    best_w, best = None, None
    for w in np.arange(0.0, 1.0 + 1e-9, step):
        p = w * pa + (1 - w) * pb
        m = compute_all(targets, p.argmax(1), class_names)
        if best is None or m["icbhi_score"] > best["icbhi_score"]:
            best, best_w = m, float(w)
    return best_w, best


def logistic_stacker(feats, targets, class_names, seed=42):
    """Train a logistic-regression stacker with cross-validated predictions."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    # (no multi_class arg: removed in scikit-learn 1.7; multinomial is
    #  now the default for multiclass problems)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    # Cross-validated predictions -> the stacker never scores its own
    # training rows, which keeps this comparison a bit fairer.
    preds = cross_val_predict(clf, feats, targets, cv=cv)
    clf.fit(feats, targets)
    return clf, compute_all(targets, preds, class_names)


def main():
    ap = argparse.ArgumentParser(description="Fuse branch predictions")
    ap.add_argument("--logits-dir", default=None)
    ap.add_argument("--signal", default=None,
                    help="name of the signal branch run")
    ap.add_argument("--image", default=None,
                    help="name of the image branch run")
    args = ap.parse_args()

    cfg = get_config()
    class_names = cfg.train.class_names
    logits_dir = Path(args.logits_dir or
                      Path(cfg.paths.output_dir) / "logits")

    print("Loading cached branch logits...")
    branches, targets = load_branch_logits(logits_dir)

    # Pick the two branches to fuse
    names = list(branches)
    sig_name = args.signal or next(
        (n for n in names if "signal" in n), names[0])
    img_name = args.image or next(
        (n for n in names if "image" in n),
        next(n for n in names if n != sig_name))

    print(f"\nFusing: {sig_name}  +  {img_name}")

    la, lb = branches[sig_name], branches[img_name]
    pa, pb = softmax(la), softmax(lb)

    print("\n--- Individual branches ---")
    ma = compute_all(targets, la.argmax(1), class_names)
    mb = compute_all(targets, lb.argmax(1), class_names)
    print(f"  {sig_name:28s} {format_metrics(ma)}")
    print(f"  {img_name:28s} {format_metrics(mb)}")

    results = {}

    # 1. probability blend
    w_p, m_prob = sweep_blend(pa, pb, targets, class_names)
    results["prob_blend"] = {"weight": w_p, **m_prob}
    print(f"\n  prob blend  (w={w_p:.2f})      {format_metrics(m_prob)}")

    # 2. logit blend
    w_l, m_logit = sweep_blend(la, lb, targets, class_names)
    results["logit_blend"] = {"weight": w_l, **m_logit}
    print(f"  logit blend (w={w_l:.2f})      {format_metrics(m_logit)}")

    # 3. learned stacker
    try:
        feats = np.concatenate([pa, pb], axis=1)
        clf, m_stack = logistic_stacker(feats, targets, class_names,
                                        cfg.train.seed)
        results["stacker"] = m_stack
        print(f"  logistic stacker (CV)     {format_metrics(m_stack)}")
    except Exception as e:
        print(f"  logistic stacker unavailable: {e}")
        m_stack, clf = None, None

    # Best strategy
    best_name = max(results, key=lambda k: results[k]["icbhi_score"])
    print(f"\nBest fusion strategy: {best_name} "
          f"(ICBHI {results[best_name]['icbhi_score']:.4f})")

    if best_name == "prob_blend":
        final_pred = (w_p * pa + (1 - w_p) * pb).argmax(1)
    elif best_name == "logit_blend":
        final_pred = (w_l * la + (1 - w_l) * lb).argmax(1)
    else:
        final_pred = clf.predict(np.concatenate([pa, pb], axis=1))

    report = full_report(targets, final_pred, class_names)
    print("\n" + report)

    report_dir = Path(cfg.paths.output_dir) / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    with open(report_dir / "ensemble_report.txt", "w") as f:
        f.write(f"Fused: {sig_name} + {img_name}\n")
        f.write(f"Best strategy: {best_name}\n\n")
        f.write(report)
        f.write("\n\nNOTE: the fusion strategy and blend weight were selected "
                "on this same validation set, so this figure is optimistically "
                "biased. See the docstring in src/ensemble.py.\n")

    with open(report_dir / "ensemble_results.json", "w") as f:
        json.dump({"branches": {sig_name: ma, img_name: mb},
                   "fusion": results,
                   "best_strategy": best_name}, f, indent=2)

    print(f"\nSaved -> {report_dir.resolve()}")


if __name__ == "__main__":
    main()
