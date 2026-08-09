from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_run_benchmark_module():
    path = Path(__file__).with_name("run_benchmark.py")
    spec = importlib.util.spec_from_file_location("run_benchmark_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--train-frac", type=float, default=0.45)
    parser.add_argument("--cal-frac", type=float, default=0.25)
    custom_args, rest = parser.parse_known_args()

    if custom_args.train_frac <= 0 or custom_args.cal_frac <= 0:
        raise ValueError("train-frac and cal-frac must be positive")
    if custom_args.train_frac + custom_args.cal_frac >= 1.0:
        raise ValueError("train-frac + cal-frac must be < 1.0 to leave a test split")

    run_benchmark = load_run_benchmark_module()
    split_fn = run_benchmark.train_test_split

    def split_with_fallback(*args, stratify, stage: str, seed: int, **kwargs):
        try:
            return split_fn(*args, stratify=stratify, **kwargs)
        except ValueError as exc:
            print(
                "[custom_split] fallback_to_unstratified "
                f"stage={stage} seed={seed} train_frac={custom_args.train_frac} "
                f"cal_frac={custom_args.cal_frac} reason={exc!r}",
                file=sys.stderr,
                flush=True,
            )
            return split_fn(*args, stratify=None, **kwargs)

    def robust_train_test_split(*args, stratify=None, random_state=None, **kwargs):
        try:
            return split_fn(*args, stratify=stratify, random_state=random_state, **kwargs)
        except ValueError as exc:
            print(
                "[custom_split] fallback_to_unstratified "
                f"stage=internal seed={random_state} train_frac={custom_args.train_frac} "
                f"cal_frac={custom_args.cal_frac} reason={exc!r}",
                file=sys.stderr,
                flush=True,
            )
            return split_fn(*args, stratify=None, random_state=random_state, **kwargs)

    def custom_split(bundle, seed, train_frac=0.45, cal_frac=0.25):
        test_frac = 1.0 - custom_args.train_frac - custom_args.cal_frac
        X_tmp, X_test, y_tmp, y_test = split_with_fallback(
            bundle.X,
            bundle.y,
            test_size=test_frac,
            random_state=seed,
            stratify=bundle.y,
            stage="test",
            seed=seed,
        )
        rel_cal = custom_args.cal_frac / (custom_args.train_frac + custom_args.cal_frac)
        X_train, X_cal, y_train, y_cal = split_with_fallback(
            X_tmp,
            y_tmp,
            test_size=rel_cal,
            random_state=seed + 1009,
            stratify=y_tmp,
            stage="calibration",
            seed=seed,
        )
        return X_train, X_cal, X_test, y_train, y_cal, y_test

    # Patch all train_test_split call sites used by run_benchmark and its imported
    # data loader. This keeps the normal stratified protocol whenever feasible,
    # but allows deliberately tiny calibration-set stress tests to complete.
    run_benchmark.train_test_split = robust_train_test_split
    try:
        import rankcover.data as rankcover_data

        rankcover_data.train_test_split = robust_train_test_split
    except Exception as exc:
        print(f"[custom_split] data_loader_patch_warning reason={exc!r}", file=sys.stderr, flush=True)

    run_benchmark._split = custom_split
    sys.argv = [sys.argv[0], *rest]
    run_benchmark.main()


if __name__ == "__main__":
    main()
