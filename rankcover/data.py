from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_digits, load_iris, load_wine
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder


@dataclass
class DatasetBundle:
    name: str
    X: np.ndarray
    y: np.ndarray


def _encode_frame(X) -> np.ndarray:
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    X = X.copy()
    num_cols = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    parts = []
    if num_cols:
        num = X[num_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        num = SimpleImputer(strategy="median").fit_transform(num)
        parts.append(num)
    if cat_cols:
        cat = X[cat_cols].astype("string").fillna("__missing__")
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-1)
        parts.append(enc.fit_transform(cat))
    if not parts:
        raise ValueError("dataset has no usable features")
    X_out = np.concatenate(parts, axis=1).astype(np.float32)
    keep = np.isfinite(X_out).all(axis=1)
    if not keep.all():
        X_out = X_out[keep]
    return X_out, keep


def _encode_y(y, keep: np.ndarray | None = None) -> np.ndarray:
    arr = np.asarray(y)
    if keep is not None and keep.shape[0] == arr.shape[0]:
        arr = arr[keep]
    mask = pd.notnull(arr)
    arr = arr[mask]
    return LabelEncoder().fit_transform(arr)


def _finalize(name: str, X, y, max_classes: int = 10) -> DatasetBundle:
    X_np, keep = _encode_frame(X)
    y_np = _encode_y(y, keep)
    if X_np.shape[0] != y_np.shape[0]:
        n = min(X_np.shape[0], y_np.shape[0])
        X_np, y_np = X_np[:n], y_np[:n]
    labels, counts = np.unique(y_np, return_counts=True)
    ok = counts >= 8
    good_labels = set(labels[ok].tolist())
    mask = np.array([v in good_labels for v in y_np])
    X_np, y_np = X_np[mask], LabelEncoder().fit_transform(y_np[mask])
    n_classes = np.unique(y_np).size
    if n_classes < 2:
        raise ValueError(f"{name}: fewer than two classes after filtering")
    if n_classes > max_classes:
        raise ValueError(f"{name}: too many classes ({n_classes})")
    return DatasetBundle(name=name, X=X_np, y=y_np.astype(int))


def load_builtin(name: str) -> DatasetBundle:
    loaders = {
        "breast_cancer": load_breast_cancer,
        "wine": load_wine,
        "digits": load_digits,
        "iris": load_iris,
    }
    if name not in loaders:
        raise KeyError(f"unknown builtin dataset: {name}")
    data = loaders[name](as_frame=True)
    return _finalize(name, data.data, data.target)


def load_openml_dataset(dataset_id: int, cache_dir: str | None = None) -> DatasetBundle:
    import openml

    if cache_dir:
        openml.config.set_root_cache_directory(cache_dir)
    ds = openml.datasets.get_dataset(dataset_id, download_data=True, download_qualities=False, download_features_meta_data=False)
    X, y, _, _ = ds.get_data(target=ds.default_target_attribute, dataset_format="dataframe")
    return _finalize(f"openml_{dataset_id}_{ds.name}", X, y)


def load_openml_task(task_id: int, cache_dir: str | None = None) -> DatasetBundle:
    import openml

    if cache_dir:
        openml.config.set_root_cache_directory(cache_dir)
    task = openml.tasks.get_task(task_id, download_data=True)
    X, y = task.get_X_and_y(dataset_format="dataframe")
    did = getattr(task, "dataset_id", "unknown")
    return _finalize(f"task_{task_id}_dataset_{did}", X, y)


def stratified_subsample(X: np.ndarray, y: np.ndarray, max_rows: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    if max_rows <= 0 or X.shape[0] <= max_rows:
        return X, y
    _, idx = train_test_split(
        np.arange(X.shape[0]),
        train_size=max_rows,
        random_state=seed,
        stratify=y,
    )
    # The split above returns the held-out indices in idx when train_size is set.
    # Use an explicit choice instead to avoid relying on sklearn's return order.
    rng = np.random.default_rng(seed)
    selected = []
    labels, counts = np.unique(y, return_counts=True)
    raw = max_rows * counts / counts.sum()
    per_class = np.maximum(2, np.floor(raw).astype(int))
    while per_class.sum() > max_rows:
        j = int(np.argmax(per_class))
        if per_class[j] > 2:
            per_class[j] -= 1
        else:
            break
    for label, n_take in zip(labels, per_class):
        pool = np.flatnonzero(y == label)
        selected.extend(rng.choice(pool, size=min(int(n_take), pool.size), replace=False).tolist())
    selected = np.array(selected, dtype=int)
    rng.shuffle(selected)
    return X[selected], y[selected]


def load_datasets(spec: str, openml_cache: str | None, max_rows: int, seed: int) -> Iterable[DatasetBundle]:
    names = []
    for token in spec.split(","):
        token = token.strip()
        if token:
            names.append(token)
    for token in names:
        if token.startswith("task:"):
            bundle = load_openml_task(int(token.split(":", 1)[1]), openml_cache)
        elif token.startswith("openml:"):
            bundle = load_openml_dataset(int(token.split(":", 1)[1]), openml_cache)
        else:
            bundle = load_builtin(token)
        X, y = stratified_subsample(bundle.X, bundle.y, max_rows=max_rows, seed=seed)
        yield DatasetBundle(bundle.name, X, y)
