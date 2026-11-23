"""Train Naive Bayes, Decision Tree and Random Forest on TF-IDF features.

Run from the project root:
    python -m src.train_model --mode five
"""
import argparse
import json
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import (StratifiedGroupKFold, StratifiedKFold,
                                     cross_validate, train_test_split)
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src import config
from src.data_preprocessing import clean_dataset, load_raw, normalize_description
from src.evaluate_model import evaluate, plot_confusion_matrix, plot_model_comparison


def make_vectorizer():
    # The SAME normalisation function is used at training and prediction time because it
    # lives inside the pipeline. No English stop-word list: merchant names such as
    # "The Body Shop" would lose words, and descriptions are only 1-4 tokens long.
    return TfidfVectorizer(
        preprocessor=normalize_description,
        ngram_range=(1, 2),
        min_df=1,
        max_features=5000,
        sublinear_tf=True,
    )


TREE_DEPTH_GRID = [10, 25, 50, None]


def make_models(tree_depth=None):
    rs = config.RANDOM_STATE
    return {
        "Naive Bayes": MultinomialNB(alpha=0.5),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=tree_depth, min_samples_leaf=2, class_weight="balanced",
            random_state=rs),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, min_samples_leaf=1, class_weight="balanced",
            n_jobs=-1, random_state=rs),
    }


def build_pipeline(model):
    # TF-IDF is a pipeline step, so it is fitted ONLY on whatever data .fit() receives
    # (the training split, or the training folds inside cross-validation) -> no leakage.
    return Pipeline([("tfidf", make_vectorizer()), ("clf", model)])


def tune_tree_depth(X_tr, y_tr, cv):
    """Pick max_depth by CV on the TRAINING split only. Ties -> shallower tree.

    A tree on sparse TF-IDF words tends to peel off roughly one merchant per split, so a
    depth cap that is too small truncates the tree and loses whole merchants.
    """
    scores = {}
    for depth in TREE_DEPTH_GRID:
        pipe = build_pipeline(make_models(depth)["Decision Tree"])
        cv_res = cross_validate(pipe, X_tr, y_tr, cv=cv, scoring="f1_macro")
        scores[depth] = float(np.mean(cv_res["test_score"]))
    best = max(TREE_DEPTH_GRID,
               key=lambda d: (round(scores[d], 4), -(d if d is not None else 10**9)))
    return best, scores


def train(mode="five", data_path=config.RAW_DATA_PATH):
    raw = load_raw(data_path)
    df, prep_report = clean_dataset(raw, mode=mode)
    labels = sorted(df["category"].unique())
    X, y, groups = df["description_raw"].astype(str), df["category"], df["merchant_key"]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.RANDOM_STATE)
    print(f"Mode={mode} | classes={labels}\nTrain={len(X_tr)} Test={len(X_te)}")

    results, fitted = {}, {}
    scoring = {"f1_macro": "f1_macro", "accuracy": "accuracy"}
    random_cv = StratifiedKFold(5, shuffle=True, random_state=config.RANDOM_STATE)
    group_cv = StratifiedGroupKFold(5, shuffle=True, random_state=config.RANDOM_STATE)

    tree_depth, depth_scores = tune_tree_depth(X_tr, y_tr, random_cv)
    print(f"Decision Tree max_depth chosen by train-CV: {tree_depth} | scores: {depth_scores}")

    for name, model in make_models(tree_depth).items():
        print(f"\n=== {name} ===")
        pipe = build_pipeline(model)

        # (1) 5-fold CV on the TRAINING split only: used to choose the model.
        cv = cross_validate(pipe, X_tr, y_tr, cv=random_cv, scoring=scoring)

        # (2) Fit on the training split, score once on the held-out test split.
        t0 = time.perf_counter()
        pipe.fit(X_tr, y_tr)
        train_seconds = time.perf_counter() - t0
        test = evaluate(pipe, X_te, y_te, labels)

        # (3) Harder check: every test fold contains merchants never seen in training.
        gcv = cross_validate(build_pipeline(make_models(tree_depth)[name]), X, y, groups=groups,
                             cv=group_cv, scoring=scoring)

        results[name] = {
            "train_seconds": train_seconds,
            "train_cv": {"f1_macro_mean": float(np.mean(cv["test_f1_macro"])),
                         "f1_macro_std": float(np.std(cv["test_f1_macro"]))},
            "test": test,
            "unseen_merchant_cv": {
                "f1_macro_mean": float(np.mean(gcv["test_f1_macro"])),
                "f1_macro_std": float(np.std(gcv["test_f1_macro"])),
                "accuracy_mean": float(np.mean(gcv["test_accuracy"])),
            },
            "n_features": int(len(pipe.named_steps["tfidf"].vocabulary_)),
        }
        fitted[name] = pipe
        plot_confusion_matrix(test["confusion_matrix"], labels,
                              f"{name}: confusion matrix (test)",
                              f"confusion_{name.lower().replace(' ', '_')}.png")
        print(f"train-CV macro-F1 {results[name]['train_cv']['f1_macro_mean']:.4f} | "
              f"test acc {test['accuracy']:.4f} macro-F1 {test['f1_macro']:.4f} | "
              f"unseen-merchant macro-F1 {results[name]['unseen_merchant_cv']['f1_macro_mean']:.4f} | "
              f"train {train_seconds:.2f}s")

    # Model selection uses TRAINING-split CV only (the test split is not used to choose).
    # Scores are rounded to 4 decimals; ties go to the simpler model (fixed order below),
    # not to whichever happened to train a few milliseconds faster.
    simplicity_order = ["Naive Bayes", "Decision Tree", "Random Forest"]
    best = max(results, key=lambda n: (round(results[n]["train_cv"]["f1_macro_mean"], 4),
                                       -simplicity_order.index(n)))
    plot_model_comparison(results)

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted[best], config.MODEL_PATH)

    metadata = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "dataset": config.DATASET_NAME,
        "dataset_note": config.DATASET_NOTE,
        "label_mode": mode,
        "classes": labels,
        "rows_after_cleaning": int(len(df)),
        "n_train": int(len(X_tr)),
        "n_test": int(len(X_te)),
        "preprocessing_report": prep_report,
        "decision_tree_depth_search": {str(k): v for k, v in depth_scores.items()},
        "decision_tree_max_depth": tree_depth,
        "selected_model": best,
        "selection_rule": ("highest 5-fold CV macro-F1 on the training split; "
                           "ties -> simpler model (Naive Bayes > Decision Tree > Random Forest)"),
        "results": results,
        "limitations": [
            "Dataset appears synthetic/templated; scores do not predict real-world accuracy.",
            "A random split puts the same merchants in train and test, which inflates scores.",
            "Unseen-merchant CV shows how the word-based model copes with new merchant names.",
            "Model confidence is not a calibrated probability.",
        ],
    }
    with open(config.METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSelected model: {best}\nSaved -> {config.MODEL_PATH}\nSaved -> {config.METADATA_PATH}")
    return metadata


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=list(config.CATEGORY_MAPS), default="five")
    ap.add_argument("--data", default=str(config.RAW_DATA_PATH))
    args = ap.parse_args()
    train(args.mode, args.data)
