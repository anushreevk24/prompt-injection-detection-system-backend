from pathlib import Path
from typing import Dict

import joblib
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from retraining.dataset_manager import ATHSDatasetManager


# ============================================================
# ATHS MODEL EVALUATOR CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

CURRENT_MODEL_PATH = (
    BASE_DIR
    / "models"
    / "random_forest.joblib"
)

CANDIDATE_MODEL_PATH = (
    BASE_DIR
    / "models"
    / "random_forest_candidate.joblib"
)


# ============================================================
# PROMOTION GATE
# ============================================================

MIN_F1_CHANGE = 0.0
MIN_RECALL_CHANGE = 0.0

MAX_FPR_INCREASE = 0.02

MAX_ROC_AUC_DECREASE = 0.01


# ============================================================
# ATHS MODEL EVALUATOR
# ============================================================

class ATHSModelEvaluator:

    def __init__(self) -> None:

        print("Initializing ATHS Model Evaluator...")

        self.dataset_manager = (
            ATHSDatasetManager()
        )

        print("ATHS Model Evaluator initialized.")

    # --------------------------------------------------------
    # LOAD MODELS
    # --------------------------------------------------------

    def load_models(self):

        if not CURRENT_MODEL_PATH.exists():

            raise FileNotFoundError(
                "Current model not found: "
                f"{CURRENT_MODEL_PATH}"
            )

        if not CANDIDATE_MODEL_PATH.exists():

            raise FileNotFoundError(
                "Candidate model not found: "
                f"{CANDIDATE_MODEL_PATH}"
            )

        current_model = joblib.load(
            CURRENT_MODEL_PATH
        )

        candidate_model = joblib.load(
            CANDIDATE_MODEL_PATH
        )

        print()
        print("MODELS LOADED")
        print("-" * 50)

        print(
            f"Current model   : "
            f"{CURRENT_MODEL_PATH}"
        )

        print(
            f"Candidate model : "
            f"{CANDIDATE_MODEL_PATH}"
        )

        return (
            current_model,
            candidate_model,
        )

    # --------------------------------------------------------
    # EVALUATE MODEL
    # --------------------------------------------------------

    def evaluate_model(
        self,
        model,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict:

        predictions = model.predict(
            X_test
        )

        probabilities = (
            model.predict_proba(X_test)[:, 1]
        )

        accuracy = accuracy_score(
            y_test,
            predictions,
        )

        precision = precision_score(
            y_test,
            predictions,
            zero_division=0,
        )

        recall = recall_score(
            y_test,
            predictions,
            zero_division=0,
        )

        f1 = f1_score(
            y_test,
            predictions,
            zero_division=0,
        )

        roc_auc = roc_auc_score(
            y_test,
            probabilities,
        )

        matrix = confusion_matrix(
            y_test,
            predictions,
        )

        tn, fp, fn, tp = matrix.ravel()

        fpr = (
            fp / (fp + tn)
            if (fp + tn) > 0
            else 0.0
        )

        fnr = (
            fn / (fn + tp)
            if (fn + tp) > 0
            else 0.0
        )

        return {
            "accuracy": float(
                accuracy
            ),
            "precision": float(
                precision
            ),
            "recall": float(
                recall
            ),
            "f1": float(
                f1
            ),
            "roc_auc": float(
                roc_auc
            ),
            "fpr": float(
                fpr
            ),
            "fnr": float(
                fnr
            ),
            "confusion_matrix": (
                matrix.tolist()
            ),
        }

    # --------------------------------------------------------
    # PRINT METRICS
    # --------------------------------------------------------

    def print_metrics(
        self,
        name: str,
        metrics: Dict,
    ) -> None:

        print()
        print(name)
        print("-" * 50)

        print(
            f"Accuracy  : "
            f"{metrics['accuracy']:.4f}"
        )

        print(
            f"Precision : "
            f"{metrics['precision']:.4f}"
        )

        print(
            f"Recall    : "
            f"{metrics['recall']:.4f}"
        )

        print(
            f"F1        : "
            f"{metrics['f1']:.4f}"
        )

        print(
            f"ROC-AUC   : "
            f"{metrics['roc_auc']:.4f}"
        )

        print(
            f"FPR       : "
            f"{metrics['fpr']:.4f}"
        )

        print(
            f"FNR       : "
            f"{metrics['fnr']:.4f}"
        )

        print(
            "Confusion Matrix:"
        )

        print(
            np.asarray(
                metrics["confusion_matrix"]
            )
        )

    # --------------------------------------------------------
    # COMPARE MODELS
    # --------------------------------------------------------

    def compare_models(
        self,
        current: Dict,
        candidate: Dict,
    ) -> Dict:

        f1_change = (
            candidate["f1"]
            - current["f1"]
        )

        recall_change = (
            candidate["recall"]
            - current["recall"]
        )

        fpr_change = (
            candidate["fpr"]
            - current["fpr"]
        )

        roc_auc_change = (
            candidate["roc_auc"]
            - current["roc_auc"]
        )

        fnr_change = (
            candidate["fnr"]
            - current["fnr"]
        )

        f1_pass = (
            f1_change
            >= MIN_F1_CHANGE
        )

        recall_pass = (
            recall_change
            >= MIN_RECALL_CHANGE
        )

        fpr_pass = (
            fpr_change
            <= MAX_FPR_INCREASE
        )

        roc_auc_pass = (
            roc_auc_change
            >= -MAX_ROC_AUC_DECREASE
        )

        promote = (
            f1_pass
            and recall_pass
            and fpr_pass
            and roc_auc_pass
        )

        return {
            "f1_change": float(
                f1_change
            ),
            "recall_change": float(
                recall_change
            ),
            "fpr_change": float(
                fpr_change
            ),
            "roc_auc_change": float(
                roc_auc_change
            ),
            "fnr_change": float(
                fnr_change
            ),
            "f1_pass": f1_pass,
            "recall_pass": recall_pass,
            "fpr_pass": fpr_pass,
            "roc_auc_pass": roc_auc_pass,
            "promote": promote,
        }

    # --------------------------------------------------------
    # PRINT COMPARISON
    # --------------------------------------------------------

    def print_comparison(
        self,
        comparison: Dict,
    ) -> None:

        print()
        print(
            "MODEL COMPARISON"
        )
        print("-" * 50)

        print(
            f"F1 change      : "
            f"{comparison['f1_change']:+.4f}"
        )

        print(
            f"Recall change  : "
            f"{comparison['recall_change']:+.4f}"
        )

        print(
            f"FPR change     : "
            f"{comparison['fpr_change']:+.4f}"
        )

        print(
            f"ROC-AUC change : "
            f"{comparison['roc_auc_change']:+.4f}"
        )

        print(
            f"FNR change     : "
            f"{comparison['fnr_change']:+.4f}"
        )

        print()
        print(
            "PROMOTION GATE"
        )

        print(
            f"F1 requirement       : "
            f"{'PASS' if comparison['f1_pass'] else 'FAIL'}"
        )

        print(
            f"Recall requirement   : "
            f"{'PASS' if comparison['recall_pass'] else 'FAIL'}"
        )

        print(
            f"FPR requirement      : "
            f"{'PASS' if comparison['fpr_pass'] else 'FAIL'}"
        )

        print(
            f"ROC-AUC requirement  : "
            f"{'PASS' if comparison['roc_auc_pass'] else 'FAIL'}"
        )

        print()

        if comparison["promote"]:

            print(
                "PROMOTION DECISION: PROMOTE"
            )

        else:

            print(
                "PROMOTION DECISION: REJECT"
            )

    # --------------------------------------------------------
    # EVALUATION PIPELINE
    # --------------------------------------------------------

    def evaluate(self) -> Dict:

        # ----------------------------------------------------
        # IMPORTANT:
        # Dataset manager recreates the SAME deterministic
        # 80/20 split used during retraining.
        #
        # Candidate was trained only on X_train + feedback.
        # Therefore X_test remains untouched.
        # ----------------------------------------------------

        dataset = (
            self.dataset_manager
            .prepare_dataset()
        )

        X_test = dataset[
            "X_test"
        ]

        y_test = dataset[
            "y_test"
        ]

        print()
        print(
            "EVALUATION DATASET"
        )
        print("-" * 50)

        print(
            f"Test samples : "
            f"{len(y_test)}"
        )

        print(
            f"Features     : "
            f"{X_test.shape[1]}"
        )

        print(
            "This is the untouched fixed "
            "test set."
        )

        # ----------------------------------------------------
        # LOAD MODELS
        # ----------------------------------------------------

        (
            current_model,
            candidate_model,
        ) = self.load_models()

        # ----------------------------------------------------
        # EVALUATE CURRENT MODEL
        # ----------------------------------------------------

        current_metrics = (
            self.evaluate_model(
                current_model,
                X_test,
                y_test,
            )
        )

        self.print_metrics(
            "CURRENT MODEL",
            current_metrics,
        )

        # ----------------------------------------------------
        # EVALUATE CANDIDATE MODEL
        # ----------------------------------------------------

        candidate_metrics = (
            self.evaluate_model(
                candidate_model,
                X_test,
                y_test,
            )
        )

        self.print_metrics(
            "CANDIDATE MODEL",
            candidate_metrics,
        )

        # ----------------------------------------------------
        # COMPARE
        # ----------------------------------------------------

        comparison = (
            self.compare_models(
                current_metrics,
                candidate_metrics,
            )
        )

        self.print_comparison(
            comparison
        )

        return {
            "current": current_metrics,
            "candidate": candidate_metrics,
            "comparison": comparison,
        }


# ============================================================
# STANDALONE TEST
# ============================================================

def main() -> None:

    print()
    print("=" * 60)
    print(
        "ATHS MODEL EVALUATION TEST"
    )
    print("=" * 60)

    evaluator = (
        ATHSModelEvaluator()
    )

    result = evaluator.evaluate()

    print()
    print("=" * 60)
    print(
        "ATHS MODEL EVALUATION TEST COMPLETE"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()