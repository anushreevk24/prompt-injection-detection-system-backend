from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from data.feedback_store import ATHSFeedbackStore
from detection.feature_loader import ML_FEATURE_COLUMNS
from preprocessing.pipeline import PreprocessingPipeline
from retraining.dataset_manager import ATHSDatasetManager


# ============================================================
# ATHS MODEL RETRAINING CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

CANDIDATE_MODEL_PATH = (
    BASE_DIR
    / "models"
    / "random_forest_candidate.joblib"
)

RANDOM_STATE = 42

MIN_FEEDBACK_SAMPLES = 1


# ============================================================
# ATHS MODEL RETRAINER
# ============================================================

class ATHSModelRetrainer:

    def __init__(self) -> None:

        print("Initializing ATHS Model Retrainer...")

        self.candidate_model_path = (
            CANDIDATE_MODEL_PATH
        )

        self.dataset_manager = (
            ATHSDatasetManager()
        )

        self.feedback_store = (
            ATHSFeedbackStore()
        )

        print("ATHS Model Retrainer initialized.")

    # --------------------------------------------------------
    # LOAD AND PREPARE DATASET
    # --------------------------------------------------------

    def prepare_dataset(self) -> Dict:

        dataset = (
            self.dataset_manager
            .prepare_dataset()
        )

        return dataset

    # --------------------------------------------------------
    # PREPROCESS HUMAN FEEDBACK
    # --------------------------------------------------------

    def preprocess_feedback(
        self,
        feedback: List[Dict],
    ) -> Tuple[np.ndarray, np.ndarray]:

        if not feedback:

            return (
                np.empty(
                    (
                        0,
                        len(ML_FEATURE_COLUMNS),
                    ),
                    dtype=np.float32,
                ),
                np.empty(
                    (0,),
                    dtype=int,
                ),
            )

        print()
        print(
            "PREPROCESSING HUMAN-LABELED PROMPTS"
        )
        print("-" * 50)

        # Embeddings are not required for ML retraining.
        pipeline = PreprocessingPipeline(
            enable_embeddings=False
        )

        features = []
        labels = []

        for index, record in enumerate(
            feedback,
            start=1,
        ):

            text = str(
                record["text"]
            )

            human_label = int(
                record["human_label"]
            )

            processed = (
                pipeline.process_text(text)
            )

            feature_sources = {
                **processed["nlp"],
                **processed["obfuscation"],
                **processed["security_features"],
            }

            missing_features = [
                column
                for column in ML_FEATURE_COLUMNS
                if column not in feature_sources
            ]

            if missing_features:

                raise ValueError(
                    "Missing features during "
                    f"feedback preprocessing: "
                    f"{missing_features}"
                )

            feature_vector = [
                feature_sources[column]
                for column in ML_FEATURE_COLUMNS
            ]

            features.append(
                feature_vector
            )

            labels.append(
                human_label
            )

            print(
                f"  Processed "
                f"{index}/{len(feedback)} "
                f"| label={human_label}"
            )

        X_feedback = np.asarray(
            features,
            dtype=np.float32,
        )

        y_feedback = np.asarray(
            labels,
            dtype=int,
        )

        print()
        print(
            "Feedback preprocessing complete."
        )

        print(
            f"Feedback feature shape : "
            f"{X_feedback.shape}"
        )

        print(
            f"Feedback benign        : "
            f"{np.sum(y_feedback == 0)}"
        )

        print(
            f"Feedback malicious     : "
            f"{np.sum(y_feedback == 1)}"
        )

        return (
            X_feedback,
            y_feedback,
        )

    # --------------------------------------------------------
    # ADD FEEDBACK TO TRAINING DATA
    # --------------------------------------------------------

    def combine_training_data(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_feedback: np.ndarray,
        y_feedback: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:

        if len(X_feedback) == 0:

            return (
                X_train,
                y_train,
            )

        X_combined = np.vstack(
            [
                X_train,
                X_feedback,
            ]
        )

        y_combined = np.concatenate(
            [
                y_train,
                y_feedback,
            ]
        )

        print()
        print(
            "TRAINING DATA WITH FEEDBACK"
        )
        print("-" * 50)

        print(
            f"Original training samples : "
            f"{len(y_train)}"
        )

        print(
            f"Human feedback samples    : "
            f"{len(y_feedback)}"
        )

        print(
            f"Final training samples    : "
            f"{len(y_combined)}"
        )

        print(
            f"Benign                    : "
            f"{np.sum(y_combined == 0)}"
        )

        print(
            f"Malicious                 : "
            f"{np.sum(y_combined == 1)}"
        )

        return (
            X_combined,
            y_combined,
        )

    # --------------------------------------------------------
    # TRAIN RANDOM FOREST
    # --------------------------------------------------------

    def train_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> RandomForestClassifier:

        print()
        print(
            "TRAINING RANDOM FOREST"
        )
        print("-" * 50)

        model = RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

        model.fit(
            X_train,
            y_train,
        )

        print(
            "Random Forest training complete."
        )

        return model

    # --------------------------------------------------------
    # EVALUATE ON FIXED TEST SET
    # --------------------------------------------------------

    def evaluate_model(
        self,
        model: RandomForestClassifier,
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

        metrics = {
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

        print()
        print(
            "CANDIDATE MODEL EVALUATION"
        )
        print("-" * 50)

        print(
            f"Accuracy  : "
            f"{accuracy:.4f}"
        )

        print(
            f"Precision : "
            f"{precision:.4f}"
        )

        print(
            f"Recall    : "
            f"{recall:.4f}"
        )

        print(
            f"F1        : "
            f"{f1:.4f}"
        )

        print(
            f"ROC-AUC   : "
            f"{roc_auc:.4f}"
        )

        print(
            f"FPR       : "
            f"{fpr:.4f}"
        )

        print(
            f"FNR       : "
            f"{fnr:.4f}"
        )

        print()
        print(
            "Confusion Matrix:"
        )

        print(matrix)

        return metrics

    # --------------------------------------------------------
    # SAVE CANDIDATE MODEL
    # --------------------------------------------------------

    def save_candidate_model(
        self,
        model: RandomForestClassifier,
    ) -> None:

        self.candidate_model_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        joblib.dump(
            model,
            self.candidate_model_path,
        )

        print()
        print(
            "CANDIDATE MODEL SAVED"
        )
        print("-" * 50)

        print(
            self.candidate_model_path
        )

    # --------------------------------------------------------
    # COMPLETE RETRAINING
    # --------------------------------------------------------

    def retrain(self) -> Dict:

        dataset = (
            self.prepare_dataset()
        )

        X_train = dataset[
            "X_train"
        ]

        y_train = dataset[
            "y_train"
        ]

        X_test = dataset[
            "X_test"
        ]

        y_test = dataset[
            "y_test"
        ]

        feedback = dataset[
            "feedback"
        ]

        if len(feedback) < MIN_FEEDBACK_SAMPLES:

            print()
            print(
                "Not enough confirmed human "
                "feedback for retraining."
            )

            return {
                "status":
                    "insufficient_feedback",
                "feedback_samples":
                    len(feedback),
            }

        # ----------------------------------------------------
        # PREPROCESS ONLY CONFIRMED FEEDBACK
        # ----------------------------------------------------

        (
            X_feedback,
            y_feedback,
        ) = self.preprocess_feedback(
            feedback
        )

        # ----------------------------------------------------
        # ADD FEEDBACK ONLY TO TRAINING DATA
        #
        # IMPORTANT:
        # X_test and y_test are NEVER modified.
        # ----------------------------------------------------

        (
            X_train_combined,
            y_train_combined,
        ) = self.combine_training_data(
            X_train,
            y_train,
            X_feedback,
            y_feedback,
        )

        print()
        print(
            "DATA ISOLATION CHECK"
        )
        print("-" * 50)

        print(
            f"Training data : "
            f"{X_train_combined.shape}"
        )

        print(
            f"Fixed test    : "
            f"{X_test.shape}"
        )

        print(
            "Fixed test set is "
            "excluded from training."
        )

        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        model = self.train_model(
            X_train_combined,
            y_train_combined,
        )

        # ----------------------------------------------------
        # EVALUATE AGAINST FIXED TEST SET
        # ----------------------------------------------------

        metrics = self.evaluate_model(
            model,
            X_test,
            y_test,
        )

        # ----------------------------------------------------
        # SAVE CANDIDATE
        # ----------------------------------------------------

        self.save_candidate_model(
            model
        )

        return {
            "status":
                "candidate_created",
            "feedback_samples":
                len(feedback),
            "training_samples":
                len(y_train_combined),
            "test_samples":
                len(y_test),
            "metrics":
                metrics,
            "candidate_model":
                str(
                    self.candidate_model_path
                ),
        }


# ============================================================
# STANDALONE TEST
# ============================================================

def main() -> None:

    print()
    print("=" * 60)
    print(
        "ATHS MODEL RETRAINING TEST"
    )
    print("=" * 60)

    retrainer = (
        ATHSModelRetrainer()
    )

    result = retrainer.retrain()

    print()
    print(
        "RETRAINING RESULT"
    )
    print("-" * 50)

    print(
        f"Status : "
        f"{result['status']}"
    )

    if "feedback_samples" in result:

        print(
            f"Feedback samples : "
            f"{result['feedback_samples']}"
        )

    if "training_samples" in result:

        print(
            f"Training samples : "
            f"{result['training_samples']}"
        )

    if "test_samples" in result:

        print(
            f"Test samples     : "
            f"{result['test_samples']}"
        )

    if "metrics" in result:

        print()
        print(
            "Candidate F1     : "
            f"{result['metrics']['f1']:.4f}"
        )

        print(
            "Candidate Recall : "
            f"{result['metrics']['recall']:.4f}"
        )

        print(
            "Candidate FNR    : "
            f"{result['metrics']['fnr']:.4f}"
        )

    print()
    print("=" * 60)
    print(
        "ATHS MODEL RETRAINING TEST COMPLETE"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()