"""
ATHS Threat Fusion Engine

Combines:
    1. ML threat probability
    2. Rule-based threat score
    3. Semantic threat score

into a single threat score and security decision.

Decision levels:
    ALLOW
    REVIEW
    BLOCK
"""

from typing import Any, Dict


class ThreatFusion:
    """
    Combines multiple threat signals into a final threat score.
    """

    # ---------------------------------------------------------
    # Fusion weights
    # ---------------------------------------------------------
    #
    # ML is given the highest weight because the trained
    # classifier provides the primary statistical signal.
    #
    # Rule detection provides deterministic security evidence.
    #
    # Semantic detection provides supporting evidence based
    # on similarity to known malicious prompts.
    #
    ML_WEIGHT = 0.50
    RULE_WEIGHT = 0.35
    SEMANTIC_WEIGHT = 0.15

    # ---------------------------------------------------------
    # Decision thresholds
    # ---------------------------------------------------------

    ALLOW_THRESHOLD = 0.30
    BLOCK_THRESHOLD = 0.75

    def __init__(
        self,
        ml_weight: float = ML_WEIGHT,
        rule_weight: float = RULE_WEIGHT,
        semantic_weight: float = SEMANTIC_WEIGHT,
        allow_threshold: float = ALLOW_THRESHOLD,
        block_threshold: float = BLOCK_THRESHOLD,
    ):
        self.ml_weight = ml_weight
        self.rule_weight = rule_weight
        self.semantic_weight = semantic_weight

        self.allow_threshold = allow_threshold
        self.block_threshold = block_threshold

        self._validate_configuration()

    def _validate_configuration(self):
        """
        Validate fusion weights and thresholds.
        """

        weights = [
            self.ml_weight,
            self.rule_weight,
            self.semantic_weight,
        ]

        # -----------------------------------------------------
        # Validate weights
        # -----------------------------------------------------

        if any(weight < 0 for weight in weights):
            raise ValueError(
                "Fusion weights cannot be negative."
            )

        weight_sum = sum(weights)

        if abs(weight_sum - 1.0) > 1e-6:
            raise ValueError(
                "Fusion weights must sum to 1.0. "
                f"Current sum: {weight_sum:.4f}"
            )

        # -----------------------------------------------------
        # Validate thresholds
        # -----------------------------------------------------

        if not 0.0 <= self.allow_threshold <= 1.0:
            raise ValueError(
                "allow_threshold must be between 0 and 1."
            )

        if not 0.0 <= self.block_threshold <= 1.0:
            raise ValueError(
                "block_threshold must be between 0 and 1."
            )

        if self.allow_threshold >= self.block_threshold:
            raise ValueError(
                "allow_threshold must be lower than "
                "block_threshold."
            )

    @staticmethod
    def _validate_score(
        score: float,
        name: str,
    ) -> float:
        """
        Validate and normalize an individual score.

        All threat scores must be in the range [0, 1].
        """

        try:
            score = float(score)
        except (TypeError, ValueError):
            raise ValueError(
                f"{name} must be a numeric value."
            )

        if not 0.0 <= score <= 1.0:
            raise ValueError(
                f"{name} must be between 0 and 1. "
                f"Received: {score}"
            )

        return score

    def calculate_threat_score(
        self,
        ml_probability: float,
        rule_score: float,
        semantic_score: float,
    ) -> float:
        """
        Calculate the weighted final threat score.

        Formula:

            Threat Score =
                ML Weight       × ML Probability
                + Rule Weight   × Rule Score
                + Semantic Weight × Semantic Score

        All input scores must be in [0, 1].
        """

        # -----------------------------------------------------
        # Validate input scores
        # -----------------------------------------------------

        ml_probability = self._validate_score(
            ml_probability,
            "ml_probability",
        )

        rule_score = self._validate_score(
            rule_score,
            "rule_score",
        )

        semantic_score = self._validate_score(
            semantic_score,
            "semantic_score",
        )

        # -----------------------------------------------------
        # Weighted fusion
        # -----------------------------------------------------

        threat_score = (
            self.ml_weight * ml_probability
            + self.rule_weight * rule_score
            + self.semantic_weight * semantic_score
        )

        return round(threat_score, 4)

    def get_decision(
        self,
        threat_score: float,
    ) -> str:
        """
        Convert the final threat score into a security decision.

        Decision logic:

            score < 0.30
                -> ALLOW

            0.30 <= score < 0.75
                -> REVIEW

            score >= 0.75
                -> BLOCK
        """

        threat_score = self._validate_score(
            threat_score,
            "threat_score",
        )

        if threat_score < self.allow_threshold:
            return "ALLOW"

        if threat_score >= self.block_threshold:
            return "BLOCK"

        return "REVIEW"

    def fuse(
        self,
        ml_probability: float,
        rule_score: float,
        semantic_score: float,
    ) -> Dict[str, Any]:
        """
        Combine all threat signals.

        Returns a structured result suitable for the
        Threat Detection Engine and dashboard.
        """

        # -----------------------------------------------------
        # Calculate final threat score
        # -----------------------------------------------------

        threat_score = self.calculate_threat_score(
            ml_probability=ml_probability,
            rule_score=rule_score,
            semantic_score=semantic_score,
        )

        # -----------------------------------------------------
        # Determine security decision
        # -----------------------------------------------------

        decision = self.get_decision(
            threat_score
        )

        # -----------------------------------------------------
        # Return structured result
        # -----------------------------------------------------

        return {
            "ml_probability": round(
                float(ml_probability),
                4,
            ),
            "rule_score": round(
                float(rule_score),
                4,
            ),
            "semantic_score": round(
                float(semantic_score),
                4,
            ),
            "threat_score": threat_score,
            "decision": decision,
            "weights": {
                "ml": self.ml_weight,
                "rules": self.rule_weight,
                "semantic": self.semantic_weight,
            },
            "thresholds": {
                "allow": self.allow_threshold,
                "block": self.block_threshold,
            },
        }


def print_result(
    title: str,
    result: Dict,
):
    """
    Pretty-print a fusion result.
    """

    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

    print(
        f"ML probability : "
        f"{result['ml_probability']:.4f}"
    )

    print(
        f"Rule score     : "
        f"{result['rule_score']:.4f}"
    )

    print(
        f"Semantic score : "
        f"{result['semantic_score']:.4f}"
    )

    print(
        f"Threat score   : "
        f"{result['threat_score']:.4f}"
    )

    print(
        f"Decision       : "
        f"{result['decision']}"
    )


def main():
    """
    Standalone fusion tests.
    """

    fusion = ThreatFusion()

    print("=" * 70)
    print("ATHS THREAT FUSION TEST")
    print("=" * 70)

    # ---------------------------------------------------------
    # TEST 1: Clearly benign
    # ---------------------------------------------------------

    benign_result = fusion.fuse(
        ml_probability=0.10,
        rule_score=0.00,
        semantic_score=0.10,
    )

    print_result(
        "BENIGN TEST",
        benign_result,
    )

    # ---------------------------------------------------------
    # TEST 2: Uncertain / medium risk
    # ---------------------------------------------------------

    review_result = fusion.fuse(
        ml_probability=0.55,
        rule_score=0.40,
        semantic_score=0.50,
    )

    print_result(
        "REVIEW TEST",
        review_result,
    )

    # ---------------------------------------------------------
    # TEST 3: Clearly malicious
    # ---------------------------------------------------------

    malicious_result = fusion.fuse(
        ml_probability=0.95,
        rule_score=0.90,
        semantic_score=0.85,
    )

    print_result(
        "MALICIOUS TEST",
        malicious_result,
    )

    # ---------------------------------------------------------
    # Display configuration
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("FUSION CONFIGURATION")
    print("=" * 70)

    print(
        f"ML weight       : "
        f"{fusion.ml_weight:.2f}"
    )

    print(
        f"Rule weight     : "
        f"{fusion.rule_weight:.2f}"
    )

    print(
        f"Semantic weight : "
        f"{fusion.semantic_weight:.2f}"
    )

    print(
        f"ALLOW threshold : "
        f"{fusion.allow_threshold:.2f}"
    )

    print(
        f"BLOCK threshold : "
        f"{fusion.block_threshold:.2f}"
    )

    print("\n" + "=" * 70)
    print("ATHS THREAT FUSION TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()