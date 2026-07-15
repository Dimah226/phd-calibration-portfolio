import numpy as np
import unittest

from src.calibrate import correct_undersampled_probabilities
from src.metrics import (
    adaptive_calibration_error,
    expected_calibration_error,
    top_fraction_calibration_error,
)


class CalibrationTests(unittest.TestCase):
    def test_perfectly_calibrated_grouped_predictions_have_zero_error(self):
        y = np.array([0, 0, 0, 1, 0, 1, 1, 1])
        p = np.array([0.25] * 4 + [0.75] * 4)
        self.assertAlmostEqual(expected_calibration_error(y, p, n_bins=4), 0.0)
        self.assertAlmostEqual(adaptive_calibration_error(y, p, n_bins=2), 0.0)

    def test_prior_correction_inverts_majority_undersampling(self):
        population_p = np.array([0.001, 0.01, 0.10, 0.50, 0.90])
        beta = 0.05
        sampled_p = population_p / (population_p + beta * (1 - population_p))
        recovered = correct_undersampled_probabilities(sampled_p, beta)
        np.testing.assert_allclose(recovered, population_p, rtol=1e-10, atol=1e-12)

    def test_top_fraction_uses_exact_queue_size_and_validates_fraction(self):
        y = np.array([0, 0, 0, 1, 1])
        p = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
        self.assertAlmostEqual(top_fraction_calibration_error(y, p, fraction=0.4), 0.15)
        with self.assertRaises(ValueError):
            top_fraction_calibration_error(y, p, fraction=0.0)


if __name__ == "__main__":
    unittest.main()
