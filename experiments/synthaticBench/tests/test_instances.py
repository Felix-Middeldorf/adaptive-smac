"""Focused tests for SynthACticBench's instance semantics."""

from __future__ import annotations

import numpy as np
import unittest

from experiments.synthaticBench.smac_instances import (
    generate_instance_map,
    load_problem,
    make_instance_features,
    make_target_function,
)


class InstanceTests(unittest.TestCase):
    def test_problem_loads_and_exposes_configspace(self) -> None:
        problem = load_problem()
        config = problem.configspace.sample_configuration()
        self.assertEqual(len(problem.configspace), 10)
        self.assertEqual(set(config), set(problem.configspace))

    def test_instance_generation_is_reproducible_and_seeded(self) -> None:
        first = generate_instance_map(5, seed=7)
        self.assertEqual(first, generate_instance_map(5, seed=7))
        self.assertNotEqual(first, generate_instance_map(5, seed=8))
        self.assertEqual(list(first), ["i0", "i1", "i2", "i3", "i4"])

    def test_invalid_instance_generation_is_rejected(self) -> None:
        for n_instances, std in [(0, 2.0), (-1, 2.0), (2, -0.1)]:
            with self.subTest(n_instances=n_instances, std=std):
                with self.assertRaises(ValueError):
                    generate_instance_map(n_instances, seed=0, std=std)

    def test_instance_features_preserve_offsets(self) -> None:
        self.assertEqual(
            make_instance_features({"i0": -1.5, "i1": 2}),
            {"i0": [-1.5], "i1": [2.0]},
        )

    def test_target_applies_the_requested_instance_offset(self) -> None:
        problem = load_problem()
        target = make_target_function(problem, {"easy": -3.5, "hard": 8.25})
        config = problem.configspace.get_default_configuration()
        easy = target(config, instance="easy", seed=1)
        hard = target(config, instance="hard", seed=999)
        self.assertTrue(np.isfinite(easy))
        self.assertAlmostEqual(hard - easy, 11.75)

    def test_benchmark_is_deterministic_across_smac_seeds(self) -> None:
        problem = load_problem()
        target = make_target_function(problem, {"i0": 0.0})
        config = problem.configspace.sample_configuration()
        self.assertAlmostEqual(
            target(config, instance="i0", seed=1),
            target(config, instance="i0", seed=2),
        )

    def test_unknown_instance_is_rejected(self) -> None:
        problem = load_problem()
        target = make_target_function(problem, {"i0": 0.0})
        with self.assertRaisesRegex(KeyError, "Unknown instance"):
            target(problem.configspace.sample_configuration(), instance="missing")

    def test_scalar_adapter_rejects_multi_objective_problem(self) -> None:
        problem = load_problem("O3-MultipleObjectives.yaml")
        with self.assertRaisesRegex(TypeError, "scalar objectives only"):
            make_target_function(problem, {"i0": 0.0})


if __name__ == "__main__":
    unittest.main()
