"""Environment compatibility checks for the experiment interpreter."""

from __future__ import annotations

from importlib.metadata import version
import unittest

from experiments.synthaticBench.smac_instances import load_problem


class EnvironmentTests(unittest.TestCase):
    def test_expected_dependency_versions_are_importable(self) -> None:
        self.assertEqual(version("carps"), "0.1.1")
        self.assertEqual(version("smac"), "2.4.0")

    def test_synthacticbench_has_a_working_problem_api(self) -> None:
        self.assertTrue(callable(load_problem().evaluate))


if __name__ == "__main__":
    unittest.main()
