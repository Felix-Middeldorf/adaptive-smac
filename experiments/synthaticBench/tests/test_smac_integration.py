"""End-to-end tests of instance-aware SMAC optimization."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from carps.utils.running import make_optimizer, make_problem
from omegaconf import OmegaConf
from smac import AlgorithmConfigurationFacade, Scenario

from experiments.synthaticBench.smac_instances import (
    DepthPolicyCallback,
    FixedMinSamplesLeafCallback,
    PROBLEM_CONFIG_ROOT,
    SYNTHACTIC_ROOT,
    depth_for_trial,
    generate_instance_map,
    load_problem,
    make_instance_features,
    make_target_function,
)


class SmacIntegrationTests(unittest.TestCase):
    def test_depth_policy_boundaries(self) -> None:
        self.assertEqual(
            [depth_for_trial("rotate_10", n) for n in (0, 9, 10, 19, 20, 30)],
            [4, 4, 8, 8, 20, 4],
        )
        self.assertEqual(
            [depth_for_trial("staged_80_50_rest", n) for n in (0, 79, 80, 129, 130)],
            [4, 4, 8, 8, 20],
        )

    def test_smac_runs_on_multiple_instances_with_adaptive_depth(self) -> None:
        problem = load_problem()
        instance_map = generate_instance_map(3, seed=17)
        target = make_target_function(problem, instance_map)
        with TemporaryDirectory() as output_directory:
            scenario = Scenario(
                name="synthactic-instance-integration-test",
                configspace=problem.configspace,
                instances=list(instance_map),
                instance_features=make_instance_features(instance_map),
                deterministic=True,
                n_trials=15,
                seed=3,
                output_directory=Path(output_directory),
            )
            callback = DepthPolicyCallback("rotate_10")
            model = AlgorithmConfigurationFacade.get_model(scenario=scenario, max_depth=4)
            smac = AlgorithmConfigurationFacade(
                scenario=scenario,
                target_function=target,
                model=model,
                callbacks=[callback],
                overwrite=True,
            )
            incumbent = smac.optimize()

        incumbent.check_valid_configuration()
        self.assertEqual(len(smac.runhistory), scenario.n_trials)
        evaluated_instances = {key.instance for key in smac.runhistory}
        self.assertTrue(evaluated_instances.issubset(instance_map))
        self.assertGreaterEqual(len(evaluated_instances), 2)
        self.assertEqual(callback.depth_changes[0], (0, 4))
        self.assertEqual(callback.last_depth, 8)

    def test_carps_wrapper_uses_custom_min_samples_leaf(self) -> None:
        problem_cfg = OmegaConf.load(PROBLEM_CONFIG_ROOT / "O1-DeterministicObjective.yaml")
        problem = make_problem(problem_cfg)
        instance_map = {"i0": 0.0, "i1": 1.0, "i2": -1.0}
        problem.set_instances(instance_map)

        optimizer_cfg = OmegaConf.load(SYNTHACTIC_ROOT / "config" / "smac20-ac.yml")
        optimizer_cfg.merge_with(problem_cfg)
        callback = FixedMinSamplesLeafCallback(7)

        with TemporaryDirectory() as output_directory:
            optimizer_cfg.seed = 3
            optimizer_cfg.outdir = output_directory
            optimizer_cfg.task.n_trials = 15
            scenario_cfg = optimizer_cfg.optimizer.smac_cfg.scenario
            scenario_cfg.instances = list(instance_map)
            scenario_cfg.instance_features = make_instance_features(instance_map)
            optimizer_cfg.optimizer.smac_cfg.smac_kwargs.callbacks = OmegaConf.create(
                [callback],
                flags={"allow_objects": True},
            )

            optimizer = make_optimizer(optimizer_cfg, problem)
            optimizer.run()

        model = optimizer.solver._config_selector._model
        self.assertTrue(callback.observed_values)
        self.assertEqual(set(callback.observed_values), {7})
        self.assertEqual(model._rf_opts["min_samples_leaf"], 7)
        self.assertEqual(len(optimizer.solver.runhistory), 15)
        self.assertEqual(
            {key.instance for key in optimizer.solver.runhistory},
            set(instance_map),
        )


if __name__ == "__main__":
    unittest.main()
