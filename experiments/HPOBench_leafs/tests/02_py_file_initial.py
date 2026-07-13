from hpobench.container.benchmarks.ml.rf_benchmark import RandomForestBenchmarkBB
from master_utils.hpobench_workaround import HPOBENCH_ML_INIT_PATCH

benchmark = RandomForestBenchmarkBB(
    task_id=10101,
    rng=0,
    #bind_str=HPOBENCH_ML_INIT_PATCH,
)

config = benchmark.get_configuration_space(seed=0).sample_configuration()

print("config:")
print(config)

result = benchmark.objective_function(
    configuration=config,
    rng=0,
)

print("result keys:", result.keys())
print("result:", result)
print("validation loss:", result["function_value"])