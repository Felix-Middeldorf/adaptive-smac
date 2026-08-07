"""Definition for the Clasp Queens Codex-controlled RF experiment."""

from pathlib import Path

from llm_policy import ExperimentDefinition


HERE = Path(__file__).resolve().parent
DEFINITION = ExperimentDefinition(
    benchmark_key="clasp_queens",
    initials="cq_llm",
    directory=HERE,
    initial_directory=HERE.parent / "01_initial_cq",
)
