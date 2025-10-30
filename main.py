import asyncio

from problem_runner import main
from problems import (
    ComplexMLTrainingWithProfiler,
)

# Uncomment to use other problems:
# from problems import (
#     MathCalculationProblem,
#     MLTrainingOptimizationNumPy,
#     MLTrainingOptimizationPyTorch,
#     MLTrainingOptimizationComplex,
# )

if __name__ == "__main__":
    # Create the problem instance
    # problem = MathCalculationProblem()
    # problem = MLTrainingOptimizationNumPy()
    # problem = MLTrainingOptimizationPyTorch()

    # 40% pass rate over 10 runs with speedup factor target of 33.
    problem = ComplexMLTrainingWithProfiler(
        include_profiler=True,
        # Setting intermediate_answer to True improves pass rate to 60% over 10 runs.
        # this is because it helps the model test and validate smaller chunks of code and nudges the model to take on this practice.
        # We want the model to learn this skill on its own, so normally we want to set it to False.
        include_intermediate_answer=True,
    )

    # Configuration
    NUM_RUNS = 10  # Takes about 20 minutes to run with this set to 10, set to 1 for just doing coding.
    CONCURRENT = False  # False because I don't want to pay money for multiple concurrent Anthropic API calls.
    VERBOSE = False  # Set to True to see agent's reasoning and tool usage

    # Run the test suite
    asyncio.run(
        main(problem=problem, num_runs=NUM_RUNS, concurrent=CONCURRENT, verbose=VERBOSE)
    )
