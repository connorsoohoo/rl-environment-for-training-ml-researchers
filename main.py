import asyncio

from problem_runner import main
from problems import (
    ComplexMLTrainingWithPySpy,
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

    # 40% pass rate over 10 runs.
    problem = ComplexMLTrainingWithPySpy(include_pyspy=True)

    # Configuration
    NUM_RUNS = 1  # Takes about 20 minutes to run with this set to 10, set to 1 for just doing coding.
    CONCURRENT = False
    VERBOSE = False  # Set to True to see agent's reasoning and tool usage

    # Run the test suite
    asyncio.run(
        main(problem=problem, num_runs=NUM_RUNS, concurrent=CONCURRENT, verbose=VERBOSE)
    )
