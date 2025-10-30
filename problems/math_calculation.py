from typing import Any

from problem import Problem
from tools import BasicToolset


class MathCalculationProblem(Problem):
    """A problem that requires exact numeric equality for grading."""

    def __init__(self):
        super().__init__(
            prompt="Calculate (2^10 + 3^5) * 7 - 100. Use the python_expression tool and then submit the answer.",
            tools=BasicToolset.get_tools(),
            tool_handlers=BasicToolset.get_handlers(),
            expected_answer=8769,
        )

    def grade(self, artifacts: dict[str, Any]) -> bool:
        """Check if result exactly equals expected answer."""
        result = artifacts.get("result")
        return result == self.expected_answer
