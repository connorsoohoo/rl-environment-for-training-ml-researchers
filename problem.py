from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from anthropic.types import ToolUnionParam


class Problem(ABC):
    """Abstract base class for agent tasks/problems."""

    def __init__(
        self,
        prompt: str,
        tools: list[ToolUnionParam],
        tool_handlers: dict[str, Callable[..., Any]],
        expected_answer: Any,
    ):
        self.prompt = prompt
        self.tools = tools
        self.tool_handlers = tool_handlers
        self.expected_answer = expected_answer

    @abstractmethod
    def grade(self, artifacts: dict[str, Any]) -> bool:
        """
        Grade the artifacts against the expected answer.
        Subclasses must implement this method.

        Args:
            artifacts: Dictionary containing artifacts from the agent's work
                      (e.g., {"result": value} for simple problems,
                       {"optimized_file": path} for optimization problems)

        Returns:
            bool: True if the result is correct, False otherwise
        """
        pass
