from collections.abc import Callable
from contextlib import redirect_stdout
from contextvars import ContextVar
from io import StringIO
from typing import Any, TypedDict

from anthropic.types import ToolUnionParam

# Context variable to track current run's output directory
current_run_context: ContextVar[dict[str, Any] | None] = ContextVar(
    "current_run_context", default=None
)


class PythonExpressionToolResult(TypedDict):
    result: Any
    error: str | None


class SubmitAnswerToolResult(TypedDict):
    answer: Any
    submitted: bool


class ReadFileToolResult(TypedDict):
    content: str | None
    error: str | None


class ProfileToolResult(TypedDict):
    trace_file: str | None
    summary: str | None
    error: str | None


def python_expression_tool(expression: str) -> PythonExpressionToolResult:
    """
    Tool that evaluates Python expressions using exec.
    Use print(...) to emit output; stdout will be captured and returned.
    """
    try:
        namespace: dict[str, Any] = {}
        stdout = StringIO()
        with redirect_stdout(stdout):
            exec(expression, namespace, namespace)
        return {"result": stdout.getvalue(), "error": None}
    except KeyboardInterrupt:
        raise
    except Exception as e:
        return {"result": None, "error": str(e)}


def submit_answer_tool(answer: Any) -> SubmitAnswerToolResult:
    """
    Tool for submitting the final answer.
    """
    return {"answer": answer, "submitted": True}


def submit_intermediate_answer_tool(answer: Any) -> SubmitAnswerToolResult:
    """
    Tool for submitting an intermediate answer, that way it can be evaluated without
    affecting the final submission.
    """
    return {"answer": answer, "submitted": False}


def read_file_tool(file_path: str) -> ReadFileToolResult:
    """
    Tool that reads a file from the problem_data directory.
    Only allows reading files from the problem_data folder for security.
    """
    from pathlib import Path

    try:
        # Get the base directory (where tools.py is located)
        base_dir = Path(__file__).parent
        problem_data_dir = base_dir / "problem_data"

        # Normalize the file path
        requested_path = Path(file_path)

        # If it's a relative path, treat it as relative to problem_data
        if not requested_path.is_absolute():
            full_path = problem_data_dir / requested_path
        else:
            full_path = requested_path

        # Resolve to canonical path
        full_path = full_path.resolve()

        # Security check: ensure the path is within problem_data directory
        if not full_path.is_relative_to(problem_data_dir.resolve()):
            return {
                "content": None,
                "error": f"Access denied: {file_path} is outside problem_data directory",
            }

        # Check if file exists
        if not full_path.exists():
            return {"content": None, "error": f"File not found: {file_path}"}

        # Read the file
        with open(full_path, encoding="utf-8") as f:
            content = f.read()

        return {"content": content, "error": None}

    except KeyboardInterrupt:
        raise
    except Exception as e:
        return {"content": None, "error": f"Error reading file: {str(e)}"}


def profile_tool(file_path: str) -> ProfileToolResult:
    """
    Tool that profiles a Python file and generates a trace file using cProfile.
    Only allows profiling files from the problem_data directory for security.

    Generates a .prof file that can be analyzed with pstats or visualization tools.
    """
    import pstats
    import subprocess
    from pathlib import Path

    try:
        # Get the base directory (where tools.py is located)
        base_dir = Path(__file__).parent
        problem_data_dir = base_dir / "problem_data"

        # Normalize the file path
        requested_path = Path(file_path)

        # If it's a relative path, treat it as relative to problem_data
        if not requested_path.is_absolute():
            full_path = problem_data_dir / requested_path
        else:
            full_path = requested_path

        # Resolve to canonical path
        full_path = full_path.resolve()

        # Security check: ensure the path is within problem_data directory
        if not full_path.is_relative_to(problem_data_dir.resolve()):
            return {
                "trace_file": None,
                "summary": None,
                "error": f"Access denied: {file_path} is outside problem_data directory",
            }

        # Check if file exists
        if not full_path.exists():
            return {
                "trace_file": None,
                "summary": None,
                "error": f"File not found: {file_path}",
            }

        # Get output directory from context
        ctx = current_run_context.get()
        if ctx is None:
            # Fallback if context not set
            output_dir = Path(".")
            run_id = 0
        else:
            output_dir = Path(ctx["output_dir"])
            run_id = ctx["run_id"]

        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)

        # Use cProfile for profiling
        stats_file = str(
            output_dir / f"profile_{Path(file_path).stem}_run{run_id}.prof"
        )

        try:
            # Run with cProfile
            result = subprocess.run(
                ["python", "-m", "cProfile", "-o", stats_file, str(full_path)],
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
            )

            if result.returncode != 0:
                return {
                    "trace_file": None,
                    "summary": None,
                    "error": f"cProfile profiling failed with error:\n{result.stderr}",
                }

            # Generate a text summary
            summary_io = StringIO()
            stats = pstats.Stats(stats_file, stream=summary_io)
            stats.sort_stats("cumulative")
            stats.print_stats(20)  # Top 20 functions

            summary = f"""Profiling completed using cProfile.

Profile file: {stats_file}

Top 20 functions by cumulative time:
{summary_io.getvalue()}

Program output:
{result.stdout}
"""

            return {"trace_file": stats_file, "summary": summary, "error": None}

        except subprocess.TimeoutExpired:
            return {
                "trace_file": None,
                "summary": None,
                "error": "Profiling timed out after 5 minutes",
            }

    except KeyboardInterrupt:
        raise
    except Exception as e:
        return {
            "trace_file": None,
            "summary": None,
            "error": f"Error profiling file: {str(e)}",
        }


# Tool schemas
PYTHON_EXPRESSION_SCHEMA: ToolUnionParam = {
    "name": "python_expression",
    "description": "Evaluates a Python expression",
    "input_schema": {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "Will be passed to exec(). Use print() to output something. Returns stdout. ",
            }
        },
        "required": ["expression"],
    },
}

SUBMIT_ANSWER_SCHEMA: ToolUnionParam = {
    "name": "submit_answer",
    "description": "Submit the final answer",
    "input_schema": {
        "type": "object",
        "properties": {"answer": {"description": "The final answer to submit"}},
        "required": ["answer"],
    },
}

SUBMIT_INTERMEDIATE_ANSWER_SCHEMA: ToolUnionParam = {
    "name": "submit_intermediate_answer",
    "description": "Submit an intermediate answer for evaluation without affecting the final submission. Use this to test your optimized file before making the final submission.",
    "input_schema": {
        "type": "object",
        "properties": {
            "answer": {"description": "The intermediate answer to evaluate"}
        },
        "required": ["answer"],
    },
}

READ_FILE_SCHEMA: ToolUnionParam = {
    "name": "read_file",
    "description": "Read a file from the problem_data directory. Provide a relative path like 'slow_ml_training.py'.",
    "input_schema": {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Path to the file relative to the problem_data directory",
            }
        },
        "required": ["file_path"],
    },
}

PROFILE_SCHEMA: ToolUnionParam = {
    "name": "profile",
    "description": "Profile a Python file and generate a trace file for performance analysis using cProfile. Generates detailed timing information about function calls to identify bottlenecks.",
    "input_schema": {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Path to the Python file to profile (relative to the problem_data directory)",
            }
        },
        "required": ["file_path"],
    },
}


class BasicToolset:
    """Common tools for math/logic problems."""

    @staticmethod
    def get_tools(
        include_profiler: bool = True, include_intermediate_answer: bool = True
    ) -> list[ToolUnionParam]:
        """
        Get tool schemas for the toolset.

        Args:
            include_profiler: Whether to include the profiling tool. Default True.
            include_intermediate_answer: Whether to include the intermediate answer tool. Default True.
        """
        tools = [
            PYTHON_EXPRESSION_SCHEMA,
            SUBMIT_ANSWER_SCHEMA,
            READ_FILE_SCHEMA,
        ]
        if include_intermediate_answer:
            tools.append(SUBMIT_INTERMEDIATE_ANSWER_SCHEMA)
        if include_profiler:
            tools.append(PROFILE_SCHEMA)
        return tools

    @staticmethod
    def get_handlers(
        include_profiler: bool = True, include_intermediate_answer: bool = True
    ) -> dict[str, Callable[..., Any]]:
        """
        Get tool handlers for the toolset.

        Args:
            include_profiler: Whether to include the profiling handler. Default True.
            include_intermediate_answer: Whether to include the intermediate answer handler. Default True.
        """
        handlers = {
            "python_expression": python_expression_tool,
            "submit_answer": submit_answer_tool,
            "read_file": read_file_tool,
        }
        if include_intermediate_answer:
            handlers["submit_intermediate_answer"] = submit_intermediate_answer_tool
        if include_profiler:
            handlers["profile"] = profile_tool
        return handlers
