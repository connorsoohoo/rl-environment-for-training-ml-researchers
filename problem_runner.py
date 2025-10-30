import asyncio
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from anthropic import AsyncAnthropic
from anthropic.types import MessageParam, ToolUnionParam

from problem import Problem
from tools import current_run_context

MAX_TOKENS = 5000


async def run_agent_loop(
    prompt: str,
    tools: list[ToolUnionParam],
    tool_handlers: dict[str, Callable[..., Any]],
    max_steps: int = 20,
    model: str = "claude-haiku-4-5",
    verbose: bool = True,
) -> Any | None:
    """
    Runs an agent loop with the given prompt and tools.

    Args:
        prompt: The initial prompt for the agent
        tools: List of tool definitions for Anthropic API
        tool_handlers: Dictionary mapping tool names to their handler functions
        max_steps: Maximum number of steps before stopping (default 5)
        model: The Anthropic model to use
        verbose: Whether to print detailed output (default True)

    Returns:
        The submitted answer if submit_answer was called, otherwise None
    """
    client = AsyncAnthropic()
    messages: list[MessageParam] = [{"role": "user", "content": prompt}]

    for step in range(max_steps):
        if verbose:
            print(f"\n=== Step {step + 1}/{max_steps} ===")

        response = await client.messages.create(
            model=model, max_tokens=MAX_TOKENS, tools=tools, messages=messages
        )

        assert response.stop_reason in [
            "max_tokens",
            "tool_use",
            "end_turn",
        ], f"unsupported stop_reason {response.stop_reason}"
        if response.stop_reason == "max_tokens":
            print(
                f"Model reached max_tokens limit {MAX_TOKENS}. Increase "
                "MAX_TOKENS, simplify your task, or update the code to provide "
                "a message back to the model when it exceeds MAX_TOKENS."
            )

        # Track if we need to continue
        has_tool_use = False
        tool_results = []
        submitted_answer = None

        # Process the response
        for content in response.content:
            if content.type == "text":
                if verbose:
                    print(f"Assistant: {content.text}")
            elif content.type == "tool_use":
                has_tool_use = True
                tool_name = content.name

                if tool_name in tool_handlers:
                    if verbose:
                        print(f"Using tool: {tool_name}")

                    # Extract arguments based on tool
                    handler = tool_handlers[tool_name]
                    tool_input = content.input

                    # Call the appropriate tool handler
                    if tool_name == "python_expression":
                        assert (
                            isinstance(tool_input, dict) and "expression" in tool_input
                        )
                        if verbose:
                            print("\nInput:")
                            print("```")
                            for line in tool_input["expression"].split("\n"):
                                print(f"{line}")
                            print("```")
                        result = handler(tool_input["expression"])
                        if verbose:
                            print("\nOutput:")
                            print("```")
                            print(result)
                            print("```")
                    elif tool_name == "submit_answer":
                        assert isinstance(tool_input, dict) and "answer" in tool_input
                        result = handler(tool_input["answer"])
                        submitted_answer = result["answer"]
                    else:
                        # Generic handler call
                        result = (
                            handler(**tool_input)
                            if isinstance(tool_input, dict)
                            else handler(tool_input)
                        )

                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": content.id,
                            "content": json.dumps(result),
                        }
                    )

        # If we have tool uses, add them to the conversation
        if has_tool_use:
            messages.append({"role": "assistant", "content": response.content})

            messages.append({"role": "user", "content": tool_results})

            # If an answer was submitted, return it
            if submitted_answer is not None:
                if verbose:
                    print(f"\nAgent submitted answer: {submitted_answer}")
                return submitted_answer
        else:
            # No tool use, conversation might be complete
            if verbose:
                print("\nNo tool use in response, ending loop.")
            break

    if verbose:
        print(f"\nReached maximum steps ({max_steps}) without submitting answer.")
    return None


async def run_single_test(
    run_id: int,
    num_runs: int,
    problem: Problem,
    verbose: bool = False,
) -> tuple[int, bool, Any]:
    # Create output directory for this run
    output_dir = Path("output") / f"run_{run_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set context for this run
    current_run_context.set({"output_dir": str(output_dir), "run_id": run_id})

    if verbose:
        print(f"\n\n{'=' * 20} RUN {run_id}/{num_runs} {'=' * 20}")
        print(f"Output directory: {output_dir}")

    result = await run_agent_loop(
        prompt=problem.prompt,
        tools=problem.tools,
        tool_handlers=problem.tool_handlers,
        max_steps=15,
        verbose=verbose,
    )

    # Wrap result in artifacts dictionary for grading
    artifacts = {"result": result}
    success = problem.grade(artifacts)

    if success:
        if isinstance(result, int | float) and result > 0.5:
            # Likely a speedup ratio
            print(
                f"✓ Run {run_id}: SUCCESS - Achieved {result:.2f}x speedup (target: {problem.expected_answer})"
            )
        else:
            print(f"✓ Run {run_id}: SUCCESS - Got {result}")
    else:
        if result is None:
            print(
                f"✗ Run {run_id}: FAILURE - No answer submitted (expected: {problem.expected_answer})"
            )
        elif isinstance(result, int | float):
            print(
                f"✗ Run {run_id}: FAILURE - Got {result:.2f}x, expected {problem.expected_answer}"
            )
        else:
            # For non-numeric results (like filepaths), just show failure
            # Detailed error message already printed by grade() method
            print(
                f"✗ Run {run_id}: FAILURE - See error details above (expected: {problem.expected_answer})"
            )

    return run_id, success, result


async def main(
    problem: Problem,
    num_runs: int = 1,
    concurrent: bool = True,
    verbose: bool = False,
):
    """
    Runs the agent test suite.

    Args:
        problem: Problem instance containing the task definition
        num_runs: Number of test iterations to run (default 10)
        concurrent: Whether to run tests concurrently or sequentially (default True)
        verbose: Whether to print detailed agent reasoning and tool usage (default False)
    """
    execution_mode = "concurrently" if concurrent else "sequentially"
    print(f"Running {num_runs} test iterations {execution_mode}...")
    if verbose:
        print("Verbose mode: ON (showing agent reasoning and tool usage)")
    print("=" * 60)

    # Create all test coroutines
    tasks = [
        run_single_test(
            run_id=i + 1,
            num_runs=num_runs,
            problem=problem,
            verbose=verbose,
        )
        for i in range(num_runs)
    ]

    # Run concurrently or sequentially based on the flag
    if concurrent:
        # Process results as they complete
        results = []
        for coro in asyncio.as_completed(tasks):
            result = await coro
            results.append(result)
    else:
        # Run sequentially by awaiting each task in order
        results = []
        for task in tasks:
            result = await task
            results.append(result)

    # Count successes
    successes = sum(1 for _, success, _ in results if success)

    # Calculate and display pass rate
    pass_rate = (successes / num_runs) * 100
    print(f"\n{'=' * 60}")
    print("Test Results:")
    print(f"  Passed: {successes}/{num_runs}")
    print(f"  Failed: {num_runs - successes}/{num_runs}")
    print(f"  Pass Rate: {pass_rate:.1f}%")
    print(f"{'=' * 60}")
