hello-py: ML training optimization task (with heavy constraints on runtime performance of final solution, number of steps needed to hit solution, and whether to prompt on intermediate outputs)
===

Setup instructions:

1. Clone the repository:
   ```
   git clone https://github.com/preferencemodel/hello-py.git
   ```

2. Navigate to the project directory:
   ```
   cd hello-py
   ```

3. Set up `ANTHROPIC_API_KEY` environment variable:
   ```
   export ANTHROPIC_API_KEY=your_api_key_here
   ```

4. Run the agent:
   ```
   uv run main.py
   ```

## Execution Modes

The test suite supports both concurrent and sequential execution.

To change modes, edit the `concurrent` parameter at the bottom of `main.py`:

```python
asyncio.run(main(concurrent=True))
asyncio.run(main(concurrent=False))
```

When running concurrently, results print as they complete (not in run order) for faster overall execution.

## Configuration Options

### Toggle `submit_intermediate_answer` Tool

You can control whether the model has access to the `submit_intermediate_answer` tool in `main.py`, which allows iterative testing without final submission. This significantly improves consistency and convergence speed.

**In code** (when instantiating the problem class):
```python
# Enable intermediate answer tool (default - improves consistency)
problem = ComplexMLTrainingWithProfiler(
    include_profiler=True,
    include_intermediate_answer=True
)

# Disable intermediate answer tool (harder task)
problem = ComplexMLTrainingWithProfiler(
    include_profiler=True,
    include_intermediate_answer=False
)
```

**Effect on behavior:**
- `include_intermediate_answer=True`: Model can test optimizations iteratively, leading to faster convergence and higher success rates (~40% pass rate with 33x speedup target)
- `include_intermediate_answer=False`: Model must submit final answer without testing, making the task significantly harder and increasing iteration count

The prompt automatically adjusts to reflect whether this tool is available.

# Connor Soohoo's Solution

## Task: **Optimize Slow ML Training Loop with Profiling; specifically on tackling the deeper issues once low hanging fruit is completed and within the alotted number of steps**

**Task Description**: Given a slow training loop implementation, use profiling to identify bottlenecks and optimize performance for a high speed performance target while maintaining equivalent model performance and a low step count.

More specifically, take `problem_data/slow_ml_training_complex.py` and turn it into a more optimized version, roughly called `ml_training_optimization.py` that is 36 times faster than the original solution while still falling roughly within the same accuracy bounds.

### Why it's interesting
- Critical skill for efficient research, specifically for showing work espeically when we **toggle disable a tool for submitting intermediate outputs.**
- Multiple optimization strategies, including theoretical (vectorization, caching, algorithmic improvements) and practical (looking at profiles, saving individual files and incrementally iterating, cythonization) techniques.
- Small number of allowed steps (15) where the model has to submit a final answer.
- **A very difficult target (36x speedup factor) that goes beyond what the model knows for traditional performance optimization techniques, which makes it difficult for the model to reach the optimal speed within the set number of steps**

### Some potential issues with this task setup
- Sample size of 10 for a 40% pass rate is kind of arbitrary, so there is variance at play here.
- Given that we are telling the model to output af file, a lot of times the model doesn't follow the prompt instructions exactly and outputs these optimized ML training files to a bunch of random locations. So we need to add fallback checks to retrieve the optimized training files.
- **Different hardwares will have different baseline performances of this slow_ml_training_complex.py script, so to standardize hardware, we should make sure that we put this task into a docker container and give it fixed CPU and GPU resources as well as standardized CPU and GPU specs. This is future work.**


### Grading approach
- Performance improvement threshold (e.g., >36x speedup)
- Model performance maintained (results match within tolerance)
- Does NOT check if profiler was actually used, mainly lets the model decide how to go about optimizing.
- Safety checks to make sure model is not cheating.

---

## Results

### Run 1: 36x target, allowing submitting intermediate outputs, 70% pass
Full run results can be found in `saved_output_36x_target_intermediate_tool_70pass`, particulaly in the `saved_output_36x_target_intermediate_tool_70pass/run_summary.md` for the high level summary.

Note that when we give the model the ability to submit intermediate outputs, it is able to achieve the optimal solution (36x speedup, keeping accuracy) within 15 steps.
```
============================================================
Test Results:
  Passed: 7/10
  Failed: 3/10
  Pass Rate: 70.0%
  Average Steps: 11.6
============================================================
```

### Run 2: 36x target, not allowing submitting intermediate outputs, 20% pass
Full run results can be found in `saved_output_36x_target_no_intermediate_outputs_20pass`, particulaly in the `saved_output_36x_target_no_intermediate_outputs_20pass/run_summary.md` for the high level summary.

However, when we remove the ability for the model to submit intermediate outputs, the models are not able to optimize well enough hit a 36x speedup within 15 steps of iteration, indicative of the model "spinning its wheels" and not being able to make substantive progress.
```
============================================================
Test Results:
  Passed: 2/10
  Failed: 8/10
  Pass Rate: 20.0%
  Average Steps: 11.7
============================================================
```

## Architecture

### Relevant Files

- `main.py`: Main place to modify which problems get invoked, how many iterations to run, whether to run with allowing submitting an intermediate output tool or not.
- `tools.py`: Adds a `read_file` and a `profile_tool` that way the model can read the slow python file and also profile these specific training files using cProfile. Also adds a `submit_intermediate_answer` tool as well that way the agent can iteratively test their solution against the baseline, which can be toggled on and off to make the task harder.
- `problem.py`: Problem class skeleton, used in problems/ folder.
- `problems/ml_training_optimization.py`: Meat and potatoes of the code, including the prompt of what the task is, the custom grading routine, safety checks, and where outputs go.
- `problem_data/slow_ml_training_complex.py`: This is the slow file and target of the ML training optimization task and is what the model should optimize (to somewhere between 33x and 36x faster than normal).
- `output/` : This is where when you run `uv run main.py`, where the model artifacts go
- `saved_output_xx/` : This folder is where I copied a run of output/ to show between 10 and 40% pass rate on a sample size of 10

### Key Nuances of architecture

1. **Custom grader** (`ml_training_optimization.py`): hidden away from optimization target file, makes sure that the model cannot cheat to the task and just skip computation using various safety checks. Tests original training file performance against new optimized file performance while not sacrificing on model accuracy. **I know that the alternative approach is to bypass submitting a file, which could potentially reduce false failures, but I am very opinionated that there should be real files being created and submitted that way the model is able to tackle more real world types of problems**

2. **Optionality of using profiler**: Multiple paths to get to the right answer, which may or may not involve using the profiler tool.

3. **Toughness of problem**: With a speedup factor between 33 and 36, you can force the model to actually optimize for the last 5% of code improvements. Generally these foundation models can get the low hanging fruit of code optimizations, but to really eek out those last performance gains (which is important since ML training is so expensive), this requires teaching the model how to really choose the right approaches to quickly performance engineer their code. This also makes it much harder for the model to converge within 15 steps.

4. **Set time to convergence:** With only 15 steps to convergence, and explicitly no prompting to let the agent know they only have a limited number of steps to converse with the model, agents who are able to achieve the final optimization with a smaller number of steps (i.e. without spinning their wheels) are rewarded. Especially with not allowing an intermediate output submission step, this will trigger models to think more deeply about the problem before spinning their wheels.

### Things that the model could learn by completing this task.

1. **Identify bottlenecks quickly without spinning wheels.**: Too often, foundation models get side tracked and get into an infinite loop or really slow iteration cycle. Sure these problems can be tackled via 4 rounds of successive profiling, or by creating several iterations of the same optimized file, but they take much longer to reach optimal solutions. A better way to tackle it is to plan ahead first at the code, or create its own eval cycle that way it can create its own intermediary outputs.
2. **Getting better at incrementally testing and evaluating code**: Note that how the model performance is dramatically better when adding a prompt around getting intermediate results and adding a new `submit_intermediate_result` tool. Note that in ML training, it is very expensive to be training and submitting final results, so it makes sense that a ML researcher is able to test and validate their code in smaller chunks without having access to being able to test on the test data set.
3. **Getting better at handling ambiguity and deep research in general**: Most models take the shortest path and incrementally inch their way to a closer answer in this ML training optimization task. With a small number of test steps and lack of intermediate prompting, we reward models who are better at being able to proactively plan as well as create their own test and evaluation strategies.

### Observed failure modes:
1. Tries to completely rewrite `ComplexFeatureExtractor` or `ComplexNN` into something non-sensical and tries to cheat the system, instead of using proper profiling tools or optimizations. Only happens in rare cases, less than 1 in 10 times.

2. Tries to cythonize the python file i.e. generate a `.cpython` extension, instead of optimizing the core functionality. This also technically accomplishes the task of performance optimization, but we aren't able to properly eval it or invoke it. A TODO is to fix the grader to handle this edge case and evalute on cythonized python files.

3. Doesn't adhere to the prompt requirements (don't rewrite the function names because of validation, or instead of outputting artifacts under `output/run_X`, it just outputs to the root directory, or forgets to `submit_answer` in the tool, or outputs the file in the top level `output` directory).

**To mitigate instances of 3, we have a few fallback checks in the event that the model forgets to invoke the `submit_answer` tool, specifically to check the current run's output directory to see if an optimized file does exist. This keeps all failures task related (optimizing a piece of ML training code instead of failing to follow the submit_answer instructions), but this does potentially indicate that these models do not follow instructions to a T as well as they should.**

4. Agent spins its wheels and doesn't take within 15 steps to get to the proper answer with the improved speedup factor. Note that this also sometimes means that it just doesn't submit an answer and outputs nothing, which also indicates that the model is spinning its wheels trying to converge to the right answer within 15 steps.

5. Optimized python file doesn't actually run to completion. For example, simple runtime errors due to using different versions of pytorch. Only happens in rare cases, less than 1 in 10 times. But again, this indicates that a model has not yet converged to an optimal solution.

6. [IMPORTANT] Doesn't get to full optimization. Specifically in Line 81-101 of `slow_ml_training_complex.py`: Replace O(n²) normalization here:
```python
  def custom_normalize(self, features):
      """Efficient z-score normalization - O(n)."""
      mean = np.mean(features, axis=0, keepdims=True)
      std = np.std(features, axis=0, keepdims=True) + 1e-8
      return (features - mean) / std
```

There are a few different approaches that could be implemented here, and agents did not fully converge to a solution here. There were significant differences in optimization strategies of this `custom_normalize()` function.

Namely, there were 4 strategies that could've been implemented, and only 1 strategy (Strategy 2) actually gives the best performance to beat the target.

# Strategy 1: Vectorized Pairwise Distances

  ## Still O(n²) but vectorized with NumPy broadcasting
  ```python
  sq_norms = np.sum(features ** 2, axis=1, keepdims=True)
  distances_sq = sq_norms - 2 * np.dot(features, features.T) + sq_norms.T
  distances = np.sqrt(distances_sq)
  scales = np.median(distances, axis=1, keepdims=True) + 1e-8
  normalized = features / scales
  ```
  - Preserves original algorithm logic (pairwise distances + median)
  - Performance: ~33-34x speedup

# Strategy 2: Simple z-score Normalization

  ## O(n) - completely different algorithm
  ```python
  means = np.mean(features, axis=0, keepdims=True)
  stds = np.std(features, axis=0, keepdims=True) + 1e-8
  return (features - means) / stds
  ```
  - Changed the algorithm entirely to standard normalization
  - Performance: 38.81x speedup (best performance!)
  - Risk: Different normalization approach might affect accuracy

# Strategy 3: Feature Magnitude Scaling


## O(n) - simplified approximation
```python
  feature_magnitude = np.linalg.norm(features, axis=1, keepdims=True)
  scale = np.median(feature_magnitude) + 1e-8
  normalized = features / scale
```
  - Approximation of the original intent
  - Performance: 32.49x (missing target)

#  Strategy 4: sklearn StandardScaler

```python
  from sklearn.preprocessing import StandardScaler
  self.scaler = StandardScaler()
  normalized = self.scaler.fit_transform(features)
```
  - Replaced custom method with sklearn
  - Performance: 30.37x speedup (missing target)

  Gradient Extraction Optimization - Universal Success

  All runs successfully vectorized this, converting from triple nested loops to NumPy operations:
  - Original: Nested loops over samples × rows × cols
  - Optimized: All runs used np.diff() or array slicing (e.g., images[:, :, 1:] - images[:, :, :-1])
  - This was a 100% convergence on the optimization approach

  Custom Activation Function - Universal Success

  All runs vectorized the activation from element-wise loops to torch operations:
  - Original: Double loop with .item() calls (extremely slow)
  - Optimized: All used torch.where() or masking with vectorized ops
  - Another 100% convergence
