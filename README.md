hello-py
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

# Connor Soohoo's Solution

## Task: **Optimize Slow ML Training Loop with Profiling; specifically on tackling the deeper issues once low hanging fruit is completed**

**Task Description**: Given a slow training loop implementation, use profiling to identify bottlenecks and optimize performance while maintaining equivalent model performance.

### Why it's interesting
- Critical skill for efficient research
- A deeper understanding of analyzing profiling tool results is *one* potential path towards solving the task.
- Multiple optimization strategies, including theoretical (vectorization, caching, algorithmic improvements) and practical (looking at profiles, saving individual files and incrementally iterating, cythonization) techniques.
- **A very difficult speedup factor i.e. target that goes beyond what the model knows for traditional performance optimization techniques**

### Expected failure modes
- Sample size of 10 for a 40% pass rate is kind of arbitrary, so there is variance at play here.
- **Different hardwares will have different baseline performances of this slow_ml_training_complex.py script, so to standardize hardware, we should make sure that we put this task into a docker container and give it fixed CPU and GPU resources as well as standardized CPU and GPU specs. This is future work.**
- Premature optimization without measurement
- Over-optimizing at cost of readability

### Grading approach
- Performance improvement threshold (e.g., >2x speedup)
- Model performance maintained (results match within tolerance)
- Does NOT check if profiler was actually used, mainly lets the model decide how to go about optimizing.

---

## Architecture

### Relevant Files

- `main.py`: Main place to modify which problems get invoked, how many iterations to run.
- `tools.py`: Adds a `read_file` and a `profile_tool` that way the model can read the slow python file and also profile these specific training files using cProfile. Also adds a `submit_intermediate_answer` tool as well that way the agent can iteratively test their solution against the baseline.
- `problem.py`: Problem class skeleton, used in problems/ folder.
- `ml_training_optimization.py`: Meat and potatoes of the code, including the prompt of what the task is, the custom grading routine, safety checks, and where outputs go.
- `problem_data/slow_ml_training_complex.py`: This is the slow file and target of the ML training optimization task and is what the model should optimize (to somewhere between 33x and 36x faster than normal).
- `output/` : This is where when you run `uv run main.py`, where the model artifacts go
- `saved_output_xx/` : This folder is where I copied a run of output/ to show between 10 and 40% pass rate on a sample size of 10

### Key Nuances of architecture

1. **Custom grader** (`ml_training_optimization.py`): hidden away from optimization target file, makes sure that the model cannot cheat to the task and just skip computation using various safety checks. Tests original training file performance against new optimized file performance while not sacrificing on model accuracy. **I know that the alternative approach is to bypass submitting a file, which could potentially reduce false failures, but I am very opinionated that there should be real files being created and submitted that way the model is able to tackle more real world types of problems**

2. **Optionality of using profiler**: Multiple paths to get to the right answer, which may or may not involve using the profiler tool.

3. **Toughness of problem: With a speedup factor between 33 and 36, you can force the model to actually optimize for the last 5% of code improvements. Generally these foundation models can get the low hanging fruit of code optimizations, but to really eek out those last performance gains (which is important since ML training is so expensive), this requires teaching the model how to really choose the right approaches to quickly performance engineer their code.**

4. **Set time to convergence:** With only 15 steps to convergence, and explicitly no prompting to let the agent know they only have a limited number of steps to converse with the model, agents who are able to achieve the final optimization with a smaller number of steps (i.e. without spinning their wheels) are rewarded.

### Things that the model could learn by completing this task.

1. Identify bottlenecks quickly without spinning wheels.
2. Once low hanging fruit optimization is done, how to continue to identify bottlenecks
3. Isolating different functions / sub-portions and experimenting with different routines to improve the problem, not just looking at it from a theory, code perspective.
4. Not getting side tracked and staying focused on given a large-ish code volume, what are the highest bottlenecks. Can be tackled via profiling, or by creating several iterations of the same optimized file, but not getting stuck in infinite loops. This helps **improve the speed of the model to quickly arrive at solutions instead of spinning wheels**.

### Observed failure modes:
1. Tries to completely rewrite `ComplexFeatureExtractor` or `ComplexNN` into something non-sensical and tries to cheat the system, instead of using proper profiling tools or optimizations. Only happens in rare cases, less than 1 in 10 times.

2. Tries to cythonize the python file i.e. generate a `.cpython` extension, instead of optimizing the core functionality. This also technically accomplishes the task of performance optimization, but we aren't able to properly eval it or invoke it. A TODO is to fix the grader to handle this edge case and evalute on cythonized python files.

3. Doesn't adhere to the prompt requirements (don't rewrite the function names because of validation, or instead of outputting artifacts under `output/run_X`, it just outputs to the root directory, or forgets to `submit_answer` in the tool, or outputs the file in the top level `output` directory).

**To mitigate instances of 3, we have a few fallback checks in the event that the model forgets to invoke the `submit_answer` tool, specifically to check the current run's output directory to see if an optimized file does exist. This keeps all failures task related (optimizing a piece of ML training code instead of failing to follow the submit_answer instructions), but this does potentially indicate that these models do not follow instructions to a T as well as they should.**

4. Agent spins its wheels and doesn't take within 15 steps to get to the proper answer with the improved speedup factor.

5. Optimized python file doesn't actually run to completion. For example, simple runtime errors due to using different versions of pytorch. Only happens in rare cases, less than 1 in 10 times.

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
