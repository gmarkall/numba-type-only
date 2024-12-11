# Numba CUDA typing-only target

For experiments with a method to get the typing of a Numba-compiled function,
without doing any lowering at all (only typing).

Key points:

- Focused on numba-cuda at present, so it will type as if being compiled for
  numba-cuda.
- Doesn't behave as expected with function calls, because they will get compiled
  through their own target (e.g. the CUDA target).
  - We can fix this with the same trick used by the CUDA target to compile calls
    to `@jit` functions.
