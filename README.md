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

## Example output

Note that the behaviour is correct for `f()`, but not for `g()`. The types are
(probably) correct, but unintended lowering will still occur.

```
$ python type_only.py
Example annotation:

# File: /home/gmarkall/numbadev/issues/numba-type-only/type_only.py
# --- LINE 150 ---
# label 0
#   x = arg(0, name=x)  :: int32
#   y = arg(1, name=y)  :: int32

def f(x, y):

    # --- LINE 151 ---
    #   $binop_add8.2 = x + y  :: int64
    #   del y
    #   del x
    #   $12return_value.3 = cast(value=$binop_add8.2)  :: int64
    #   del $binop_add8.2
    #   return $12return_value.3

    return x + y



Args f(int32, int32) give return type: int64
Args f(float32, float32) give return type: float32
Args f(float32, int32) give return type: float64

For g(), args f(float32, int32) give return type: float64

Annotation of g():

# File: /home/gmarkall/numbadev/issues/numba-type-only/type_only.py
# --- LINE 183 ---
# label 0
#   x = arg(0, name=x)  :: float32
#   y = arg(1, name=y)  :: int32

def g(x, y):

    # --- LINE 184 ---
    #   $4load_global.0 = global(f_jit: CUDADispatcher(<function f at 0x79f62c70e840>))  :: type(CUDADispatcher(<function f at 0x79f62c70e840>))
    #   $18load_global.3 = global(np: <module 'numpy' from '/home/gmarkall/miniforge3/envs/numbadev/lib/python3.11/site-packages/numpy/__init__.py'>)  :: Module(<module 'numpy' from '/home/gmarkall/miniforge3/envs/numbadev/lib/python3.11/site-packages/numpy/__init__.py'>)
    #   $30load_attr.5 = getattr(value=$18load_global.3, attr=cos)  :: Function(<ufunc 'cos'>)
    #   del $18load_global.3
    #   $46call.7 = call $30load_attr.5(y, func=$30load_attr.5, args=[Var(y, type_only.py:183)], kws=(), vararg=None, varkwarg=None, target=None)  :: (int32,) -> float64
    #   del y
    #   del $30load_attr.5
    #   $60call.8 = call $4load_global.0(x, $46call.7, func=$4load_global.0, args=[Var(x, type_only.py:183), Var($46call.7, type_only.py:184)], kws=(), vararg=None, varkwarg=None, target=None)  :: (float32, float64) -> float64
    #   del x
    #   del $4load_global.0
    #   del $46call.7
    #   $70return_value.9 = cast(value=$60call.8)  :: float64
    #   del $60call.8
    #   return $70return_value.9

    return f_jit(x, np.cos(y))
```
