# Generate the typing for a given function (specifically as it would be in
# CUDA) without doing any lowering.

# The majority of the file implements a typing-only pipeline.

# The end of the file contains an example of its use.

from numba.core.sigutils import normalize_signature
from numba.core.compiler import (sanitize_compile_result_entries, CompilerBase,
                                 CompileResult, DefaultPassBuilder,
                                 compile_extra)
from numba.core.compiler_lock import global_compiler_lock
from numba.core.compiler_machinery import (LoweringPass,
                                           PassManager, register_pass)
from numba.core import typing
from numba.core.typed_passes import IRLegalization, AnnotateTypes
from numba.cuda.compiler import CUDAFlags
from numba.cuda.descriptor import cuda_target

import numpy as np


class TypingOnlyCompileResult(CompileResult):
    pass


def typing_only_compile_result(**entries):
    entries = sanitize_compile_result_entries(entries)
    return TypingOnlyCompileResult(**entries)


@register_pass(mutates_CFG=True, analysis_only=False)
class TypingOnlyBackend(LoweringPass):

    _name = "typing_only_backend"

    def __init__(self):
        LoweringPass.__init__(self)

    def run_pass(self, state):
        """
        Back-end: Packages lowering output in a compile result
        """
        signature = typing.signature(state.return_type, *state.args)

        state.cr = typing_only_compile_result(
            typing_context=state.typingctx,
            target_context=state.targetctx,
            typing_error=state.status.fail_reason,
            type_annotation=state.type_annotation,
            library=state.library,
            call_helper=None,
            signature=signature,
            fndesc=None,
        )
        return True


class TypingOnlyCompiler(CompilerBase):
    def define_pipelines(self):
        dpb = DefaultPassBuilder
        pm = PassManager('typing_only')

        untyped_passes = dpb.define_untyped_pipeline(self.state)
        pm.passes.extend(untyped_passes.passes)

        typed_passes = dpb.define_typed_pipeline(self.state)
        pm.passes.extend(typed_passes.passes)

        lowering_passes = self.define_typing_lowering_pipeline(self.state)
        pm.passes.extend(lowering_passes.passes)

        pm.finalize()
        return [pm]

    def define_typing_lowering_pipeline(self, state):
        pm = PassManager('typing_only_lowering')
        # legalise
        pm.add_pass(IRLegalization,
                    "ensure IR is legal prior to lowering")
        pm.add_pass(AnnotateTypes, "annotate types")
        pm.add_pass(TypingOnlyBackend, "typing only backend")
        pm.finalize()
        return pm


@global_compiler_lock
def type_cuda(pyfunc, return_type, args, debug=False, lineinfo=False,
              inline=False, fastmath=False, nvvm_options=None,
              cc=None, max_registers=None, lto=False):

    typingctx = cuda_target.typing_context
    targetctx = cuda_target.target_context

    flags = CUDAFlags()
    # Do not compile (generate native code), just lower (to LLVM)
    flags.no_compile = True
    flags.no_cpython_wrapper = True
    flags.no_cfunc_wrapper = True

    # Both debug and lineinfo turn on debug information in the compiled code,
    # but we keep them separate arguments in case we later want to overload
    # some other behavior on the debug flag. In particular, -opt=3 is not
    # supported with debug enabled, and enabling only lineinfo should not
    # affect the error model.
    if debug or lineinfo:
        flags.debuginfo = True

    if lineinfo:
        flags.dbg_directives_only = True

    if debug:
        flags.error_model = 'python'
    else:
        flags.error_model = 'numpy'

    if inline:
        flags.forceinline = True
    if fastmath:
        flags.fastmath = True
    if nvvm_options:
        flags.nvvm_options = nvvm_options
    flags.max_registers = max_registers
    flags.lto = lto

    # Run compilation pipeline
    from numba.core.target_extension import target_override
    with target_override('cuda'):
        cres = compile_extra(typingctx=typingctx,
                             targetctx=targetctx,
                             func=pyfunc,
                             args=args,
                             return_type=return_type,
                             flags=flags,
                             locals={},
                             pipeline_class=TypingOnlyCompiler)

    return cres


def type_infer(func, args):
    args, retty = normalize_signature(args)
    cres = type_cuda(func, retty, args)
    return cres.type_annotation, cres.signature


# The following is code that a user might be expected to write.

# An example function definition
def f(x, y):
    return x + y


# Example arguments to compile for
args = "(int32, int32)"

annotation, signature = type_infer(f, args)

print("Example annotation:\n")
print(annotation.annotate())
print(f"\nArgs f{args} give return type: {signature.return_type}")

# Another set of arguments
args = "(float32, float32)"
annotation, signature = type_infer(f, args)
print(f"Args f{args} give return type: {signature.return_type}")

# And another set of arguments
args = "(float32, int32)"
annotation, signature = type_infer(f, args)
print(f"Args f{args} give return type: {signature.return_type}")


# A more complicated function - tests pipeline doing function calls

# Limitation: f_jit goes through the normal CUDA pipeline and compiles, so we
# need a trick similar to the one done by the CUDA target to work with @jit
# functions

from numba import cuda
f_jit = cuda.jit(f)

def g(x, y):
    return f_jit(x, np.cos(y))


annotation, signature = type_infer(g, args)
print()
print(f"For g(), args f{args} give return type: {signature.return_type}")
print(f"\nAnnotation of g():\n\n{annotation.annotate()}")
