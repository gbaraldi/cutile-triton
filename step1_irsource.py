"""Step 1: validate the IRSource path on triton 3.7.1.
1) Compile a trivial vadd from Python source (ASTSource) -> dump its TTIR.
2) Re-compile from the .ttir FILE via IRSource -> cubin + metadata.
"""
import json, os, sys
import triton
import triton.language as tl
from triton.compiler import ASTSource, IRSource, compile as tt_compile
from triton.backends.compiler import GPUTarget

@triton.jit
def vadd(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)

target = GPUTarget("cuda", 90, 32)  # sm_90, no GPU needed to compile

src = ASTSource(
    fn=vadd,
    signature={"x_ptr": "*fp32", "y_ptr": "*fp32", "out_ptr": "*fp32", "n": "i32", "BLOCK": "constexpr"},
    constexprs={"BLOCK": 128},
)
k1 = tt_compile(src, target=target, options={"num_warps": 4})
print("== ASTSource compile OK; artifacts:", sorted(k1.asm.keys()))
ttir = k1.asm["ttir"]
open("vadd.ttir", "w").write(ttir)
print("== TTIR ==")
print(ttir)
meta = {f: getattr(k1.metadata, f) for f in ("name", "num_warps", "shared", "cluster_dims") if hasattr(k1.metadata, f)}
print("== metadata:", json.dumps(meta))

# Now the part that matters: compile from the .ttir file
k2 = tt_compile(IRSource("vadd.ttir", target=target, options=None) if "options" in IRSource.__init__.__code__.co_varnames else "vadd.ttir", target=target, options={"num_warps": 4})
print("== IRSource compile OK; artifacts:", sorted(k2.asm.keys()))
print("== cubin bytes:", len(k2.asm["cubin"]))
print("== metadata2: name=%s num_warps=%s shared=%s" % (k2.metadata.name, k2.metadata.num_warps, k2.metadata.shared))
print("PASS step 1")
