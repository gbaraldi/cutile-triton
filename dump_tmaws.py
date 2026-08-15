"""Ground truth: what TTIR does Python's tl.make_tensor_descriptor produce,
and what launch metadata does the descriptor path need?"""
import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource, compile as tt_compile


@triton.jit
def matmul_tma(A, B, C, M, N, K,
               BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    # row-major convention: A is (M, K) with strides (K, 1)
    a_desc = tl.make_tensor_descriptor(A, shape=[M, K], strides=[K, 1], block_shape=[BM, BK])
    b_desc = tl.make_tensor_descriptor(B, shape=[K, N], strides=[N, 1], block_shape=[BK, BN])
    c_desc = tl.make_tensor_descriptor(C, shape=[M, N], strides=[N, 1], block_shape=[BM, BN])
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in tl.range(0, tl.cdiv(K, BK), warp_specialize=True):
        a = a_desc.load([pid_m * BM, k * BK])
        b = b_desc.load([k * BK, pid_n * BN])
        acc = tl.dot(a, b, acc)
    c_desc.store([pid_m * BM, pid_n * BN], acc)


src = ASTSource(
    fn=matmul_tma,
    signature={"A": "*fp32", "B": "*fp32", "C": "*fp32",
               "M": "i32", "N": "i32", "K": "i32",
               "BM": "constexpr", "BN": "constexpr", "BK": "constexpr"},
    constexprs={"BM": 128, "BN": 128, "BK": 64},
)
k = tt_compile(src, target=GPUTarget("cuda", 90, 32), options={"num_warps": 8, "num_stages": 3})
print(k.asm["ttir"])
print("== metadata ==")
for f in ("name", "num_warps", "shared", "global_scratch_size", "global_scratch_align",
          "tensordesc_meta", "cluster_dims"):
    if hasattr(k.metadata, f):
        print(f, "=", getattr(k.metadata, f))
open("matmul_tmaws_py.cubin", "wb").write(k.asm["cubin"])
import json
md = {f: getattr(k.metadata, f) for f in ("name", "num_warps", "warp_size", "shared",
                                          "global_scratch_size", "global_scratch_align")
      if hasattr(k.metadata, f)}
json.dump(md, open("matmul_tmaws_py.json", "w"))
