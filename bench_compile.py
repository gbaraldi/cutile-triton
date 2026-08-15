"""Compile Python-frontend Triton reference kernels to cubin + metadata,
matching the cuTile kernels' tile configs, for launching from Julia."""
import json
import os

import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource, compile as tt_compile

OUT = os.path.join(os.path.dirname(__file__), "bench_py")
os.makedirs(OUT, exist_ok=True)
TARGET = GPUTarget("cuda", 90, 32)


@triton.jit
def vadd(x, y, out, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(out + offs, tl.load(x + offs, mask=mask) + tl.load(y + offs, mask=mask), mask=mask)


@triton.jit
def matmul(A, B, C, M, N, K, sam, sak, sbk, sbn, scm, scn,
           BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    A_ptr = A + rm[:, None] * sam + rk[None, :] * sak
    B_ptr = B + rk[:, None] * sbk + rn[None, :] * sbn
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BK)):
        a = tl.load(A_ptr, mask=(rm[:, None] < M) & ((rk[None, :] + k * BK) < K), other=0.0)
        b = tl.load(B_ptr, mask=((rk[:, None] + k * BK) < K) & (rn[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc)
        A_ptr += BK * sak
        B_ptr += BK * sbk
    c_mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(C + rm[:, None] * scm + rn[None, :] * scn, acc, mask=c_mask)


@triton.jit
def softmax(out, inp, n_rows, n_cols, stride_o, stride_i, TILE: tl.constexpr):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    rows = tl.arange(0, TILE)
    col = pid
    while col < n_cols:
        x = tl.load(inp + col * stride_i + rows, mask=rows < n_rows, other=float("-inf"))
        m = tl.max(x, axis=0)
        e = tl.exp(x - m)
        s = tl.sum(e, axis=0)
        tl.store(out + col * stride_o + rows, e / s, mask=rows < n_rows)
        col += nprog


def build(fn, name, signature, constexprs, num_warps, num_stages=3):
    src = ASTSource(fn=fn, signature=signature, constexprs=constexprs)
    k = tt_compile(src, target=TARGET,
                   options={"num_warps": num_warps, "num_stages": num_stages})
    with open(os.path.join(OUT, name + ".cubin"), "wb") as f:
        f.write(k.asm["cubin"])
    md = {f: getattr(k.metadata, f)
          for f in ("name", "num_warps", "warp_size", "shared") if hasattr(k.metadata, f)}
    with open(os.path.join(OUT, name + ".json"), "w") as f:
        json.dump(md, f)
    print(name, md)


build(vadd, "vadd_py",
      {"x": "*fp32", "y": "*fp32", "out": "*fp32", "n": "i32", "BLOCK": "constexpr"},
      {"BLOCK": 1024}, num_warps=4)

build(matmul, "matmul_py",
      {"A": "*fp32", "B": "*fp32", "C": "*fp32", "M": "i32", "N": "i32", "K": "i32",
       "sam": "i32", "sak": "i32", "sbk": "i32", "sbn": "i32", "scm": "i32", "scn": "i32",
       "BM": "constexpr", "BN": "constexpr", "BK": "constexpr"},
      {"BM": 128, "BN": 128, "BK": 32}, num_warps=8)

build(softmax, "softmax_py",
      {"out": "*fp32", "inp": "*fp32", "n_rows": "i32", "n_cols": "i32",
       "stride_o": "i32", "stride_i": "i32", "TILE": "constexpr"},
      {"TILE": 1024}, num_warps=4)

print("BENCH COMPILE OK")
