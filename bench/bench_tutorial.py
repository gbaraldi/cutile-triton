"""Canonical tutorial-09 TMA matmul (device-descriptor variant), no torch."""
import json, sys
import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource, compile as tt_compile

@triton.jit
def matmul_tut(a_ptr, b_ptr, c_ptr, M, N, K,
               BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
               GROUP: tl.constexpr, WS: tl.constexpr):
    a_desc = tl.make_tensor_descriptor(a_ptr, [M, K], [K, 1], [BM, BK])
    b_desc = tl.make_tensor_descriptor(b_ptr, [N, K], [K, 1], [BN, BK])
    c_desc = tl.make_tensor_descriptor(c_ptr, [M, N], [N, 1], [BM, BN])
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BM)
    num_pid_n = tl.cdiv(N, BN)
    num_pid_in_group = GROUP * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP
    group_size_m = min(num_pid_m - first_pid_m, GROUP)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    k_tiles = tl.cdiv(K, BK)
    offs_am = pid_m * BM
    offs_bn = pid_n * BN
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in tl.range(k_tiles, warp_specialize=WS):
        offs_k = k * BK
        a = a_desc.load([offs_am, offs_k])
        b = b_desc.load([offs_bn, offs_k])
        acc = tl.dot(a, b.T, acc)
    c_desc.store([offs_am, offs_bn], acc.to(tl.float16))

for name, bm, bn, bk, s, w, ws in [
    ("tut_128_256_64_s3_w8", 128, 256, 64, 3, 8, False),
    ("tut_128_256_64_s3_w8_ws", 128, 256, 64, 3, 8, True),
    ("tut_128_128_64_s4_w8", 128, 128, 64, 4, 8, False),
]:
    src = ASTSource(fn=matmul_tut,
                    signature={"a_ptr": "*fp16", "b_ptr": "*fp16", "c_ptr": "*fp16",
                               "M": "i32", "N": "i32", "K": "i32",
                               "BM": "constexpr", "BN": "constexpr", "BK": "constexpr",
                               "GROUP": "constexpr", "WS": "constexpr"},
                    constexprs={"BM": bm, "BN": bn, "BK": bk, "GROUP": 8, "WS": ws})
    k = tt_compile(src, target=GPUTarget("cuda", 90, 32),
                   options={"num_warps": w, "num_stages": s})
    open(f"sass/{name}.cubin", "wb").write(k.asm["cubin"])
    md = {f: getattr(k.metadata, f) for f in ("name", "num_warps", "warp_size", "shared",
                                              "global_scratch_size") if hasattr(k.metadata, f)}
    json.dump(md, open(f"sass/{name}.json", "w"))
    print(name, md)
