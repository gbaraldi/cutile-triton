import json
import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource, compile as tt_compile

@triton.jit
def k_empty(ptr, M, N, K):
    pid = tl.program_id(0)
    if pid == 1073741824:  # never
        tl.store(ptr, 1.0)

@triton.jit
def k_desc3(ptr, M, N, K):
    a = tl.make_tensor_descriptor(ptr, [M, K], [K, 1], [128, 64])
    b = tl.make_tensor_descriptor(ptr, [N, K], [K, 1], [256, 64])
    c = tl.make_tensor_descriptor(ptr, [M, N], [N, 1], [128, 256])
    pid = tl.program_id(0)
    if pid == 1073741824:
        x = a.load([0, 0])
        y = b.load([0, 0])
        c.store([0, 0], tl.dot(x, y.T))

for fn, name in ((k_empty, "k_empty"), (k_desc3, "k_desc3")):
    src = ASTSource(fn=fn, signature={"ptr": "*fp16", "M": "i32", "N": "i32", "K": "i32"}, constexprs={})
    k = tt_compile(src, target=GPUTarget("cuda", 90, 32), options={"num_warps": 8, "num_stages": 3})
    open(f"sass/{name}.cubin", "wb").write(k.asm["cubin"])
    md = {f: getattr(k.metadata, f) for f in ("name", "num_warps", "shared", "global_scratch_size") if hasattr(k.metadata, f)}
    json.dump(md, open(f"sass/{name}.json", "w"))
    print(name, md)
