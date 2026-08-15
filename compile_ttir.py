#!/usr/bin/env python
"""compile_ttir.py <in.ttir> <out_prefix> <num_warps> <sm> [num_stages]
Compiles a TTIR file with the triton wheel and writes <out_prefix>.cubin
and <out_prefix>.json (launch metadata)."""
import json
import sys

from triton.backends.compiler import GPUTarget
from triton.compiler import compile as tt_compile

src, out, num_warps, sm = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
num_stages = int(sys.argv[5]) if len(sys.argv) > 5 else 3

dump_stages = len(sys.argv) > 6 and sys.argv[6] == "dump"

k = tt_compile(src, target=GPUTarget("cuda", sm, 32),
               options={"num_warps": num_warps, "num_stages": num_stages})
with open(out + ".cubin", "wb") as f:
    f.write(k.asm["cubin"])
if dump_stages:
    for stage in ("ttir", "ttgir", "llir", "ptx"):
        if stage in k.asm:
            with open(out + "." + stage, "w") as f:
                f.write(k.asm[stage])
md = {f: getattr(k.metadata, f)
      for f in ("name", "num_warps", "warp_size", "shared", "global_scratch_size",
                "global_scratch_align", "launch_cooperative_grid", "cluster_dims")
      if hasattr(k.metadata, f)}
with open(out + ".json", "w") as f:
    json.dump(md, f)
print(json.dumps(md))
