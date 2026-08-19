# Flow-matrix SpMM: cuTile vs KernelAbstractions vs cuSPARSE

Results of `bench/spmm_flow.jl` (job 7972, 2026-08-20) on the min-cost-flow
constraint matrix used by CoolPDLP: the node-arc incidence of the TX road
network (`road_flow_07_TX_{a..e}`), **A = 2,073,870 × 5,116,492 with
10,232,984 nonzeros** (all ±1, ~5 per row of A, exactly ≤2 per row of Aᵀ).

- GPU: NVIDIA L40S (96 MB L2, ~864 GB/s DRAM), CUDA 13.1, Float32
- CUDA.jl 6.3.0, cuTile.jl 1.0.0, triton 3.7.1 through TileTriton
- C = α·A·B + β·C timed at α=1, β=0 (the αβ rows read C as well);
  every kernel's output verified against a CPU reference first
- GFLOP/s counts 2·nnz·n; best tile config per kernel shown,
  full sweep in the RESULT rows of the job log

## A·B — nodes×arcs, JDS formats (GFLOP/s)

| n | cuTile jds pm | cuTile jds | cuTile csr | KA jds pm | KA jds | KA csr | cuSPARSE |
|-----|------|------|------|---------|------|------|------|
| 8   | 143  | 135  | 155  | **215** | 103  | 97   | 117  |
| 64  | 114  | 101  | 130  | **245** | 101  | 69   | 101  |
| 256 | 90   | 89   | 99   | **158** | 87   | 83   | 101  |

## Aᵀ·B — arcs×nodes, 2-per-row formats (GFLOP/s)

| n | cuTile 2pr pm | cuTile 2pr | cuTile csr | KA 2pr pm | KA 2pr | KA csr | cuSPARSE |
|-----|---------|------|------|------|------|------|------|
| 8   | **309** | 277  | 119  | 216  | 111  | 89   | 165  |
| 64  | **345** | 337  | 34   | 249  | 118  | 94   | 144  |
| 256 | **194** | 187  | 44   | 159  | 94   | 77   | 125  |

("pm" = the value-free ±1 variants, "vals" columns carry explicit values;
KA = the KernelAbstractions kernels, jds/2pr being SpMM ports of
MinimumCostFlows' matrix_zoo kernels, csr being CoolPDLP's
row-per-thread spmm_csr!.)

## Findings

**The cuTile 2-per-row kernel is the fastest way to compute Aᵀ·B at every
n** — 1.4–2.4× over cuSPARSE and 1.2–1.4× over the KA 2-per-row kernel.
It is loop-free: one program covers TILE_M rows × TILE_N columns, gathers
each row's two column ids and the two corresponding B slabs, and relies on
cuTile's bounds-masked gathers to turn the "absent" id 0 into a zero
contribution. At the CoolPDLP-relevant n=256 it does the product in
27.0 ms vs 33.0 ms (KA) and 42.0 ms (cuSPARSE).

**On A·B the KA jds-pm kernel is still the leader** (215/245/158). The
cuTile JDS port is correct but trails (90–143), as does cuSPARSE; the gap
is unprofiled. Candidate causes: per-diagonal scalar iterptr loads and the
trip-count while-loop per program, and the tile_n ≤ 64 register cap
limiting shape choices at large n.

**The ±1 trick (pm) pays in the KA kernels, less so in cuTile.** KA 2pr:
249 vs 118 GFLOP/s at n=64; the cuTile kernels see only a few percent
(345 vs 337) because the B gathers dominate their traffic.

**Why throughput falls at large n instead of improving.** Wider B does not
amortize the dominant traffic: per nonzero, n elements of B are gathered
and n of C written, so bytes scale with flops and arithmetic intensity
plateaus at ~⅓ flop/byte — only the per-row metadata amortizes, and it is
negligible already at n=8. The n-dependence that remains is cache
residency of B (k×n Float32):

| n | B size | vs 96 MB L2 |
|-----|--------|-------------|
| 8   | 66 MB  | fully resident |
| 64  | 530 MB | sliding window only |
| 256 | 2.1 GB | (+5.2 GB of C) |

Each B row is reused ~deg(node) ≈ 5 times; that reuse is an L2 hit at
n=8 and a DRAM miss at n=256. Achieved DRAM bandwidth (compulsory-traffic
estimate) for cuTile 2pr: ~510 GB/s (n=8), ~485 GB/s (n=64), ~270 GB/s
(n=256) — flops per DRAM byte are nearly constant across n, so the
GFLOP/s drop is exactly the bandwidth drop. Column-major B makes it
worse: a gathered "row" is n elements strided 8.3 MB apart, i.e. n
distinct sectors and pages with 4 useful bytes each. And since tile_n is
capped at 64, n=256 is structurally four independent sweeps with no
cross-sweep reuse.

**Next steps worth trying**: store B (and C) transposed so a gathered row
is n contiguous floats — sector utilization and TLB behaviour then stop
degrading with n; profile the cuTile JDS kernel vs KA jds-pm with ncu;
raise the jds/csr tile caps for the n ≥ 64 cases.

## Reproducing

```
julia --project=. bench/spmm_flow.jl 8 64 256     # needs bench/data/*.jls
```

`bench/data/` (gitignored, ~270 MB) is regenerated from the DIMACS files in
`~/min-cost-flow/road/` with MinimumCostFlows: read the problem, then
serialize `SparseMatrixCSC{Float32}` of A and Aᵀ as `flow_TX{,_t}.jls` and
the raw `JDSMatrixPM`/`Matrix2PerRowPM` arrays as `flow_TX_zoo.jls`
(NamedTuple with fields `jds_colidx, jds_iterptr, jds_nrows, jds_ncols,
tpr_colidx, tpr_ncols`) — see the header of `bench/spmm_flow.jl`.
