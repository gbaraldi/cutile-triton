# Flow-matrix SpMM: cuTile vs KernelAbstractions vs cuSPARSE

Results of `bench/spmm_flow.jl` (job 7981, 2026-08-21) on the min-cost-flow
constraint matrix used by CoolPDLP: the node-arc incidence of the TX road
network (`road_flow_07_TX_{a..e}`), **A = 2,073,870 × 5,116,492 with
10,232,984 nonzeros** (all ±1, ~5 per row of A, ≤2 per row of Aᵀ).

- GPU: NVIDIA L40S (96 MB L2, ~864 GB/s DRAM), CUDA 13.1, Float32
- CUDA.jl 6.3.0, cuTile.jl 1.0.0, triton 3.7.1 through TileTriton
- C = α·A·B + β·C, timed with device events (`CUDA.@elapsed`, best of 20);
  every kernel's output verified against a CPU/device reference first
- GFLOP/s = 2·nnz·n / time; cuTile rows show the best tile config
  `tile_m×tile_n(×tile_k)` of the sweep
- Tables generated from the job log by `bench/spmm_flow_tables.py`

Implementations:

| label | what |
|---|---|
| cuTile csr / jds / 2pr | TileTriton tile kernels (`spmm_csr_kernels.jl`, `spmm_zoo_kernels.jl`) |
| … pm | value-free ±1 variants (`JDSMatrixPM`, `Matrix2PerRowPM`) |
| … t | fully transposed layout: B passed as n×k, C as n×m (rhs columns contiguous) |
| KA jds / 2pr [nb=k] | KernelAbstractions SpMM ports of MinimumCostFlows' matrix_zoo kernels, one thread per row × `nb` rhs columns (SIMD.Vec accumulator); nb=1 is the original one-thread-per-element shape |
| KA csr | CoolPDLP's row-per-thread `spmm_csr!` |
| cuSPARSE | `CuSparseMatrixCSR` `mul!` |

## Headline

1. **cuTile 2-per-row is the fastest Aᵀ·B at every n**: 387 / 355 / 352
   GFLOP/s for n = 8 / 64 / 256 in the standard layout, **425 at n=64 in
   the transposed layout** — 2.0–2.4× cuSPARSE (170–179) and 1.4–1.7× the
   best KA kernel (KA 2pr pm, 253–256). At CoolPDLP's n=256 that is
   14.9 ms vs 20.5 ms (KA) and 29.2 ms (cuSPARSE).
2. **KA jds pm is still the fastest A·B** (250–252 GFLOP/s at every n,
   nb=1), 1.9× cuSPARSE (127–133). The cuTile JDS port reaches 228 at n=8
   only in the transposed layout (175 / 147 at n = 64 / 256); in the
   standard layout it trails at 115–158.
3. **KA JDS vs cuSPARSE, the nb question.** The pm variant beats cuSPARSE
   at every nb (252 / 189 / 137 for nb = 1 / 4 / 8 vs 127–133), and nb=1
   stays its best shape — blocking only costs it parallelism. The **vals
   variant flips from losing to winning with nb=4**: 110 GFLOP/s at nb=1
   (0.85× cuSPARSE) → **150 at nb=4 (1.15× cuSPARSE)**, 129 at nb=8. Same
   on Aᵀ: KA 2pr vals goes 119 → 192 / 198 (nb = 4 / 8) vs cuSPARSE 179.
   So blocking pays exactly where the per-element shape re-reads values
   and ids n times, and is neutral-to-harmful where it doesn't.
4. **Throughput is flat in n** for essentially every kernel (e.g. KA jds
   pm 252 / 250 / 251, cuSPARSE 127 / 132 / 133, cuTile 2pr pm 387 / 355 /
   352). This matches the roofline: per nonzero, n elements of B are
   gathered and n of C written, so arithmetic intensity is independent
   of n (~⅓ flop/byte) and only the negligible per-row metadata
   amortizes. The earlier report's "throughput falls once B outgrows L2"
   was an artifact of host-clock timing, see below.
5. **The transposed layout helps only the tile kernels** (cuTile jds +44% /
   +52% / +28%, cuTile 2pr +20% at n=64) and is catastrophic for the KA
   kernels (7–30 GFLOP/s at nb=1: consecutive threads then stride by n).
6. **General α/β (reading C)** costs the cuTile 2pr kernel ~40% (the extra
   5.2 GB read of C at n=256), cuSPARSE 17–50%, and the KA pm kernels
   nothing (they read C regardless).

## Timing-method correction

The previous version of this file (job 7972) timed with a host clock
around `CUDA.synchronize()`. That rounds every kernel longer than ~2 ms up
to the ~1 ms granularity of the host-side wait — 60% of those timings sat
within ±40 µs of a whole millisecond — and it also charged host overhead
to long kernels (KA jds pm at n=256: 33.4 ms then vs 20.9 ms now, cuTile
2pr at n=8: 0.53 ms vs 0.42 ms). Both effects grew with n, which produced
the spurious large-n slowdown (e.g. 215 → 158 GFLOP/s for KA jds pm, now
flat at ~251) and ±30–45% run-to-run swings in unchanged baselines at
n=64. Everything below is event-timed; the roofline argument from the old
analysis stands, its conclusion ("throughput must fall at large n") does
not — the plateau is what the data shows.

## Tables

### A·B (nodes×arcs, JDS formats), β = 0

| implementation | n=8 | n=64 | n=256 |
|---|---:|---:|---:|
| cuTile csr | 172 (0.95 ms) `8×8×32` | 133 (9.87 ms) `1×64×32` | 131 (39.87 ms) `1×64×32` |
| cuTile jds pm | 158 (1.04 ms) `32×8` | 115 (11.41 ms) `32×64` | 115 (45.69 ms) `32×64` |
| cuTile jds | 148 (1.11 ms) `32×8` | 112 (11.70 ms) `32×64` | 112 (46.95 ms) `32×64` |
| KA jds pm[nb=1] | **252 (0.65 ms)** | **250 (5.24 ms)** | **251 (20.89 ms)** |
| KA jds[nb=1] | 110 (1.49 ms) | 110 (11.89 ms) | 110 (47.51 ms) |
| KA jds pm[nb=4] | 188 (0.87 ms) | 189 (6.93 ms) | 190 (27.61 ms) |
| KA jds[nb=4] | 149 (1.10 ms) | 150 (8.75 ms) | 150 (34.93 ms) |
| KA jds pm[nb=8] | 138 (1.19 ms) | 137 (9.57 ms) | 137 (38.25 ms) |
| KA jds[nb=8] | 129 (1.27 ms) | 129 (10.14 ms) | 129 (40.52 ms) |
| KA csr | 104 (1.57 ms) | 104 (12.55 ms) | 104 (50.13 ms) |
| cuSPARSE | 127 (1.29 ms) | 132 (9.90 ms) | 133 (39.41 ms) |
| cuTile jds pm t | 228 (0.72 ms) `32×8` | 175 (7.47 ms) `32×64` | 147 (35.68 ms) `32×64` |
| cuTile jds t | 208 (0.79 ms) `32×8` | 166 (7.89 ms) `32×64` | 138 (37.82 ms) `32×64` |
| KA jds pm[nb=1 t] | 24 (6.76 ms) | 10 (131.15 ms) | 9 (596.61 ms) |
| KA jds[nb=1 t] | 22 (7.33 ms) | 10 (135.42 ms) | 8 (614.80 ms) |
| KA jds pm[nb=4 t] | 97 (1.69 ms) | 41 (32.03 ms) | 36 (147.27 ms) |
| KA jds[nb=4 t] | 90 (1.82 ms) | 40 (33.08 ms) | 34 (151.89 ms) |
| KA jds pm[nb=8 t] | 197 (0.83 ms) | 84 (15.68 ms) | 72 (72.43 ms) |
| KA jds[nb=8 t] | 182 (0.90 ms) | 81 (16.13 ms) | 70 (74.33 ms) |

### A·B (nodes×arcs, JDS formats), general α/β (reads C)

| implementation | n=8 | n=64 | n=256 |
|---|---:|---:|---:|
| cuTile csr | 128 (1.27 ms) `8×8×32` | 82 (16.01 ms) `1×64×32` | 82 (64.09 ms) `1×64×32` |
| cuTile jds pm | 140 (1.17 ms) `32×8` | 121 (10.82 ms) `32×64` | 122 (43.04 ms) `32×64` |
| cuTile jds | 133 (1.23 ms) `32×8` | 121 (10.80 ms) `32×64` | 122 (43.02 ms) `32×64` |
| KA jds pm[nb=1] | **252 (0.65 ms)** | **250 (5.24 ms)** | **251 (20.88 ms)** |
| KA jds pm[nb=4] | 188 (0.87 ms) | 189 (6.93 ms) | 190 (27.61 ms) |
| KA jds pm[nb=8] | 137 (1.19 ms) | 137 (9.57 ms) | 137 (38.24 ms) |
| cuSPARSE | 106 (1.54 ms) | 106 (12.41 ms) | 106 (49.59 ms) |
| cuTile jds pm t | 204 (0.80 ms) `32×8` | 191 (6.86 ms) `32×64` | 162 (32.42 ms) `32×64` |
| cuTile jds t | 187 (0.87 ms) `32×8` | 188 (6.98 ms) `32×64` | 159 (32.99 ms) `32×64` |
| KA jds pm[nb=1 t] | 24 (6.76 ms) | 10 (131.36 ms) | 9 (596.36 ms) |
| KA jds pm[nb=4 t] | 97 (1.68 ms) | 41 (32.04 ms) | 35 (147.81 ms) |
| KA jds pm[nb=8 t] | 197 (0.83 ms) | 84 (15.67 ms) | 72 (72.42 ms) |

### Aᵀ·B (arcs×nodes, 2-per-row formats), β = 0

| implementation | n=8 | n=64 | n=256 |
|---|---:|---:|---:|
| cuTile csr | 129 (1.27 ms) `32×8×16` | 49 (26.74 ms) `1×64×32` | 49 (106.91 ms) `1×64×32` |
| cuTile 2pr pm | **387 (0.42 ms) `32×8`** | 355 (3.69 ms) `32×64` | **352 (14.88 ms) `32×64`** |
| cuTile 2pr | 340 (0.48 ms) `32×8` | 348 (3.77 ms) `32×64` | 345 (15.21 ms) `32×64` |
| KA 2pr pm[nb=1] | 253 (0.65 ms) | 255 (5.13 ms) | 256 (20.49 ms) |
| KA 2pr[nb=1] | 119 (1.37 ms) | 119 (11.04 ms) | 118 (44.21 ms) |
| KA 2pr pm[nb=4] | 218 (0.75 ms) | 220 (5.95 ms) | 220 (23.79 ms) |
| KA 2pr[nb=4] | 189 (0.87 ms) | 191 (6.84 ms) | 192 (27.35 ms) |
| KA 2pr pm[nb=8] | 208 (0.79 ms) | 210 (6.25 ms) | 210 (25.00 ms) |
| KA 2pr[nb=8] | 197 (0.83 ms) | 198 (6.61 ms) | 198 (26.42 ms) |
| KA csr | 95 (1.73 ms) | 95 (13.80 ms) | 95 (55.18 ms) |
| cuSPARSE | 170 (0.96 ms) | 179 (7.32 ms) | 179 (29.24 ms) |
| cuTile 2pr pm t | 387 (0.42 ms) `32×8` | **425 (3.08 ms) `32×64`** | 346 (15.12 ms) `32×64` |
| cuTile 2pr t | 338 (0.48 ms) `32×8` | 420 (3.12 ms) `32×64` | 344 (15.21 ms) `32×64` |
| KA 2pr pm[nb=1 t] | 30 (5.38 ms) | 9 (138.69 ms) | 7 (725.59 ms) |
| KA 2pr[nb=1 t] | 28 (5.88 ms) | 9 (143.20 ms) | 7 (734.92 ms) |
| KA 2pr pm[nb=4 t] | 122 (1.34 ms) | 43 (30.34 ms) | 30 (177.49 ms) |
| KA 2pr[nb=4 t] | 112 (1.46 ms) | 38 (34.11 ms) | 29 (182.33 ms) |
| KA 2pr pm[nb=8 t] | 250 (0.65 ms) | 80 (16.28 ms) | 59 (88.87 ms) |
| KA 2pr[nb=8 t] | 229 (0.71 ms) | 79 (16.62 ms) | 59 (89.07 ms) |

### Aᵀ·B (arcs×nodes, 2-per-row formats), general α/β (reads C)

| implementation | n=8 | n=64 | n=256 |
|---|---:|---:|---:|
| cuTile csr | 128 (1.28 ms) `32×8×16` | 35 (37.12 ms) `1×64×32` | 35 (148.57 ms) `1×64×32` |
| cuTile 2pr pm | 214 (0.76 ms) `32×8` | 207 (6.32 ms) `32×64` | 207 (25.31 ms) `32×64` |
| cuTile 2pr | 211 (0.78 ms) `32×8` | 209 (6.26 ms) `32×64` | 209 (25.07 ms) `32×64` |
| KA 2pr pm[nb=1] | **253 (0.65 ms)** | 256 (5.13 ms) | **256 (20.49 ms)** |
| KA 2pr pm[nb=4] | 218 (0.75 ms) | 220 (5.95 ms) | 220 (23.80 ms) |
| KA 2pr pm[nb=8] | 208 (0.79 ms) | 210 (6.25 ms) | 210 (25.00 ms) |
| cuSPARSE | 92 (1.79 ms) | 89 (14.66 ms) | 89 (58.61 ms) |
| cuTile 2pr pm t | 240 (0.68 ms) `32×8` | **260 (5.04 ms) `32×64`** | 225 (23.27 ms) `32×64` |
| cuTile 2pr t | 232 (0.71 ms) `32×8` | 254 (5.15 ms) `32×64` | 222 (23.65 ms) `32×64` |
| KA 2pr pm[nb=1 t] | 30 (5.38 ms) | 9 (138.69 ms) | 7 (726.27 ms) |
| KA 2pr pm[nb=4 t] | 122 (1.34 ms) | 43 (30.34 ms) | 30 (177.74 ms) |
| KA 2pr pm[nb=8 t] | 250 (0.65 ms) | 80 (16.29 ms) | 59 (88.82 ms) |

## Reproducing

```
julia --project=. bench/spmm_flow.jl 8 64 256          # needs bench/data/*.jls
python3 bench/spmm_flow_tables.py ~/spmm-flow-<job>.log  # the tables above
```

`bench/data/` (gitignored, ~270 MB) is regenerated from the DIMACS files in
`~/min-cost-flow/road/` with MinimumCostFlows: read the problem, then
serialize `SparseMatrixCSC{Float32}` of A and Aᵀ as `flow_TX{,_t}.jls` and
the raw `JDSMatrixPM`/`Matrix2PerRowPM` arrays as `flow_TX_zoo.jls`
(NamedTuple with fields `jds_colidx, jds_iterptr, jds_nrows, jds_ncols,
tpr_colidx, tpr_ncols`) — see the header of `bench/spmm_flow.jl`.
