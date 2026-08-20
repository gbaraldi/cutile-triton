# Flow-matrix SpMM: cuTile vs KernelAbstractions vs cuSPARSE

Results of `bench/spmm_flow.jl` (job 7983, 2026-08-21) on the min-cost-flow
constraint matrix used by CoolPDLP: the node-arc incidence of the TX road
network (`road_flow_07_TX_{a..e}`), **A = 2,073,870 × 5,116,492 with
10,232,984 nonzeros** (all ±1, ~5 per row of A, ≤2 per row of Aᵀ).

- GPU: NVIDIA L40S (96 MB L2, ~864 GB/s DRAM), CUDA 13.1, Float32
- CUDA.jl 6.3.0, cuTile.jl 1.0.0, triton 3.7.1 through TileTriton
- C = α·A·B + β·C, timed with device events (`CUDA.@elapsed`, best of 20);
  every kernel's output verified against a CPU/device reference first.
  The β = 0 tables use each kernel's no-read-of-C specialization where it
  has one (cuTile `BETA_NZ`, KA `Val{BETA_NZ}`, the equivalent of
  MinimumCostFlows' `Zero()` dispatch); C is NaN-poisoned beforehand, so
  the check proves C really isn't read. KA csr (CoolPDLP's kernel, kept
  as is) and cuSPARSE read C regardless.
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

1. **With β = 0 specialized, the KA pm kernels are the fastest on both
   products.** A·B: KA jds pm 352 / 357 / 359 GFLOP/s for n = 8 / 64 / 256,
   2.7× cuSPARSE (127–133). Aᵀ·B: KA 2pr pm 433 / 441 / 442, 2.5× cuSPARSE
   (170–179) and 1.15–1.25× the cuTile 2pr kernel (385 / 355 / 352; 426 in
   the transposed layout at n=64). At CoolPDLP's n=256, Aᵀ·B takes 11.9 ms
   (KA 2pr pm) vs 14.9 ms (cuTile 2pr pm) vs 29.3 ms (cuSPARSE).
2. **Not reading C for β = 0 was worth +40% (KA jds pm, 252 → 357) and
   +72% (KA 2pr pm, 256 → 442)** — far more than the read's share of the
   traffic (~15–25%): the load-add-store dependency on C was serializing
   the thread, not just costing bytes. The previous report's KA numbers
   were the always-read path, so they understated MinimumCostFlows'
   real β = 0 `mul!`, which has had this specialization all along.
3. **KA JDS vs cuSPARSE, the nb question.** The pm variant beats cuSPARSE
   at every nb (357 / 234 / 152 for nb = 1 / 4 / 8 vs 132); nb=1 stays its
   best shape, blocking only costs it parallelism. The **vals variant needs
   nb=4 to win**: 120 at nb=1 (0.9× cuSPARSE) → **174 at nb=4 (1.3×)** →
   142 at nb=8. On Aᵀ the vals kernel keeps gaining with nb: 145 → 274 →
   322 (nb = 1 / 4 / 8) vs cuSPARSE 179. Blocking pays where the
   per-element shape re-reads ids and values n times, and nowhere else.
4. **cuTile CSR is no longer slow at n > 8.** Two changes: masked column
   ids are padded with 0 so the B gather bounds-masks them and issues no
   loads (before, every masked lane fetched B row 1 — 94% of the gather
   work on ≤2-nonzero rows at tile_k=32), and tile_k ∈ {4, 8} candidates
   let multi-row tiles fit under the 4096-element register cap at
   tile_n=64. A·B at n ≥ 64: 133 → **182** (`8×64×8`); Aᵀ·B: 129 / 49 / 49
   → **290 / 245 / 246** (`16×8×4`, `8×64×4`). It now beats cuSPARSE on
   both products (1.4× at n ≥ 64). What's left vs the format-specific
   kernels is the cap itself (8 rows per program at tile_n=64 vs 32 for
   2pr) and the per-row pointer gathers and masks the format needs;
   raising the cap with `num_warps=8`, or the transposed layout, are the
   untried knobs.
5. **The cuTile zoo kernels vs the KA pm kernels.** cuTile 2pr is 10–20%
   behind KA 2pr pm for β = 0 and comparable for general α/β (207–214 vs
   248–256 standard; 240–260 transposed). cuTile jds trails KA jds pm by
   2–3× in the standard layout (115 vs 357 at n ≥ 64) and 1.5–2.5× in the
   transposed one (228 / 175 / 147); the tile-granular B gathers don't
   beat one thread per (row, column) on this matrix.
6. **Throughput is flat in n** for every kernel (e.g. KA jds pm 352 / 357 /
   359, cuSPARSE 127 / 132 / 133): per nonzero, n elements of B are
   gathered and n of C written, so arithmetic intensity is independent of
   n (~⅓ flop/byte) and only the negligible per-row metadata amortizes.
7. **The transposed layout helps only the tile kernels** (cuTile jds
   +28–44%, cuTile 2pr +20% at n=64) and is catastrophic for the KA
   kernels (7–30 GFLOP/s at nb=1: consecutive threads then stride by n).
8. **General α/β (reading C)** costs the KA pm kernels 30–40%, cuTile 2pr
   ~40%, cuSPARSE 17–50%.

## Timing-method note

Job 7972's numbers (first version of this file) used a host clock around
`CUDA.synchronize()`, which rounds kernels longer than ~2 ms up to the
~1 ms granularity of the host-side wait and charges host overhead to long
kernels; that manufactured a spurious "slower at large n" effect and
±30–45% swings in unchanged baselines. Everything here is event-timed.

## Tables

### A·B (nodes×arcs, JDS formats), β = 0

| implementation | n=8 | n=64 | n=256 |
|---|---:|---:|---:|
| cuTile csr | 172 (0.95 ms) `8×8×16` | 182 (7.18 ms) `8×64×8` | 182 (28.74 ms) `8×64×8` |
| cuTile jds pm | 158 (1.04 ms) `32×8` | 115 (11.42 ms) `32×64` | 115 (45.70 ms) `32×64` |
| cuTile jds | 148 (1.11 ms) `32×8` | 112 (11.70 ms) `32×64` | 112 (46.89 ms) `32×64` |
| KA jds pm[nb=1] | **352 (0.46 ms)** | **357 (3.67 ms)** | **359 (14.60 ms)** |
| KA jds[nb=1] | 121 (1.35 ms) | 120 (10.87 ms) | 120 (43.47 ms) |
| KA jds pm[nb=4] | 232 (0.71 ms) | 234 (5.59 ms) | 236 (22.23 ms) |
| KA jds[nb=4] | 172 (0.95 ms) | 174 (7.55 ms) | 174 (30.15 ms) |
| KA jds pm[nb=8] | 154 (1.06 ms) | 152 (8.61 ms) | 152 (34.47 ms) |
| KA jds[nb=8] | 143 (1.14 ms) | 142 (9.20 ms) | 142 (36.85 ms) |
| KA csr | 104 (1.57 ms) | 104 (12.55 ms) | 104 (50.14 ms) |
| cuSPARSE | 127 (1.29 ms) | 132 (9.90 ms) | 133 (39.42 ms) |
| cuTile jds pm t | 228 (0.72 ms) `32×8` | 175 (7.48 ms) `32×64` | 147 (35.69 ms) `32×64` |
| cuTile jds t | 208 (0.79 ms) `32×8` | 166 (7.90 ms) `32×64` | 138 (37.82 ms) `32×64` |
| KA jds pm[nb=1 t] | 23 (7.01 ms) | 9 (139.32 ms) | 9 (611.40 ms) |
| KA jds[nb=1 t] | 22 (7.53 ms) | 9 (143.31 ms) | 8 (629.92 ms) |
| KA jds pm[nb=4 t] | 94 (1.75 ms) | 40 (32.39 ms) | 34 (151.69 ms) |
| KA jds[nb=4 t] | 87 (1.88 ms) | 39 (33.35 ms) | 34 (155.89 ms) |
| KA jds pm[nb=8 t] | 216 (0.76 ms) | 90 (14.47 ms) | 81 (64.40 ms) |
| KA jds[nb=8 t] | 198 (0.83 ms) | 88 (14.91 ms) | 79 (66.31 ms) |

### A·B (nodes×arcs, JDS formats), general α/β (reads C)

| implementation | n=8 | n=64 | n=256 |
|---|---:|---:|---:|
| cuTile csr | 129 (1.27 ms) `8×8×16` | 124 (10.55 ms) `8×64×8` | 124 (42.27 ms) `8×64×8` |
| cuTile jds pm | 140 (1.17 ms) `32×8` | 121 (10.82 ms) `32×64` | 122 (43.06 ms) `32×64` |
| cuTile jds | 133 (1.23 ms) `32×8` | 121 (10.80 ms) `32×64` | 122 (43.03 ms) `32×64` |
| KA jds pm[nb=1] | **252 (0.65 ms)** | **250 (5.24 ms)** | **251 (20.88 ms)** |
| KA jds pm[nb=4] | 188 (0.87 ms) | 189 (6.93 ms) | 190 (27.61 ms) |
| KA jds pm[nb=8] | 138 (1.19 ms) | 137 (9.57 ms) | 137 (38.22 ms) |
| cuSPARSE | 107 (1.53 ms) | 106 (12.40 ms) | 106 (49.58 ms) |
| cuTile jds pm t | 203 (0.80 ms) `32×8` | 191 (6.86 ms) `32×64` | 162 (32.40 ms) `32×64` |
| cuTile jds t | 188 (0.87 ms) `32×8` | 188 (6.98 ms) `32×64` | 159 (33.00 ms) `32×64` |
| KA jds pm[nb=1 t] | 24 (6.76 ms) | 10 (131.33 ms) | 9 (596.95 ms) |
| KA jds pm[nb=4 t] | 97 (1.68 ms) | 41 (31.99 ms) | 36 (147.36 ms) |
| KA jds pm[nb=8 t] | 197 (0.83 ms) | 84 (15.66 ms) | 72 (72.44 ms) |

### Aᵀ·B (arcs×nodes, 2-per-row formats), β = 0

| implementation | n=8 | n=64 | n=256 |
|---|---:|---:|---:|
| cuTile csr | 290 (0.57 ms) `16×8×4` | 245 (5.35 ms) `8×64×4` | 246 (21.33 ms) `8×64×4` |
| cuTile 2pr pm | 385 (0.42 ms) `32×8` | 355 (3.69 ms) `32×64` | 352 (14.87 ms) `32×64` |
| cuTile 2pr | 335 (0.49 ms) `32×8` | 348 (3.77 ms) `32×64` | 344 (15.21 ms) `32×64` |
| KA 2pr pm[nb=1] | **433 (0.38 ms)** | **441 (2.97 ms)** | **442 (11.86 ms)** |
| KA 2pr[nb=1] | 146 (1.12 ms) | 145 (9.01 ms) | 145 (36.08 ms) |
| KA 2pr pm[nb=4] | 337 (0.49 ms) | 341 (3.85 ms) | 340 (15.41 ms) |
| KA 2pr[nb=4] | 274 (0.60 ms) | 274 (4.78 ms) | 274 (19.14 ms) |
| KA 2pr pm[nb=8] | 363 (0.45 ms) | 365 (3.59 ms) | 365 (14.37 ms) |
| KA 2pr[nb=8] | 322 (0.51 ms) | 322 (4.06 ms) | 322 (16.25 ms) |
| KA csr | 95 (1.73 ms) | 95 (13.80 ms) | 95 (55.18 ms) |
| cuSPARSE | 170 (0.96 ms) | 179 (7.32 ms) | 179 (29.28 ms) |
| cuTile 2pr pm t | 387 (0.42 ms) `32×8` | 426 (3.08 ms) `32×64` | 346 (15.16 ms) `32×64` |
| cuTile 2pr t | 336 (0.49 ms) `32×8` | 420 (3.12 ms) `32×64` | 345 (15.20 ms) `32×64` |
| KA 2pr pm[nb=1 t] | 30 (5.40 ms) | 8 (166.58 ms) | 7 (745.89 ms) |
| KA 2pr[nb=1 t] | 28 (5.83 ms) | 8 (166.04 ms) | 7 (756.48 ms) |
| KA 2pr pm[nb=4 t] | 125 (1.31 ms) | 40 (32.85 ms) | 27 (191.44 ms) |
| KA 2pr[nb=4 t] | 113 (1.45 ms) | 40 (32.72 ms) | 28 (190.59 ms) |
| KA 2pr pm[nb=8 t] | 383 (0.43 ms) | 99 (13.26 ms) | 70 (74.93 ms) |
| KA 2pr[nb=8 t] | 336 (0.49 ms) | 94 (14.01 ms) | 70 (74.30 ms) |

### Aᵀ·B (arcs×nodes, 2-per-row formats), general α/β (reads C)

| implementation | n=8 | n=64 | n=256 |
|---|---:|---:|---:|
| cuTile csr | 134 (1.22 ms) `16×8×4` | 104 (12.59 ms) `8×64×4` | 104 (50.53 ms) `8×64×4` |
| cuTile 2pr pm | 214 (0.76 ms) `32×8` | 207 (6.32 ms) `32×64` | 207 (25.31 ms) `32×64` |
| cuTile 2pr | 211 (0.78 ms) `32×8` | 209 (6.26 ms) `32×64` | 209 (25.08 ms) `32×64` |
| KA 2pr pm[nb=1] | 248 (0.66 ms) | 255 (5.13 ms) | **256 (20.47 ms)** |
| KA 2pr pm[nb=4] | 218 (0.75 ms) | 220 (5.95 ms) | 220 (23.79 ms) |
| KA 2pr pm[nb=8] | 208 (0.79 ms) | 209 (6.25 ms) | 210 (25.00 ms) |
| cuSPARSE | 92 (1.79 ms) | 90 (14.64 ms) | 89 (58.58 ms) |
| cuTile 2pr pm t | 240 (0.68 ms) `32×8` | **260 (5.05 ms) `32×64`** | 225 (23.27 ms) `32×64` |
| cuTile 2pr t | 230 (0.71 ms) `32×8` | 255 (5.14 ms) `32×64` | 221 (23.67 ms) `32×64` |
| KA 2pr pm[nb=1 t] | 30 (5.39 ms) | 9 (138.69 ms) | 7 (725.93 ms) |
| KA 2pr pm[nb=4 t] | 122 (1.35 ms) | 43 (30.34 ms) | 30 (177.50 ms) |
| KA 2pr pm[nb=8 t] | **250 (0.66 ms)** | 80 (16.27 ms) | 59 (88.86 ms) |

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
