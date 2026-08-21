# Flow-matrix SpMM: cuTile vs KernelAbstractions vs cuSPARSE

Results of `bench/spmm_flow.jl` (job 7997, 2026-08-21; follow-ups 1–3 quote earlier jobs) on the min-cost-flow
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
| cuTile csr / jds / rc / 2pr | TileTriton tile kernels (`spmm_csr_kernels.jl`, `spmm_zoo_kernels.jl`) |
| … pm | value-free ±1 variants (`JDSMatrixPM`, `Matrix2PerRowPM`) |
| … t | fully transposed layout: B passed as n×k, C as n×m (rhs columns contiguous) |
| KA jds / rc / 2pr [nb=k] | KernelAbstractions SpMM ports of MinimumCostFlows' matrix_zoo kernels, one thread per row × `nb` rhs columns (SIMD.Vec accumulator); nb=1 is the original one-thread-per-element shape |
| KA csr | CoolPDLP's row-per-thread `spmm_csr!` |
| cuSPARSE | `CuSparseMatrixCSR` `mul!` |

## Headline

1. **With β = 0 specialized, the KA pm kernels are the fastest on both
   products.** A·B: KA jds pm 352 / 357 / 359 GFLOP/s for n = 8 / 64 / 256,
   2.7× cuSPARSE (127–133). Aᵀ·B: KA 2pr pm 433 / 441 / 442, 2.5× cuSPARSE
   (170–179) and 1.15–1.25× the cuTile 2pr kernel (385 / 355 / 352; 426 in
   the transposed layout at n=64). At CoolPDLP's n=256, Aᵀ·B takes 11.9 ms
   (KA 2pr pm) vs 14.9 ms (cuTile 2pr pm) vs 29.3 ms (cuSPARSE).
   Follow-up 4 moves both leaders: the range + CSR format (KA rc pm, 386)
   on A·B and the transposed-layout KA 2pr vals nb=8 (477) on Aᵀ·B.
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
| cuTile csr | 172 (0.95 ms) `8×8×16` | 182 (7.19 ms) `8×64×8` | 182 (28.75 ms) `8×64×8` |
| cuTile jds pm | 166 (0.99 ms) `32×8×8` | 167 (7.84 ms) `16×16×8` | 167 (31.37 ms) `32×8×8` |
| cuTile jds | 156 (1.05 ms) `32×8×8` | 162 (8.10 ms) `16×16×8` | 161 (32.48 ms) `16×16×8` |
| KA jds pm[nb=1] | 347 (0.47 ms) | 357 (3.66 ms) | 357 (14.67 ms) |
| KA jds[nb=1] | 121 (1.35 ms) | 121 (10.86 ms) | 121 (43.45 ms) |
| KA jds pm[nb=4] | 229 (0.71 ms) | 236 (5.55 ms) | 235 (22.29 ms) |
| KA jds[nb=4] | 171 (0.96 ms) | 174 (7.54 ms) | 174 (30.17 ms) |
| KA jds pm[nb=8] | 154 (1.07 ms) | 152 (8.62 ms) | 152 (34.51 ms) |
| KA jds[nb=8] | 143 (1.14 ms) | 142 (9.20 ms) | 142 (36.89 ms) |
| cuTile rc pm | 174 (0.94 ms) `16×8×4×4` | 176 (7.45 ms) `16×8×4×4` | 176 (29.75 ms) `16×8×4×4` |
| cuTile rc | 155 (1.06 ms) `16×8×4×8` | 165 (7.92 ms) `16×32×4×4` | 166 (31.64 ms) `16×32×4×4` |
| KA rc pm[nb=1] | **385 (0.42 ms)** | **386 (3.39 ms)** | **386 (13.58 ms)** |
| KA rc[nb=1] | 144 (1.14 ms) | 144 (9.06 ms) | 145 (36.25 ms) |
| KA rc pm[nb=4] | 231 (0.71 ms) | 237 (5.53 ms) | 237 (22.13 ms) |
| KA rc[nb=4] | 149 (1.10 ms) | 149 (8.77 ms) | 149 (35.07 ms) |
| KA rc pm[nb=8] | 156 (1.05 ms) | 155 (8.46 ms) | 155 (33.82 ms) |
| KA rc[nb=8] | 139 (1.18 ms) | 140 (9.37 ms) | 140 (37.41 ms) |
| KA csr | 104 (1.57 ms) | 104 (12.55 ms) | 104 (50.17 ms) |
| cuSPARSE | 127 (1.29 ms) | 132 (9.89 ms) | 133 (39.42 ms) |
| cuTile jds pm t | 245 (0.67 ms) `16×8×4` | 346 (3.79 ms) `16×64×8` | 340 (15.40 ms) `16×64×8` |
| cuTile jds t | 226 (0.72 ms) `32×8×8` | 339 (3.86 ms) `16×64×8` | 333 (15.72 ms) `16×64×8` |
| KA jds pm[nb=1 t] | 248 (0.66 ms) | 353 (3.71 ms) | 369 (14.20 ms) |
| KA jds[nb=1 t] | 228 (0.72 ms) | 340 (3.86 ms) | 353 (14.84 ms) |
| KA jds pm[nb=4 t] | 220 (0.74 ms) | 355 (3.69 ms) | 370 (14.17 ms) |
| KA jds[nb=4 t] | 199 (0.82 ms) | 349 (3.76 ms) | 368 (14.22 ms) |
| KA jds pm[nb=8 t] | 214 (0.76 ms) | 354 (3.70 ms) | 370 (14.14 ms) |
| KA jds[nb=8 t] | 198 (0.83 ms) | 347 (3.77 ms) | 369 (14.21 ms) |
| cuTile rc pm t | 267 (0.61 ms) `16×8×4×4` | 230 (5.70 ms) `16×32×4×4` | 224 (23.44 ms) `16×32×4×4` |
| cuTile rc t | 231 (0.71 ms) `16×8×4×4` | 227 (5.77 ms) `16×32×4×4` | 221 (23.74 ms) `16×64×4×4` |
| KA rc pm[nb=1 t] | 275 (0.60 ms) | 355 (3.69 ms) | 370 (14.17 ms) |
| KA rc[nb=1 t] | 235 (0.70 ms) | 344 (3.81 ms) | 366 (14.32 ms) |
| KA rc pm[nb=4 t] | 236 (0.69 ms) | 355 (3.69 ms) | 368 (14.23 ms) |
| KA rc[nb=4 t] | 202 (0.81 ms) | 344 (3.81 ms) | 366 (14.31 ms) |
| KA rc pm[nb=8 t] | 226 (0.73 ms) | 353 (3.71 ms) | 370 (14.16 ms) |
| KA rc[nb=8 t] | 199 (0.82 ms) | 343 (3.82 ms) | 367 (14.28 ms) |

### A·B (nodes×arcs, JDS formats), general α/β (reads C)

| implementation | n=8 | n=64 | n=256 |
|---|---:|---:|---:|
| cuTile csr | 129 (1.27 ms) `8×8×16` | 124 (10.57 ms) `8×64×8` | 124 (42.36 ms) `8×64×8` |
| cuTile jds pm | 145 (1.13 ms) `32×8×8` | 130 (10.09 ms) `16×16×8` | 146 (35.94 ms) `32×8×8` |
| cuTile jds | 137 (1.19 ms) `32×8×8` | 126 (10.37 ms) `16×16×8` | 126 (41.42 ms) `16×16×8` |
| KA jds pm[nb=1] | 251 (0.65 ms) | 250 (5.23 ms) | 250 (20.92 ms) |
| KA jds pm[nb=4] | 187 (0.88 ms) | 190 (6.91 ms) | 189 (27.68 ms) |
| KA jds pm[nb=8] | 137 (1.19 ms) | 137 (9.57 ms) | 137 (38.26 ms) |
| cuTile rc pm | 133 (1.23 ms) `16×8×4×4` | 135 (9.73 ms) `16×8×4×4` | 135 (38.89 ms) `16×8×4×4` |
| cuTile rc | 121 (1.36 ms) `16×8×4×8` | 134 (9.75 ms) `16×32×4×4` | 134 (39.13 ms) `16×32×4×4` |
| KA rc pm[nb=1] | **272 (0.60 ms)** | 270 (4.85 ms) | 270 (19.43 ms) |
| KA rc pm[nb=4] | 191 (0.86 ms) | 195 (6.73 ms) | 195 (26.89 ms) |
| KA rc pm[nb=8] | 140 (1.17 ms) | 140 (9.35 ms) | 140 (37.36 ms) |
| cuSPARSE | 107 (1.53 ms) | 106 (12.41 ms) | 106 (49.63 ms) |
| cuTile jds pm t | 217 (0.75 ms) `16×8×4` | 292 (4.48 ms) `16×64×8` | 288 (18.23 ms) `16×64×8` |
| cuTile jds t | 202 (0.81 ms) `32×8×8` | 288 (4.56 ms) `16×64×8` | 282 (18.57 ms) `16×64×8` |
| KA jds pm[nb=1 t] | 219 (0.75 ms) | 296 (4.43 ms) | 307 (17.08 ms) |
| KA jds pm[nb=4 t] | 198 (0.83 ms) | 296 (4.42 ms) | **308 (16.99 ms)** |
| KA jds pm[nb=8 t] | 197 (0.83 ms) | 296 (4.42 ms) | 308 (17.03 ms) |
| cuTile rc pm t | 239 (0.69 ms) `16×8×4×4` | 223 (5.87 ms) `16×32×4×4` | 215 (24.41 ms) `16×32×4×4` |
| cuTile rc t | 206 (0.79 ms) `16×8×4×4` | 220 (5.94 ms) `16×32×4×4` | 213 (24.61 ms) `16×64×4×4` |
| KA rc pm[nb=1 t] | 238 (0.69 ms) | **297 (4.41 ms)** | 308 (17.03 ms) |
| KA rc pm[nb=4 t] | 210 (0.78 ms) | 296 (4.42 ms) | 308 (17.02 ms) |
| KA rc pm[nb=8 t] | 203 (0.80 ms) | 296 (4.43 ms) | 308 (17.04 ms) |

### Aᵀ·B (arcs×nodes, 2-per-row formats), β = 0

| implementation | n=8 | n=64 | n=256 |
|---|---:|---:|---:|
| cuTile csr | 286 (0.57 ms) `16×8×4` | 245 (5.35 ms) `8×64×4` | 245 (21.35 ms) `8×64×4` |
| cuTile 2pr pm | 388 (0.42 ms) `32×8×4` | 424 (3.09 ms) `16×32×8` | 425 (12.32 ms) `16×32×8` |
| cuTile 2pr | 345 (0.48 ms) `32×8×8` | 385 (3.40 ms) `32×32×8` | 386 (13.56 ms) `32×32×8` |
| KA 2pr pm[nb=1] | **432 (0.38 ms)** | 440 (2.98 ms) | 441 (11.89 ms) |
| KA 2pr[nb=1] | 141 (1.16 ms) | 141 (9.26 ms) | 142 (37.03 ms) |
| KA 2pr pm[nb=4] | 339 (0.48 ms) | 340 (3.85 ms) | 340 (15.41 ms) |
| KA 2pr[nb=4] | 274 (0.60 ms) | 274 (4.78 ms) | 274 (19.13 ms) |
| KA 2pr pm[nb=8] | 364 (0.45 ms) | 365 (3.59 ms) | 364 (14.37 ms) |
| KA 2pr[nb=8] | 325 (0.50 ms) | 322 (4.06 ms) | 322 (16.29 ms) |
| KA csr | 94 (1.73 ms) | 95 (13.80 ms) | 95 (55.19 ms) |
| cuSPARSE | 170 (0.96 ms) | 179 (7.32 ms) | 179 (29.26 ms) |
| cuTile 2pr pm t | 391 (0.42 ms) `64×8×8` | 455 (2.88 ms) `16×64×8` | 392 (13.35 ms) `16×64×8` |
| cuTile 2pr t | 350 (0.47 ms) `32×8×8` | 444 (2.95 ms) `16×64×8` | 377 (13.90 ms) `16×64×8` |
| KA 2pr pm[nb=1 t] | 372 (0.44 ms) | 434 (3.02 ms) | 440 (11.92 ms) |
| KA 2pr[nb=1 t] | 340 (0.48 ms) | 424 (3.09 ms) | 432 (12.13 ms) |
| KA 2pr pm[nb=4 t] | 392 (0.42 ms) | 469 (2.79 ms) | 473 (11.07 ms) |
| KA 2pr[nb=4 t] | 340 (0.48 ms) | 457 (2.86 ms) | 470 (11.14 ms) |
| KA 2pr pm[nb=8 t] | 384 (0.43 ms) | **474 (2.76 ms)** | **481 (10.90 ms)** |
| KA 2pr[nb=8 t] | 337 (0.49 ms) | 462 (2.84 ms) | 477 (10.98 ms) |

### Aᵀ·B (arcs×nodes, 2-per-row formats), general α/β (reads C)

| implementation | n=8 | n=64 | n=256 |
|---|---:|---:|---:|
| cuTile csr | 134 (1.22 ms) `16×8×4` | 104 (12.65 ms) `8×64×4` | 104 (50.55 ms) `8×64×4` |
| cuTile 2pr pm | 214 (0.76 ms) `32×8×4` | 160 (8.20 ms) `16×32×8` | 160 (32.72 ms) `16×32×8` |
| cuTile 2pr | 206 (0.79 ms) `32×8×8` | 205 (6.38 ms) `32×32×8` | 205 (25.51 ms) `32×32×8` |
| KA 2pr pm[nb=1] | 249 (0.66 ms) | 257 (5.10 ms) | 256 (20.46 ms) |
| KA 2pr pm[nb=4] | 218 (0.75 ms) | 220 (5.95 ms) | 220 (23.80 ms) |
| KA 2pr pm[nb=8] | 208 (0.79 ms) | 209 (6.26 ms) | 210 (25.01 ms) |
| cuSPARSE | 92 (1.78 ms) | 89 (14.68 ms) | 89 (58.59 ms) |
| cuTile 2pr pm t | **254 (0.64 ms) `64×8×8`** | 268 (4.88 ms) `16×64×8` | 257 (20.41 ms) `16×64×8` |
| cuTile 2pr t | 234 (0.70 ms) `32×8×8` | 265 (4.94 ms) `16×64×8` | 242 (21.61 ms) `16×64×8` |
| KA 2pr pm[nb=1 t] | 249 (0.66 ms) | 281 (4.66 ms) | 281 (18.63 ms) |
| KA 2pr pm[nb=4 t] | 245 (0.67 ms) | **285 (4.59 ms)** | 287 (18.26 ms) |
| KA 2pr pm[nb=8 t] | 253 (0.65 ms) | 285 (4.59 ms) | **288 (18.20 ms)** |


## Follow-up 1: why KA 2pr pm beats cuTile 2pr pm — tile/num_warps sweep (job 7985)

`zoo_tile_candidates` was widened to tile_n ∈ {8, 16, 64} and
num_warps ∈ {4, 8}, with `TRITON_KERNEL_INFO=1` printing registers and
spills per compiled kernel.

- **Register pressure is not the cause**: every 2pr pm config compiles to
  12–64 registers, JDS to ≤128, all with zero spills.
- **Tile shape is part of it.** Narrow slabs at 8 warps win everywhere:

  | β = 0, GFLOP/s | n=8 | n=64 | n=256 |
  |---|---:|---:|---:|
  | cuTile 2pr pm, 32×64 @4 warps (old best) | 385 | 355 | 352 |
  | cuTile 2pr pm, 32×16 @8 warps | 385 | **410** | **411** |
  | cuTile 2pr pm t, 32×64 @8 warps | 392 | **433** | 346 |
  | KA 2pr pm | 433 | 440 | 441 |
  | cuTile jds pm, 32×8 @8 warps | 166 | 167 | 167 |
  | cuTile jds pm t, 32×64 @8 warps | 244 | 298 | 277 |
  | KA jds pm | 352 | 357 | 358 |

  The 2pr gap to KA shrinks from ~20% to ~7% (a tie at n=64 transposed);
  JDS gains 45% standard / 70% transposed but KA jds pm stays 1.2× ahead.
- **What the remaining gap is**: per-element index arithmetic and bounds
  masks of the 2-D gathers, plus the layout conversions Triton inserts
  for the reshape/broadcast of the id tiles — most configs carry 1–16 KB
  of shared memory for exactly those `convert_layout` round trips. The KA
  kernel does one index computation per element and nothing else.

## Follow-up 2: cuTile native (tileiras) vs TileTriton on the same kernels (job 7990)

`bench/spmm_backends.jl` launches the same kernel functions through
`cuTile.launch` with `CUTILE_BACKEND=native` (Tile IR → tileiras 13.3.36)
and `triton` (TileTriton's shim replacing `cuTile.cufunction`), on the
flow matrix, β = 0, a few fixed configs at 4 warps. Native Tile IR on the
L40S (sm_89) needs bytecode ≥ 13.2, i.e. a CUDA ≥ 13.2 toolchain: the
runtime preference was moved from 13.1 to 13.3 (runs on the 13.1 driver
via CUDA minor-version compatibility).

| case | kernel (config) | native | triton | triton / native |
|---|---|---:|---:|---:|
| A n=8 | csr 8×8×16 | 133 | 170 | 1.28× |
| A n=8 | csr 16×8×4 | 167 | 161 | 0.96× |
| A n=8 | jds pm 32×8 | 170 | 157 | 0.92× |
| A n=64 | csr 8×64×8 | 115 | 182 | 1.58× |
| A n=64 | csr 8×64×4 | 106 | 165 | 1.56× |
| A n=64 | jds pm 32×64 | 66 | 114 | 1.72× |
| A n=64 | jds pm 32×16 | 154 | 153 | 0.99× |
| A n=256 | csr 8×64×8 | 116 | 182 | 1.56× |
| A n=256 | csr 8×64×4 | 107 | 166 | 1.55× |
| A n=256 | jds pm 32×64 | 67 | 115 | 1.72× |
| A n=256 | jds pm 32×16 | 155 | 153 | 0.99× |
| Aᵀ n=8 | csr 8×8×16 | 74 | 102 | 1.38× |
| Aᵀ n=8 | csr 16×8×4 | 136 | 287 | 2.11× |
| Aᵀ n=8 | 2pr pm 32×8 | 389 | 384 | 0.99× |
| Aᵀ n=64 | csr 8×64×8 | 137 | 164 | 1.20× |
| Aᵀ n=64 | csr 8×64×4 | 124 | 245 | 1.98× |
| Aᵀ n=64 | 2pr pm 32×64 | 141 | 273 | 1.93× |
| Aᵀ n=64 | 2pr pm 32×16 | 304 | 396 | 1.30× |
| Aᵀ n=256 | csr 8×64×8 | 137 | 164 | 1.20× |
| Aᵀ n=256 | csr 8×64×4 | 124 | 245 | 1.98× |
| Aᵀ n=256 | 2pr pm 32×64 | 141 | 274 | 1.94× |
| Aᵀ n=256 | 2pr pm 32×16 | 305 | 397 | 1.30× |

(GFLOP/s; both backends verified against the same reference on every row.)

- **TileTriton is ahead or tied on every config but two**, and the two
  native wins are within 4–8% at n=8. The advantage grows with slab width
  and with gather waste: 1.5–1.7× on the wide 8×64 / 32×64 tiles of A,
  up to 2.1× on Aᵀ's CSR tiles where most gather lanes are masked.
- **tileiras handles wide gathered slabs poorly**: native 2pr pm drops
  from 389 (32×8) to 141 (32×64) while Triton holds 273–385; native jds
  pm 32×64 is 2.3× slower than its own 32×16. Both backends prefer the
  narrow slab, and on the narrow slabs they converge (jds pm 32×16:
  154 vs 153; 2pr pm 32×8: 389 vs 384).
- Caveat: the shim path is not bit-identical to the explicit `TritonRun`
  path used by `spmm_flow.jl` — it takes cuTile's default `ArraySpec`
  (weaker stride/alignment hints) and its own autotune candidates. The
  same 2pr pm 32×64 config measures 273 here vs 355 there, so the
  explicit path's numbers are the better TileTriton figures; the table
  above is the like-for-like comparison.

## Follow-up 3: the last 7% of cuTile 2pr pm vs KA (bench/spmm_2pr_variants.jl)

Variants of the 2pr pm kernel isolating each suspected cost, Aᵀ·B, β = 0,
n = 64 / 256 (identical at both), GFLOP/s at 32×16 / 8 warps:

| variant | what changes | n=64 |
|---|---|---:|
| V0 current | `eachtile` ids + reshape, 2-D bounds-checked B gather, block store | 410 |
| V1 | ids via flat 1-D gather + `expand_dims` | 410 |
| V2 | V1 + flat 1-D B gather, precomputed column offsets, one explicit mask, `check_bounds=false` | 409 |
| V4 | `eachtile` ids + flat B gather | 410 |
| KA 2pr pm | one thread per element | 441 |

Every 2pr config compiles to 12–28 registers, no spills, **no shared
memory** (no layout conversions), and the variants' TTIR differs only in
the expected reshape/broadcast counts. Neither the id path nor the gather
arithmetic nor the bounds masks cost anything measurable; tall narrow
tiles (`tile_n = 4`) are *worse*, monotonically with height (64×4: 352,
512×4: 324), ruling out column locality too.

What moves it is tile granularity towards KA's one element per thread.
Below the sweep's `tile_m = 32` floor:

| config (8 warps) | n=64 | n=256 |
|---|---:|---:|
| 32×16 (sweep best) | 410 | 411 |
| **16×32** | **424** | **425** |
| 16×16 @ 4 warps | 413 | 413 |
| 8×32 | 250 | 250 |
| KA 2pr pm | 441 | 441 |

16×32 at 8 warps — exactly 2 elements per thread — leaves a ~4% gap, and
8 rows falls off a cliff (the row-direction coalescing of the C store
breaks below 16 rows). Nothing is left inside the kernel body; the
residue is the fixed per-program cost (prologue, index tiles, grid
bookkeeping) amortized over 512 elements, which a bare SIMT kernel does
not pay. `zoo_tile_candidates` now includes `tile_m = 16` and
`tile_n = 32` (and drops the never-winning `tile_m ≥ 128`).

## Follow-up 4: a better format than JDS for A·B — range + CSR (job 7997)

With arcs numbered by one endpoint (the CSC order of the adjacency matrix
in `construct_constraint_matrix`), each row of the nodes×arcs incidence
is one **contiguous** column range (the −1 entries; on the TX matrix in
every row, max length 8) plus a scattered list (the +1 entries, exactly
nnz/2). JDS and CSR store an id and gather a B row for all of them; the
"rc" format stores only `[lo, hi)` per row for the range — no column ids,
a dense B strip — and a CSR for the scattered half (`range_csr` derives
it from A's CSR and asserts the contiguity). Same run also switched the
KA kernels to consecutive-columns-per-consecutive-thread in the transposed
layout, which makes the B-row gathers and C stores coalesced there.

A·B, β = 0, GFLOP/s (n = 8 / 64 / 256):

| | standard layout | transposed (`t`) |
|---|---|---|
| KA rc pm (nb=1) | **385 / 386 / 386** | 275 / 355 / 370 |
| KA jds pm (nb=1) | 347 / 357 / 357 | 248 / 353 / 369 |
| KA rc vals (nb=1) | 144 / 144 / 145 | 235 / 344 / 366 |
| KA jds vals (nb=1) | 121 / 121 / 121 | 228 / 340 / 353 |
| cuTile rc pm | 174 / 176 / 176 | 267 / 230 / 224 |
| cuTile jds pm | 166 / 167 / 167 | 245 / 346 / 340 |
| cuSPARSE | 127 / 132 / 133 | – |

- **Range + CSR is better than JDS for the same kernel shape**: +8–11%
  for KA pm (386 vs 357; α/β path 270 vs 250), +19% for KA vals (144 vs
  121). Half the index loads disappear and half the B reads become a
  contiguous strip. KA rc pm at 386 is ~63% of the A·B roofline
  (read B once + write C ≈ 7.3 GB at n=256 → ~615 GFLOP/s).
- **The transposed layout with the right thread order nearly erases the
  vals penalty**: KA rc vals 366 vs rc pm 370 at n=256 (standard layout:
  145 vs 386). A gathered B row is n contiguous floats there, so the
  per-entry value load is amortized over a coalesced row instead of
  competing with 4-byte scattered gathers. Previously the KA `t` rows
  were 7–30 GFLOP/s purely because of the row-fastest thread order.
- **The cuTile rc kernel does not keep up** (176 standard, 230 `t`, vs
  cuTile jds `t` 346): with tile_k=4 on ranges of average length 2.5 and
  unsorted scattered lengths within a tile, most gather lanes are masked;
  JDS's length-sorted rows avoid that. tile_k ∈ {1, 2} candidates and
  sorting rows by scattered length are the obvious next steps. The
  tile_m=16 candidates added in follow-up 3 lifted cuTile jds pm `t` from
  298 to 346 (`16×64`, 8 warps).
- **Aᵀ·B side effects of the thread-order fix**: KA 2pr vals nb=8 `t`
  reaches **462 / 477** at n = 64 / 256 — the fastest Aᵀ·B of all, above
  KA 2pr pm nb=1 (441) — each thread reads 8 contiguous floats (a full
  32-byte sector) per B row. cuTile 2pr pm `t` hits 455 at n=64 (`16×64`,
  8 warps) but 392 at n=256; standard layout `16×32` @ 8 warps: 424 / 425.

## Reproducing

```
julia --project=. bench/spmm_flow.jl 8 64 256          # needs bench/data/*.jls
CUTILE_BACKEND=native julia --project=. bench/spmm_backends.jl 8 64 256  # then triton
python3 bench/spmm_flow_tables.py ~/spmm-flow-<job>.log  # the tables above
```

`bench/data/` (gitignored, ~270 MB) is regenerated from the DIMACS files in
`~/min-cost-flow/road/` with MinimumCostFlows: read the problem, then
serialize `SparseMatrixCSC{Float32}` of A and Aᵀ as `flow_TX{,_t}.jls` and
the raw `JDSMatrixPM`/`Matrix2PerRowPM` arrays as `flow_TX_zoo.jls`
(NamedTuple with fields `jds_colidx, jds_iterptr, jds_nrows, jds_ncols,
tpr_colidx, tpr_ncols`) — see the header of `bench/spmm_flow.jl`.
