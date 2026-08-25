# Flow-matrix SpMM: cuTile vs KernelAbstractions vs cuSPARSE

Results of `bench/spmm_flow.jl` (job 8004, 2026-08-21; follow-ups 1–3 quote earlier jobs) on the min-cost-flow
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
| … pm | value-free ±1 variants (`JDSMatrixPM`, `Matrix2PerRowPM`); a label **without** `pm` is the explicit-values (vals) variant of the same format |
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
   Follow-up 7 moves the A·B leader again: permuting the rows to node
   order (where the rc ranges chain) takes KA rc pm to **509**.
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

**Larger instance (follow-up 6):** on `vision_rnd_05` (4.5× TX, two rows
with 92k entries) the row-parallel kernels collapse to 0.15–0.6× cuSPARSE
on A·B until the heavy rows are peeled into chunk-parallel partial sums;
the resulting hybrid ("rch") reaches 568 GFLOP/s (KA) / 541 (cuTile) at
n=64, 2× cuSPARSE. Aᵀ·B is unaffected (541 / 531, 2.6× cuSPARSE).

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
| cuTile csr | 172 (0.95 ms) `8×8×16` | 182 (7.19 ms) `8×64×8` | 182 (28.76 ms) `8×64×8` |
| cuTile jds pm | 166 (0.98 ms) `32×8×8` | 167 (7.83 ms) `32×8×8` | 167 (31.40 ms) `32×8×8` |
| cuTile jds | 156 (1.05 ms) `32×8×8` | 162 (8.09 ms) `16×16×8` | 161 (32.51 ms) `16×16×8` |
| KA jds pm[nb=1] | 347 (0.47 ms) | 358 (3.66 ms) | 357 (14.68 ms) |
| KA jds[nb=1] | 121 (1.35 ms) | 121 (10.86 ms) | 121 (43.44 ms) |
| KA jds pm[nb=4] | 229 (0.71 ms) | 236 (5.55 ms) | 235 (22.27 ms) |
| KA jds[nb=4] | 171 (0.96 ms) | 174 (7.54 ms) | 174 (30.16 ms) |
| KA jds pm[nb=8] | 153 (1.07 ms) | 152 (8.61 ms) | 152 (34.51 ms) |
| KA jds[nb=8] | 143 (1.14 ms) | 142 (9.21 ms) | 142 (36.88 ms) |
| cuTile rc pm | 178 (0.92 ms) `16×8×2×8` | 180 (7.30 ms) `16×8×2×8` | 180 (29.13 ms) `16×8×2×8` |
| cuTile rc | 157 (1.04 ms) `16×8×1×8` | 167 (7.83 ms) `16×16×2×8` | 168 (31.25 ms) `16×16×2×8` |
| KA rc pm[nb=1] | **385 (0.42 ms)** | **386 (3.39 ms)** | **386 (13.58 ms)** |
| KA rc[nb=1] | 144 (1.14 ms) | 144 (9.06 ms) | 145 (36.24 ms) |
| KA rc pm[nb=4] | 231 (0.71 ms) | 237 (5.53 ms) | 237 (22.11 ms) |
| KA rc[nb=4] | 149 (1.10 ms) | 149 (8.78 ms) | 150 (35.05 ms) |
| KA rc pm[nb=8] | 156 (1.05 ms) | 155 (8.46 ms) | 155 (33.82 ms) |
| KA rc[nb=8] | 139 (1.18 ms) | 140 (9.37 ms) | 140 (37.41 ms) |
| KA csr | 104 (1.57 ms) | 104 (12.55 ms) | 104 (50.19 ms) |
| cuSPARSE | 127 (1.29 ms) | 132 (9.90 ms) | 133 (39.44 ms) |
| cuTile jds pm t | 245 (0.67 ms) `16×8×4` | 346 (3.79 ms) `16×64×8` | 340 (15.41 ms) `16×64×8` |
| cuTile jds t | 226 (0.72 ms) `32×8×8` | 339 (3.86 ms) `16×64×8` | 333 (15.73 ms) `16×64×8` |
| KA jds pm[nb=1 t] | 248 (0.66 ms) | 353 (3.71 ms) | 369 (14.20 ms) |
| KA jds[nb=1 t] | 228 (0.72 ms) | 340 (3.85 ms) | 353 (14.83 ms) |
| KA jds pm[nb=4 t] | 220 (0.74 ms) | 355 (3.69 ms) | 370 (14.18 ms) |
| KA jds[nb=4 t] | 199 (0.82 ms) | 349 (3.76 ms) | 369 (14.20 ms) |
| KA jds pm[nb=8 t] | 216 (0.76 ms) | 354 (3.70 ms) | 370 (14.15 ms) |
| KA jds[nb=8 t] | 198 (0.83 ms) | 347 (3.77 ms) | 369 (14.21 ms) |
| cuTile rc pm t | 276 (0.59 ms) `32×8×1×8` | 342 (3.83 ms) `16×64×1×8` | 335 (15.63 ms) `16×64×1×8` |
| cuTile rc t | 236 (0.69 ms) `16×8×1×8` | 332 (3.95 ms) `16×64×1×8` | 326 (16.07 ms) `16×64×1×8` |
| KA rc pm[nb=1 t] | 275 (0.60 ms) | 355 (3.69 ms) | 370 (14.17 ms) |
| KA rc[nb=1 t] | 235 (0.70 ms) | 344 (3.81 ms) | 366 (14.32 ms) |
| KA rc pm[nb=4 t] | 236 (0.69 ms) | 355 (3.69 ms) | 368 (14.22 ms) |
| KA rc[nb=4 t] | 202 (0.81 ms) | 344 (3.81 ms) | 366 (14.33 ms) |
| KA rc pm[nb=8 t] | 225 (0.73 ms) | 353 (3.71 ms) | 370 (14.16 ms) |
| KA rc[nb=8 t] | 199 (0.82 ms) | 342 (3.82 ms) | 367 (14.28 ms) |

### A·B (nodes×arcs, JDS formats), general α/β (reads C)

| implementation | n=8 | n=64 | n=256 |
|---|---:|---:|---:|
| cuTile csr | 129 (1.27 ms) `8×8×16` | 124 (10.57 ms) `8×64×8` | 124 (42.38 ms) `8×64×8` |
| cuTile jds pm | 145 (1.13 ms) `32×8×8` | 146 (9.00 ms) `32×8×8` | 146 (35.94 ms) `32×8×8` |
| cuTile jds | 138 (1.19 ms) `32×8×8` | 126 (10.37 ms) `16×16×8` | 126 (41.43 ms) `16×16×8` |
| KA jds pm[nb=1] | 251 (0.65 ms) | 250 (5.23 ms) | 250 (20.92 ms) |
| KA jds pm[nb=4] | 187 (0.88 ms) | 190 (6.91 ms) | 189 (27.66 ms) |
| KA jds pm[nb=8] | 137 (1.19 ms) | 137 (9.57 ms) | 137 (38.24 ms) |
| cuTile rc pm | 149 (1.10 ms) `16×8×2×8` | 151 (8.70 ms) `16×8×2×8` | 151 (34.76 ms) `16×8×2×8` |
| cuTile rc | 125 (1.31 ms) `16×8×1×8` | 135 (9.67 ms) `16×16×2×8` | 136 (38.61 ms) `16×16×2×8` |
| KA rc pm[nb=1] | **272 (0.60 ms)** | 270 (4.86 ms) | 270 (19.43 ms) |
| KA rc pm[nb=4] | 191 (0.86 ms) | 195 (6.73 ms) | 195 (26.87 ms) |
| KA rc pm[nb=8] | 140 (1.17 ms) | 140 (9.35 ms) | 140 (37.35 ms) |
| cuSPARSE | 107 (1.54 ms) | 106 (12.41 ms) | 106 (49.61 ms) |
| cuTile jds pm t | 217 (0.75 ms) `16×8×4` | 293 (4.47 ms) `16×64×8` | 287 (18.23 ms) `16×64×8` |
| cuTile jds t | 202 (0.81 ms) `32×8×8` | 287 (4.56 ms) `16×64×8` | 282 (18.57 ms) `16×64×8` |
| KA jds pm[nb=1 t] | 219 (0.75 ms) | 296 (4.43 ms) | 307 (17.08 ms) |
| KA jds pm[nb=4 t] | 198 (0.83 ms) | 296 (4.42 ms) | **308 (16.98 ms)** |
| KA jds pm[nb=8 t] | 197 (0.83 ms) | 296 (4.42 ms) | 308 (17.03 ms) |
| cuTile rc pm t | 239 (0.69 ms) `32×8×1×8` | 289 (4.53 ms) `16×64×1×8` | 284 (18.46 ms) `16×64×1×8` |
| cuTile rc t | 210 (0.78 ms) `16×8×1×8` | 282 (4.65 ms) `16×64×1×8` | 276 (18.96 ms) `16×64×1×8` |
| KA rc pm[nb=1 t] | 238 (0.69 ms) | **297 (4.41 ms)** | 308 (17.04 ms) |
| KA rc pm[nb=4 t] | 211 (0.78 ms) | 296 (4.43 ms) | 308 (17.02 ms) |
| KA rc pm[nb=8 t] | 204 (0.80 ms) | 296 (4.43 ms) | 308 (17.04 ms) |

### Aᵀ·B (arcs×nodes, 2-per-row formats), β = 0

| implementation | n=8 | n=64 | n=256 |
|---|---:|---:|---:|
| cuTile csr | 287 (0.57 ms) `16×8×4` | 245 (5.34 ms) `8×64×4` | 246 (21.33 ms) `8×64×4` |
| cuTile 2pr pm | 385 (0.42 ms) `32×8×4` | 424 (3.09 ms) `16×32×8` | 425 (12.32 ms) `16×32×8` |
| cuTile 2pr | 344 (0.48 ms) `32×8×8` | 386 (3.40 ms) `32×32×8` | 387 (13.54 ms) `32×32×8` |
| KA 2pr pm[nb=1] | **433 (0.38 ms)** | 440 (2.97 ms) | 441 (11.87 ms) |
| KA 2pr[nb=1] | 142 (1.16 ms) | 142 (9.26 ms) | 142 (37.03 ms) |
| KA 2pr pm[nb=4] | 339 (0.48 ms) | 340 (3.85 ms) | 340 (15.41 ms) |
| KA 2pr[nb=4] | 273 (0.60 ms) | 274 (4.77 ms) | 274 (19.12 ms) |
| KA 2pr pm[nb=8] | 367 (0.45 ms) | 365 (3.59 ms) | 365 (14.37 ms) |
| KA 2pr[nb=8] | 322 (0.51 ms) | 322 (4.06 ms) | 322 (16.27 ms) |
| KA csr | 94 (1.73 ms) | 95 (13.80 ms) | 95 (55.19 ms) |
| cuSPARSE | 171 (0.96 ms) | 179 (7.32 ms) | 180 (29.19 ms) |
| cuTile 2pr pm t | 390 (0.42 ms) `64×8×8` | 455 (2.88 ms) `16×64×8` | 392 (13.37 ms) `16×64×8` |
| cuTile 2pr t | 350 (0.47 ms) `32×8×8` | 443 (2.96 ms) `16×64×8` | 377 (13.90 ms) `16×64×8` |
| KA 2pr pm[nb=1 t] | 372 (0.44 ms) | 434 (3.01 ms) | 440 (11.92 ms) |
| KA 2pr[nb=1 t] | 340 (0.48 ms) | 424 (3.09 ms) | 432 (12.12 ms) |
| KA 2pr pm[nb=4 t] | 389 (0.42 ms) | 469 (2.79 ms) | 473 (11.07 ms) |
| KA 2pr[nb=4 t] | 337 (0.49 ms) | 456 (2.87 ms) | 470 (11.14 ms) |
| KA 2pr pm[nb=8 t] | 386 (0.42 ms) | **475 (2.76 ms)** | **480 (10.91 ms)** |
| KA 2pr[nb=8 t] | 338 (0.48 ms) | 462 (2.84 ms) | 477 (10.97 ms) |

### Aᵀ·B (arcs×nodes, 2-per-row formats), general α/β (reads C)

| implementation | n=8 | n=64 | n=256 |
|---|---:|---:|---:|
| cuTile csr | 134 (1.22 ms) `16×8×4` | 104 (12.64 ms) `8×64×4` | 104 (50.55 ms) `8×64×4` |
| cuTile 2pr pm | 214 (0.76 ms) `32×8×4` | 160 (8.20 ms) `16×32×8` | 160 (32.73 ms) `16×32×8` |
| cuTile 2pr | 206 (0.79 ms) `32×8×8` | 205 (6.38 ms) `32×32×8` | 205 (25.50 ms) `32×32×8` |
| KA 2pr pm[nb=1] | 248 (0.66 ms) | 256 (5.11 ms) | 256 (20.46 ms) |
| KA 2pr pm[nb=4] | 218 (0.75 ms) | 220 (5.96 ms) | 220 (23.80 ms) |
| KA 2pr pm[nb=8] | 208 (0.79 ms) | 209 (6.25 ms) | 210 (25.01 ms) |
| cuSPARSE | 92 (1.78 ms) | 89 (14.69 ms) | 89 (58.61 ms) |
| cuTile 2pr pm t | **253 (0.65 ms) `64×8×8`** | 268 (4.88 ms) `16×64×8` | 257 (20.40 ms) `16×64×8` |
| cuTile 2pr t | 235 (0.70 ms) `32×8×8` | 265 (4.95 ms) `16×64×8` | 242 (21.61 ms) `16×64×8` |
| KA 2pr pm[nb=1 t] | 253 (0.65 ms) | 281 (4.66 ms) | 281 (18.63 ms) |
| KA 2pr pm[nb=4 t] | 243 (0.67 ms) | **285 (4.59 ms)** | 287 (18.24 ms) |
| KA 2pr pm[nb=8 t] | 250 (0.65 ms) | 285 (4.59 ms) | **288 (18.21 ms)** |


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
| cuTile rc pm | 178 / 180 / 180 | 276 / 342 / 335 |
| cuTile jds pm | 166 / 167 / 167 | 245 / 346 / 340 |
| cuTile rc vals | 157 / 167 / 168 | 236 / 332 / 326 |
| cuTile jds vals | 156 / 162 / 161 | 226 / 339 / 333 |
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
- **The cuTile rc kernel matches JDS once it walks nonzeros the way JDS
  does.** A first version ran two loops (range, then scattered), each to
  the tile's maximum length of that part, while A's rows are sorted by
  *total* length — most gather lanes were masked (176 standard / 230
  transposed). One loop over k = 0 … total−1 that takes `lo + k` below the
  range length and the CSR entry above it (`rc_accumulate`) has JDS's lane
  utilisation with half the id gathers masked off: 180 / 180 standard
  (ahead of jds's 167) and 342 / 335 transposed (jds 346 / 340), at
  `16×64×1`, 8 warps — `tile_k = 1` wins, as the row lengths (avg 5)
  suggest. The tile_m=16 candidates added in follow-up 3 are what lifted
  cuTile jds pm `t` from 298 to 346 (`16×64`, 8 warps).
- **Aᵀ·B side effects of the thread-order fix**: KA 2pr vals nb=8 `t`
  reaches **462 / 477** at n = 64 / 256 — the fastest Aᵀ·B of all, above
  KA 2pr pm nb=1 (441) — each thread reads 8 contiguous floats (a full
  32-byte sector) per B row. cuTile 2pr pm `t` hits 455 at n=64 (`16×64`,
  8 warps) but 392 at n=256; standard layout `16×32` @ 8 warps: 424 / 425.

## Follow-up 5: dropping the "slot empty" checks in 2pr (job 8010)

Every row of Aᵀ has exactly two entries (each arc has a tail and a head),
so the `j > 0` tests in the KA 2pr kernels and the bounds-masked B gather
in the cuTile kernels (id 0 → zero row) are provably never needed. The
`x2` variants drop them: KA via a `Val{EXACT2}` that turns `slot_present`
into `true`; cuTile via `check_bounds=false` on the B gathers (pad rows
past m are clamped to B row 1 and discarded by the masked store; needs
`tile_n | n`). Aᵀ·B, β = 0, GFLOP/s at n = 8 / 64 / 256, best config:

| | checked | `x2` |
|---|---|---|
| KA 2pr pm nb=1 | 434 / 440 / 441 | 436 / 442 / 443 |
| KA 2pr pm nb=4 `t` | 390 / 468 / 473 | 392 / 468 / 473 |
| KA 2pr pm nb=8 `t` | 383 / 474 / **480** | 387 / 443 / 385 |
| KA 2pr vals nb=8 `t` | 337 / 463 / 477 | 340 / 440 / 406 |
| cuTile 2pr pm | 387 / 424 / 424 | 384 / 424 / 424 |
| cuTile 2pr pm `t` | 392 / 455 / 392 | 387 / 457 / 411 |
| cuTile 2pr vals `t` | 348 / 443 / 377 | 349 / 446 / 403 |

- **No gain.** Where the kernels are at their best the difference is
  within ±1%: the checks are warp-uniform branches (KA) or a compare
  folded into the load predicate (cuTile), and the kernels are bound by
  the B-row gathers, not by instruction count. This matches follow-up 3,
  where removing the bounds check (V2) was also worth ~0.
- The one large difference is negative: KA nb=8 `t` loses 20% without the
  branches (480 → 385 at n=256), reproducibly across nb=8 pm and vals.
  The likely cause is instruction scheduling — with the loads
  unconditional the compiler is free to issue both 8-wide row loads back
  to back and the resulting schedule is worse — but I have not confirmed
  that in the SASS. (The `t` cuTile variants gain 5–7% at n=256 but stay
  below KA's 480.)
- Practical conclusion: keep the checks; the exact-2 knowledge is not
  worth a format or a kernel variant. (A default argument
  `::Val{EXACT2}=Val(false)` on the `@kernel` silently cost the checked
  kernels 7× — KA generates a poorly specialized wrapper for it — so the
  flag is passed explicitly.)

## Follow-up 6: a larger instance — vision_rnd_05 — and the hybrid "rch" format (jobs 8012, 8013)

`vision_rnd_05_bone_subx_n6c100_a` (Kovács' VISION-RND family, a max-flow
problem on a 3-D 6-neighbourhood grid turned into min-cost flow): 3.9M
nodes, 23.1M arcs, A = 3,899,394 × 23,091,149 with 46.2M nonzeros — 4.5×
the TX matrix. Exported with `bench/export_flow_matrix.jl`, run with
`SPMM_DATA=vision_rnd_05`; n = 8 and 64 only (at n = 256 the Aᵀ·B bench
needs four 23.6 GB C-sized arrays, more than the L40S has).

The structural difference from the road network: every pixel row of A has
≤ 14 entries (6 in, 6 out, terminal arcs), but the two max-flow terminal
nodes have **92,626 entries between them** (max contiguous range 90,916,
max scattered 1,710), vs a maximum row length of 8 on TX. On a row-parallel
kernel those two rows are the critical path: one thread (KA) or one
16-row program (cuTile) walks 92k entries while everything else finishes.

### The hybrid

"rch" = range + CSR + heavy rows (`split_heavy` / `heavy_chunks` in
`bench/spmm_formats.jl`): rows longer than `SPMM_HEAVY_MAXLEN` (64) are
emptied in the rc arrays — the unchanged rc kernel still writes
α·0 + β·C for them — and collected separately with their range and
scattered ids. Two small passes then handle them with nonzero-level
parallelism: pass 1 splits each heavy row into chunks (KA: 64 entries
per thread, interleaved so consecutive threads read consecutive B rows
of the range; cuTile: 256 contiguous entries per program, `tile_k` = 32
rows of B per step) and writes one partial sum per (chunk, column); pass 2
sums a row's chunks and adds α·Σ into C. The heavy part is 0.2% of the
nonzeros and deterministic (no atomics; the two extra launches are not
timed separately). On TX no row exceeds the cap, so rch degenerates to rc
and is skipped.

### A·B (nodes×arcs), β = 0, GFLOP/s at n = 8 / 64 (job 8012; best config)

| | standard layout | transposed (`t`) |
|---|---|---|
| **KA rch pm** | 242 / 247 (nb=1), 349 / 368 (nb=4) | 423 / **568** (nb=1) |
| **cuTile rch pm** | 390 / 394 `16×8×1`, 8 warps | **427** / 541 `16×8×2` → `16×64×2` |
| KA rch vals | 272 / 282 (nb=4) | 354 / 545 (nb=1) |
| cuTile rch vals | 336 / 358 | 381 / 530 |
| KA rc pm | 160 / 233 (nb=1) | 171 / 488 (nb=1) |
| cuTile rc pm | 110 / 299 | 80 / 283 |
| KA jds pm | 50 / 152 (nb=1) | 23 / 149 |
| cuTile jds pm | 38 / 194 | 21 / 129 |
| cuTile csr | 205 / 164 `1×8×32`, `1×64×64` | – |
| cuSPARSE | 271 / 283 | – |
| KA csr | 39 / 93 | – |

- **Without the split every row-parallel kernel loses to cuSPARSE**, by
  up to 7× at n=8 (KA jds pm 50, cuTile jds pm 38 vs 271). cuSPARSE
  load-balances over nonzeros; our kernels balance over rows. Among the
  unsplit kernels cuTile csr is the least affected (205 at n=8) because
  it consumes `tile_k` = 32 entries of the long row per loop step instead
  of one, and rc is next (its range half is a contiguous strip).
  Interleaving columns per thread makes it *worse* (KA jds nb=8: 16) —
  the serial tail grows with nb.
- **With the split, the picture from TX returns**: KA rch pm `t` **568**
  and cuTile rch pm `t` 541 at n=64 are **2.0× / 1.9× cuSPARSE** (283),
  and at n=8 cuTile rch pm `t` 427 and KA 423 are 1.6×. The light part of
  vision is actually friendlier than TX (regular degree ~12, no short
  rows), so at n=64 both beat their own TX numbers (386 / 342).
- The cuTile heavy kernels are the same tile code as rc with `TILE_M = 1`
  (`heavy_accumulate` reuses `gather_b_wsum`), plus a 30-line reduce
  kernel using a 2-D `gather`/`scatter` on C; the TileTriton emitter
  handled the size-1 tile dimensions and the `(1, TILE_N)` partial store
  without changes.
- Standard layout: cuTile rch pm (394) > KA rch pm nb=4 (368) > cuSPARSE
  (283); KA nb=1 falls to 247 here (on TX it was the best at 386): with
  23M-entry index arrays and a 125 MB B (larger than the 96 MB L2 at n=8)
  one float per thread per B row is too little per gather, and nb=4 or the
  transposed layout is needed.
- α/β path (reads C): KA rch pm nb=4 316 / 331 standard, cuTile rch pm
  297 / 299; cuSPARSE 186 / 190.

### Aᵀ·B (arcs×nodes, exactly 2 per row), β = 0, GFLOP/s at n = 8 / 64 (job 8013)

| | standard layout | transposed (`t`) |
|---|---|---|
| KA 2pr pm | 436 / 435 (nb=8) | 458 / **541** (nb=8) |
| cuTile 2pr pm | 459 / 531 `32×8×4`, `16×32×8` | **468** / 517 |
| KA 2pr vals | 377 / 376 (nb=8) | 396 / 524 (nb=8) |
| cuTile 2pr vals | 408 / 500 | 402 / 505 |
| cuTile csr | 325 / 244 | – |
| cuSPARSE | 190 / 206 | – |
| KA csr | 99 / 99 | – |

- Nothing changes structurally (every row has two entries), and the 2pr
  kernels are 2.5–2.6× cuSPARSE as on TX. Two shifts with size: cuTile 2pr
  pm now **beats KA in the standard layout** (531 vs 435 at n=64 — the
  16×32 tile gathers 32 columns of a B row at once, while KA nb=1 drops
  to 227 for the same per-gather-width reason as above), and the transposed
  layout is worth less to cuTile than on TX (517 vs 531).
- `x2` (follow-up 5) again makes no difference: cuTile 2pr pm x2 459 /
  530 vs 459 / 531; KA nb=8 `t` x2 462 / 518 vs 458 / 541.

## Follow-up 7: collapsing rc's lo/hi into one array — and the row order that allows it (job 8135)

Because every column holds exactly one range entry (each arc has exactly
one head), the rc ranges *partition* the columns: sorted by `lo` they tile
`1:k` and chain — `hi[i] == lo[i+1]` — so the two bounds arrays collapse
into a single `rptr` of length m+1, which is literally the adjacency
matrix's CSC colptr. The chain only holds in **node order**; the exported
A is in JDS length-sorted row order (`construct_constraint_matrix`'s
`sortperm` by length), where it doesn't. `bench/spmm_rptr.jl` therefore
benches the KA rc pm kernel in three variants on TX: the current `lo`/`hi`
on the matrix as exported, `lo`/`hi` with rows permuted to node order (the
control that isolates the row order from the array merge), and the single
`rptr` in node order (`node_order_rptr` in `bench/spmm_formats.jl`; the
36 in-degree-0 rows get `rptr[i] == rptr[i+1]`). β = 0 path; every variant
verified against the (row-permuted) reference.

A·B, β = 0, GFLOP/s at n = 8 / 64 / 256:

| KA rc pm | lo/hi, jds order | lo/hi, node order | rptr, node order |
|---|---|---|---|
| nb=1 | 374 / 380 / 379 | 498 / 507 / 508 | 503 / 508 / **509** |
| nb=4 | 233 / 236 / 236 | 342 / 347 / 347 | 360 / 366 / 366 |
| nb=8 | 155 / 155 / 155 | 365 / 370 / 371 | 374 / 382 / 382 |
| nb=1 `t` | 275 / 355 / 360 | 385 / 496 / 507 | 395 / 499 / 508 |
| nb=4 `t` | 236 / 355 / 359 | 387 / 486 / 490 | 399 / 488 / 491 |
| nb=8 `t` | 226 / 354 / 358 | 360 / 483 / 492 | 369 / 485 / 493 |

(The jds-order column reproduces follow-up 4's baseline within ~2%.)

- **The answer to the question asked: merging lo/hi into rptr is worth
  0–5%.** At the same row order it's +0.2–1% at nb=1 and in the transposed
  layout, +3–5% at nb=4/8 standard. Never slower, and it sheds an
  m-length Int32 array (8.3 MB), so it's free to take — but it is not
  where the time is.
- **The row permutation it requires is worth ~34% by itself** and is the
  real finding: 379 → 508 at nb=1 (both layouts, flat in n), and it
  rescues the previously pathological nb=8 standard shape (155 → 371).
  In node order consecutive threads read consecutive B strips — the
  ranges tile the columns in row order, so the range reads stream through
  B globally instead of jumping per the length sort — and the scattered
  gathers inherit the road network's node locality too. 508 GFLOP/s is
  ~83% of the ~615 A·B roofline from follow-up 4, up from 63%.
- The permutation is legitimate for the LP: C's row order is a free
  choice (permute `b`/supply along, exactly as `construct_constraint_matrix`
  already does for its own length sort). But the length sort exists for
  JDS's benefit — before switching the shared export to node order, the
  JDS kernels (whose iterptr construction assumes it) would need
  re-benching, and cuTile rc + the vision instance haven't been run in
  node order yet.

## Reproducing

```
julia --project=. bench/spmm_flow.jl 8 64 256          # needs bench/data/*.jls
julia --project=. bench/spmm_rptr.jl 8 64 256          # follow-up 7 (rptr / node order)
CUTILE_BACKEND=native julia --project=. bench/spmm_backends.jl 8 64 256  # then triton
python3 bench/spmm_flow_tables.py ~/spmm-flow-<job>.log  # the tables above
```

`bench/data/` (gitignored; ~270 MB for TX, 1.2 GB for vision_rnd_05) is
regenerated from the DIMACS files in `~/min-cost-flow/{road,vision}/` with
`bench/export_flow_matrix.jl <name> <files...>` in an environment that has
MinimumCostFlows (the road a–e files are parts of one instance; the vision
a–e files are cost variants of one graph, use one). It serializes
`SparseMatrixCSC{Float32}` of A and Aᵀ as `<name>{,_t}.jls` and the raw
`JDSMatrixPM`/`Matrix2PerRowPM` arrays as `<name>_zoo.jls`. Select with
`SPMM_DATA=<name>`; `SPMM_MATS=A|At` runs one side only; the hybrid's cap
and chunk sizes are `SPMM_HEAVY_MAXLEN`, `SPMM_HEAVY_CHUNK_KA/CT`.
