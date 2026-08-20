#!/usr/bin/env python3
"""Markdown tables from a bench/spmm_flow.jl log.

    python3 bench/spmm_flow_tables.py ~/spmm-flow-7980.log

One table per matrix (A, Aᵀ) with one row per implementation variant and one
column per n; cuTile entries show the best tile config. β=0 timings only; the
α/β rows of the log are listed in a second pair of tables.
"""
import re, sys
from collections import OrderedDict

def parse(path):
    rows = OrderedDict()           # (mat, impl) -> {n: (gflops, us, cfg)}
    ns = []
    for line in open(path):
        if not line.startswith("RESULT"):
            continue
        parts = line.rstrip("\n").split("\t")
        m = re.match(r"flow (\w+) n=(\d+)", parts[1])
        mat, n = m.group(1), int(m.group(2))
        n in ns or ns.append(n)
        impl = parts[2]
        if parts[3] == "FAILED":
            rows.setdefault((mat, impl), {})[n] = None
            continue
        us = float(parts[3].split()[0]); g = float(parts[4].split()[0])
        cfg = re.search(r"\[(\d+(?:×\d+)*)\]", impl)
        fam = re.sub(r"\[\d+(?:×\d+)*\]", "", impl).replace(" best", "").strip()
        cur = rows.setdefault((mat, fam), {}).get(n)
        if cur is None or g > cur[0]:
            rows[(mat, fam)][n] = (g, us, cfg.group(1) if cfg else "")
    return rows, sorted(ns)

def table(rows, ns, mat, ab):
    out = [f"| implementation | " + " | ".join(f"n={n}" for n in ns) + " |",
           "|---|" + "---:|" * len(ns)]
    for (m, fam), vals in rows.items():
        if m != mat or (("αβ" in fam) != ab):
            continue
        cells = []
        for n in ns:
            v = vals.get(n)
            if v is None:
                cells.append("–" if n not in vals else "FAILED")
            else:
                g, us, cfg = v
                cells.append(f"**{g:.0f}**" if False else
                             f"{g:.0f} ({us/1000:.2f} ms)" + (f" `{cfg}`" if cfg else ""))
        out.append(f"| {fam.replace(' αβ', '')} | " + " | ".join(cells) + " |")
    # bold the best per column
    best = {}
    for (m, fam), vals in rows.items():
        if m != mat or (("αβ" in fam) != ab):
            continue
        for n, v in vals.items():
            if v and (n not in best or v[0] > best[n][0]):
                best[n] = (v[0], fam)
    for i, line in enumerate(out[2:], start=2):
        fam = line.split("|")[1].strip()
        cells = line.split("|")[2:-1]
        for j, n in enumerate(ns):
            if n in best and best[n][1].replace(" αβ", "") == fam:
                cells[j] = " **" + cells[j].strip() + "** "
        out[i] = f"| {fam} |" + "|".join(cells) + "|"
    return "\n".join(out)

if __name__ == "__main__":
    rows, ns = parse(sys.argv[1])
    for mat, title in (("A", "A·B (nodes×arcs, JDS formats)"),
                       ("At", "Aᵀ·B (arcs×nodes, 2-per-row formats)")):
        print(f"### {title}, β = 0\n")
        print(table(rows, ns, mat, False) + "\n")
        print(f"### {title}, general α/β (reads C)\n")
        print(table(rows, ns, mat, True) + "\n")
