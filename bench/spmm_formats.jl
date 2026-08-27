# Host-side builders of the range+CSR ("rc") format and its hybrid split
# with heavy rows ("rch"), shared by bench/spmm_flow.jl and the zoo test.

"""
    range_csr(rowptr, colval, nzval) -> (; lo, hi, inptr, inids, rsign)

Range + CSR of a ±1 incidence matrix given as CSR: per row the contiguous
column range `[lo, hi)` of the entries with sign `rsign` and a CSR
(`inptr`, `inids`) of the others. Tries both signs; errors if neither
gives a contiguous range in every row.
"""
function range_csr(rowptr, colval, nzval)
    Ti = eltype(colval)          # index type of the result arrays
    m = length(rowptr) - 1
    for rsign in (-1, 1)
        lo = ones(Ti, m); hi = ones(Ti, m)
        inptr = Vector{Ti}(undef, m + 1); inptr[1] = 1
        inids = sizehint!(Ti[], length(colval) ÷ 2)
        ok = true
        for i in 1:m
            cnt = 0; first = typemax(Ti); last = Ti(0)
            for p in rowptr[i]:(rowptr[i + 1] - 1)
                if sign(nzval[p]) == rsign
                    cnt += 1; first = min(first, colval[p]); last = max(last, colval[p])
                else
                    push!(inids, colval[p])
                end
            end
            cnt == 0 || cnt == last - first + 1 || (ok = false; break)
            cnt > 0 && (lo[i] = first; hi[i] = last + 1)
            inptr[i + 1] = length(inids) + 1
        end
        ok && return (; lo, hi, inptr, inids, rsign)
    end
    error("range_csr: no sign has contiguous columns in every row")
end

"""
    split_heavy(rc; maxlen) -> (; light, heavy)

Hybrid range+CSR: rows with more than `maxlen` nonzeros are emptied in the
`light` arrays (same layout as `rc`, so the rc kernels run unchanged and
write α·0 + β·C for them) and collected in `heavy`: their row indices,
ranges and scattered ids as a small CSR (`ptr`, `ids`).
"""
function split_heavy(rc; maxlen)
    (; lo, hi, inptr, inids) = rc
    Ti = eltype(inids)
    m = length(lo)
    len = (hi .- lo) .+ diff(inptr)
    llo = copy(lo); lhi = copy(hi)
    linptr = similar(inptr); linptr[1] = 1
    linids = Ti[]
    hrow = Ti[]; hptr = Ti[1]; hids = Ti[]
    for i in 1:m
        rng = inptr[i]:(inptr[i + 1] - 1)
        if len[i] > maxlen
            push!(hrow, i)
            lhi[i] = llo[i]
            append!(hids, view(inids, rng)); push!(hptr, Ti(length(hids) + 1))
        else
            append!(linids, view(inids, rng))
        end
        linptr[i + 1] = length(linids) + 1
    end
    light = (; lo=llo, hi=lhi, inptr=linptr, inids=linids, rc.rsign)
    heavy = (; row=hrow, lo=lo[hrow], hi=hi[hrow], ptr=hptr, ids=hids, len=len[hrow])
    return (; light, heavy)
end

"""
    node_order_rptr(rc, ncols) -> (; perm, rptr, lo, hi, inptr, inids)

Row order in which the ranges of `rc` chain: since every column holds exactly
one range entry, the per-row ranges partition `1:ncols`, so sorted by `lo`
(empty rows first among ties) they tile it and collapse into a single
`rptr[i] = lo[i] = hi[i-1]` array of length m+1 — on the incidence matrix in
node order this is the adjacency matrix's CSC colptr. Returns the permutation,
the collapsed `rptr`, and the permuted rc arrays (`lo`/`hi` kept alongside as
the two-array control in the same order). Errors if the ranges don't tile.
"""
function node_order_rptr(rc, ncols)
    (; lo, hi, inptr, inids) = rc
    Ti = eltype(inids)
    m = length(lo)
    perm = sortperm(eachindex(lo); by=i -> (lo[i], lo[i] == hi[i] ? 0 : 1))
    plo = lo[perm]; phi = hi[perm]
    rptr = Vector{Ti}(undef, m + 1)
    rptr[m + 1] = ncols + 1
    for i in m:-1:1
        rptr[i] = plo[i] == phi[i] ? rptr[i + 1] : plo[i]
    end
    for i in 1:m
        ok = plo[i] == phi[i] ? rptr[i] == rptr[i + 1] :
             rptr[i] == plo[i] && rptr[i + 1] == phi[i]
        ok || error("node_order_rptr: ranges don't tile at row $i: ",
                    "[$(plo[i]),$(phi[i])) vs rptr [$(rptr[i]),$(rptr[i + 1]))")
    end
    len = diff(inptr)
    pinptr = Ti[1; cumsum(len[perm]) .+ 1]
    pinids = similar(inids)
    for (i, r) in enumerate(perm)
        src = inptr[r]:(inptr[r + 1] - 1)
        pinids[pinptr[i]:(pinptr[i + 1] - 1)] = view(inids, src)
    end
    return (; perm, rptr, lo=plo, hi=phi, inptr=pinptr, inids=pinids)
end

"Chunk table of the heavy rows for `chunk` nonzeros per chunk: chunk c belongs
to heavy row crow[c]; row h owns chunks cptr[h]:cptr[h+1]-1."
function heavy_chunks(heavy; chunk)
    Ti = eltype(heavy.ids)
    nch = cld.(heavy.len, chunk)
    cptr = Ti[1; cumsum(nch) .+ 1]
    crow = Ti[h for h in eachindex(nch) for _ in 1:nch[h]]
    return (; crow, cptr)
end

