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
    m = length(rowptr) - 1
    for rsign in (-1, 1)
        lo = ones(Int32, m); hi = ones(Int32, m)
        inptr = Vector{Int32}(undef, m + 1); inptr[1] = 1
        inids = sizehint!(Int32[], length(colval) ÷ 2)
        ok = true
        for i in 1:m
            cnt = 0; first = typemax(Int32); last = Int32(0)
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
    m = length(lo)
    len = (hi .- lo) .+ diff(inptr)
    llo = copy(lo); lhi = copy(hi)
    linptr = similar(inptr); linptr[1] = 1
    linids = Int32[]
    hrow = Int32[]; hptr = Int32[1]; hids = Int32[]
    for i in 1:m
        rng = inptr[i]:(inptr[i + 1] - 1)
        if len[i] > maxlen
            push!(hrow, i)
            lhi[i] = llo[i]
            append!(hids, view(inids, rng)); push!(hptr, Int32(length(hids) + 1))
        else
            append!(linids, view(inids, rng))
        end
        linptr[i + 1] = length(linids) + 1
    end
    light = (; lo=llo, hi=lhi, inptr=linptr, inids=linids, rc.rsign)
    heavy = (; row=hrow, lo=lo[hrow], hi=hi[hrow], ptr=hptr, ids=hids, len=len[hrow])
    return (; light, heavy)
end

"Chunk table of the heavy rows for `chunk` nonzeros per chunk: chunk c belongs
to heavy row crow[c]; row h owns chunks cptr[h]:cptr[h+1]-1."
function heavy_chunks(heavy; chunk)
    nch = cld.(heavy.len, chunk)
    cptr = Int32[1; cumsum(nch) .+ 1]
    crow = Int32[h for h in eachindex(nch) for _ in 1:nch[h]]
    return (; crow, cptr)
end

