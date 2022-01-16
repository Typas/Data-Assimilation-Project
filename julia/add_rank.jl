using DelimitedFiles;
using LinearAlgebra;

full = readdlm("data/y_o_full.txt");
sync = readdlm("data/y_o_sync.txt");
async = readdlm("data/y_o_async.txt");

ϵ = 1e-6;
ilist = 3:2:41;
tlist = full[ilist, 1];
is = 1;
ia = 1;
for i = ilist
    while abs(sync[is, 1] - tlist[i÷2]) > ϵ
        global is += 1
    end
    while abs(sync[ia, 1] - tlist[i÷2]) > ϵ
        global ia += 1
    end
    sync[is, i] = full[i, i];
    async[ia, i] = full[i, i];
end

ef = Vector{Bool}(undef, 41)
es = Vector{Bool}(undef, 41)
ea = Vector{Bool}(undef, 41)

for i = 1:41
    ef[i] = any(!isnan, full[:, i])
    es[i] = any(!isnan, sync[:, i])
    ea[i] = any(!isnan, async[:, i])
end

@show all(ef)
@show all(es)
@show all(ea)


open("data/y_o_sync_add.txt", "w") do ysa
    writedlm(ysa, sync)
end

open("data/y_o_async_add.txt", "w") do yaa
    writedlm(yaa, async)
end
