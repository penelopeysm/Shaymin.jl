rss() = only(filter(startswith("VmRSS"), readlines("/proc/self/status")))

v = Vector{Float64}(undef, 10^8)
@info rss()              # baseline
count(iszero, v)
@info rss()              # after reading: should barely change
fill!(v, 1.0)
@info rss()              # after writing: should jump by about 800 MB (~780,000 kB)
