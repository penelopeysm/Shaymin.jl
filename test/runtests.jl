s = raw"""
a
b\
c
d
"""
@show collect(codeunits(s))
