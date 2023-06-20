for (index, column) in enumerate(eachcol(S))
    if supertype(typeof(column[1])) in [InlineString, AbstractString]
       S[!, index] = convert.(String, column)
    end
end