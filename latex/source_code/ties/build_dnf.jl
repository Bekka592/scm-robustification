# not using ties
conjunctions = Vector{Vector{BaseClassifier}}()
...
conj = build_conjunction(S, p, sC)

# using ties
conjunctions = Vector{Vector{BaseClassifier}}()
nextConjunctions = Vector{Vector{BaseClassifier}}()
...
if length(nextConjunctions) > 0
    (conj, ties) = build_conjunction(S, p, sC, popfirst!(nextConjunctions))
else
    (conj, ties) = build_conjunction(S, p, sC)
end
foreach(tie -> push!(nextConjunctions, tie), ties)