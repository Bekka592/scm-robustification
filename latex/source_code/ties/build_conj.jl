selectedBaseClassifiers = copy(preselectedBaseClassifiers)
ties = Vector{Vector{BaseClassifier}}()
while false in S.label && length(selectedBaseClassifiers) < s
    (bc, newTies) = selectedBaseClassifiers == preselectedBaseClassifiers && length(selectedBaseClassifiers) > 0 ? (preselectedBaseClassifiers[end], []) : find_base_classifier(S, p)

    dataPredictions = map(value -> classify_value(value, bc.operator, bc.threshold), S[!, bc.featureId])
    if !issubset([0,2], dataPredictions .+ S.label)
        break
    end

    foreach(tie -> push!(ties, [selectedBaseClassifiers..., tie]), newTies)
    S = S[findall(==(true), dataPredictions), :]
    push!(selectedBaseClassifiers, bc)
end
return (selectedBaseClassifiers, ties)