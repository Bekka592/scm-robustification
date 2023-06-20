function build_dnf(S::DataFrame, p::Float32, minConjSize::UInt8, sC::UInt8, sD::UInt8)
    """
        Computes a disjunction of up to sD conjunctions.

        Input parameters:
        S  - Set of (up to 2^31 - 1) training samples.
             Those sample each consist of up to 2^31 - 2 features and a label.
        p  - Penalty value for misclassifying a positive sample.
        sC - Max number of features every conjunction is allowed to contain.
        sD - Max number of conjunction the disjunction is allowed to contain.
        minConjSize - Minimum number of positive samples that each conjunction needs to cover.
        
        Output: Vector of conjunction vectors.
    """
    conjunctions = Vector{Vector{BaseClassifier}}()
    while true in S.label && length(conjunctions) < sD
        conj = build_conjunction(S, p, sC)
        dataPredictions = map(sample -> classify_sample(conj, sample), eachrow(S))

        if length(conj) == 0 || sum(length(findall(==(true), dataPredictions .& S.label))) < minConjSize
            break
        end

        S = S[findall(==(false), dataPredictions), :]
        push!(conjunctions, conj)
    end
    return conjunctions
end