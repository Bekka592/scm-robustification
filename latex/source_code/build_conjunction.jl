function build_conjunction(S::DataFrame, p::Float32, s::UInt8)
    """
        Computes a conjunction of up to s base classifiers.

        Input parameters:
        S  - Set of (up to 2^31 - 1) training samples.
            Those sample each consist of up to 2^31 - 2 features and a label.
        p  - Penalty value for misclassifying a positive sample.
        s  - Max number of features this conjunction is allowed to contain.
        
        Output: Vector of base classifiers.
    """
    selectedBaseClassifiers = Vector{BaseClassifier}()
    while false in S.label && length(selectedBaseClassifiers) < s
        bc = find_base_classifier(S, p)
        dataPredictions = map(value -> classify_value(value, bc.operator, bc.threshold), S[!, bc.featureId])

        if !issubset([0,2], dataPredictions .+ S.label)
            break
        end
        S = S[findall(==(true), dataPredictions), :]

        push!(selectedBaseClassifiers, bc)
    end

    return selectedBaseClassifiers
end