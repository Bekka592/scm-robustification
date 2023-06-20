function find_border(featureId::Int, reducedFeatureVector::NumericalFeature, posSamples::Dict{Float64, Int32}, negSamples::Dict{Float64, Int32}, operator::Char, p::Float32)
    """
        Finding the optimal upper or lower border for an numerical feature.
        Using an advanced computation mechanism, without an exclicit call of "calc_core()".
    """
    winnerThreshold = 0
    maxScore = -Inf
    R = 0
    Q = 0
    for threshold in reducedFeatureVector
        score = Q-p*R
        if score > maxScore
            maxScore = score
            winnerThreshold = threshold
        end

        R = R + posSamples[threshold]
        Q = Q + negSamples[threshold]
    end
    return BaseClassifier(featureId, operator, winnerThreshold, maxScore)
end