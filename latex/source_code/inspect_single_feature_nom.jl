function inspect_single_feature(featureId::Int, featureVector::NominalFeature, labelVector::BoolVector, p::Float32)
    """
        Comparing all possible thresholds of a nominal feature to find the optimal one.
        This algorithm uses the data of the first n-1 possible thresholds to
            derive the last possible threshold's data.
    """
    reducedFeatureVector = collect(Set(featureVector))
    winnerThreshold = ""
    maxScore = -Inf

    totalAmountOfClass1Samples = sum(labelVector)
    totalAmountOfClass0Samples = length(labelVector) - totalAmountOfClass1Samples
    sumQ = 0
    sumR = 0

    for threshold in reducedFeatureVector
        if threshold == reducedFeatureVector[end]
            score = sumQ - p * sumR
        else
            score, Q, R = calc_score(threshold, '=', featureVector, labelVector, p)
            sumQ += totalAmountOfClass0Samples - Q
            sumR += totalAmountOfClass1Samples - R
        end
        if score > maxScore
            maxScore = score
            winnerThreshold = threshold
        end
    end
    return BaseClassifier(featureId, '=', winnerThreshold, maxScore)
end