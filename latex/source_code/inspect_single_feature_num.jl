function inspect_single_feature(featureId::Int, featureVector::NumericalFeature, labelVector::BoolVector, p::Float32)
    """ Collecting the optimal lower and upper border of a numerical feature und returning their optimum. """
    reducedFeatureVector = sort!(collect(Set(featureVector)))
    posSamples = Dict{Float64, Int32}(reducedFeatureVector .=> 0)
    negSamples = Dict{Float64, Int32}(reducedFeatureVector .=> 0)
    for (index, value) in enumerate(featureVector)
        dict = labelVector[index] ? posSamples : negSamples
        dict[value] += 1
    end

    lowerBorder = find_border(featureId, reducedFeatureVector, posSamples, negSamples, '≥', p)
    upperBorder = find_border(featureId, reverse(reducedFeatureVector), posSamples, negSamples, '≤', p)
    return upperBorder.score > lowerBorder.score ? upperBorder : lowerBorder
end