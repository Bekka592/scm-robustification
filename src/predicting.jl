""" Provides a routine to test the fitness of the constructed classification rule on test data. """

function classify_value(value::Union{Float64, Int64, Bool, AbstractString},
        operator::Char, threshold::Union{Float64, Int64, Bool, AbstractString})
    """ As the operator is stored as a Char, this method is needed to make use of it. """
    if operator == '<'
        return value < threshold
    elseif operator == '>'
        return value > threshold
    elseif operator == '≤'
        return value <= threshold
    elseif operator == '≥'
        return value >= threshold
    else # operator == '='
        return value == threshold
    end
end

function classify_sample(conjunction::Vector{BaseClassifier}, sample::DataFrameRow)
    """ Determining the class that the given conjunction would assign to this sample. """
    for baseClassifier in conjunction
        if !classify_value(sample[baseClassifier.featureId], baseClassifier.operator, baseClassifier.threshold)
            return false # all base classifiers need to match
        end
    end
    return true
end

function classify_sample(disjunction::Vector{Vector{BaseClassifier}}, sample::DataFrameRow)
    """ Determining the class that the given dnf would assign to this sample. """
    for conjunction in disjunction
        if classify_sample(conjunction, sample)
            return true # just one match is enough
        end
    end
    return false
end

function predict(model::Vector{Vector{BaseClassifier}}, testData::DataFrame)
    """
        Testing the model on unknown and independent verification data.
        Thus, the model is checked for generalizability and the chance of over-fitting.

        Input:
        model       - Result of the training algorithm.
        testData    - Set of verification samples.

        Output:
        sensitivity of the model    - Fraction of correctly classified class 1 samples.
        specificity of the model    - Fraction of correctly classified class 0 samples.
        accuracy of the model       - Fraction of correctly classified overall samples.
    """
    correct0 = Int32(0)
    correct1 = Int32(0)

    for sample in eachrow(testData)
        class = classify_sample(model, sample)

        if sample.label && class == 1
            correct1 += 1
        elseif !sample.label && class == 0
            correct0 += 1
        end
    end

    negSamples = length(filter(==(false), testData[!,:label]))
    posSamples = length(filter(==(true), testData[!,:label]))
    sensitivity = posSamples == 0 ? 0 : correct1/posSamples
    specificity = negSamples == 0 ? 0 : correct0/negSamples
    accuracy = posSamples + negSamples == 0 ? 0 : (correct0+correct1)/(negSamples+posSamples)
    return (sensitivity, specificity, accuracy)
end