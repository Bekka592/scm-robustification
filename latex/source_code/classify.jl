function classify_sample(conjunction::Vector{BaseClassifier}, sample::DataFrameRow)
    """ Determining the class that the given conjunction would assign to this sample. """
    for bc in conjunction
        if !classify_value(sample[bc.featureId], bc.operator, bc.threshold)
            return false # all base classifiers need to match
        end
    end
    return true
end

function classify_sample(disjunction::Vector{Vector{BaseClassifier}}, sample::DataFrameRow)
    """ Determining the class that the given DNF would assign to this sample. """
    for conjunction in disjunction
        if classify_sample(conjunction, sample)
            return true # just one match is enough
        end
    end
    return false
end