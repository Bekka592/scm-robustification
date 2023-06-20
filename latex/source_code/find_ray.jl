function find_base_classifier(S::DataFrame, p::Float32)
    """ Compares the base classifiers of all features and returns the optimum. """
    baseClassifiers = Array{BaseClassifier}(undef, ncol(S)-1)
    Threads.@threads for feature in 1:(ncol(S)-1)
        baseClassifiers[feature] = inspect_single_feature(feature, S[!,feature], S.label, p)
    end
    return sort(baseClassifiers, by=(r -> r.score), rev=true)[1]
end