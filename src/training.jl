"""
    Extended Julia implementation of the "BuildSCM" algorithm from
        "The Set Covering Machine" by Marchand and Shawe-Taylor (2002).
    Supports datasets with numerical (real valued) and/or nominal (raw text) features.
    Extension into a disjunctive normal form (DNF).
"""

function calc_score(threshold::Union{Bool, String}, operator::Char, featureVector::NominalFeature, labelVector::BoolVector, p::Float32)
    """ Calculating the usefulness score of a base classifier by using the basic formula of Marchand and Shawe-Taylor (2002). """
    classifiedAs0 = map(m -> !classify_value(m, operator, threshold), featureVector)
    Q = sum(classifiedAs0 .& .!labelVector)
    R = sum(classifiedAs0 .& labelVector)
    return (Q - p * R), Q, R
end

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
        if threshold == reducedFeatureVector[end] # optimization by one dimension
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

function find_border(featureId::Int, reducedFeatureVector::NumericalFeature, posSamples::Dict{Float64, Int32}, negSamples::Dict{Float64, Int32}, operator::Char, p::Float32)
    """
        Finding the optimal upper or lower border for an numerical feature.
        Using an advanced computation mechanism, without an exclicit call of "calc_core()".
    """
    winnerThreshold = 0
    maxScore = -Inf
    R = 0 # misclassified positive samples
    Q = 0 # correctly classified negative samples
    for threshold in reducedFeatureVector
        score = Q-p*R
        if score > maxScore
            maxScore = score
            winnerThreshold = threshold
        end

        # calc values for next iteration
        R = R + posSamples[threshold]
        Q = Q + negSamples[threshold]
    end
    return BaseClassifier(featureId, operator, winnerThreshold, maxScore)
end

function inspect_single_feature(featureId::Int, featureVector::NumericalFeature, labelVector::BoolVector, p::Float32)
    """ Collecting the optimal lower and upper border of a numerical feature und returning their optimum. """
    reducedFeatureVector = sort!(collect(Set(featureVector))) # removing duplicates and ordering
    posSamples = Dict{Float64, Int32}(reducedFeatureVector .=> 0) # number of pos/neg samples for which xi=n (n is the key)
    negSamples = Dict{Float64, Int32}(reducedFeatureVector .=> 0)
    for (index, value) in enumerate(featureVector)
        dict = labelVector[index] ? posSamples : negSamples
        dict[value] += 1
    end

    lowerBorder = find_border(featureId, reducedFeatureVector, posSamples, negSamples, '≥', p)
    # Reversing the feature vector ensures Q=0 and R=0 at index 0 when using "<=".
    upperBorder = find_border(featureId, reverse(reducedFeatureVector), posSamples, negSamples, '≤', p)
    winnerRay = upperBorder.score > lowerBorder.score ? upperBorder : lowerBorder

    # placing threshold right in between the data points:
    thresholdIndex = findfirst(==(winnerRay.threshold), reducedFeatureVector)
    if winnerRay.operator == '≥' && thresholdIndex > 1
        return BaseClassifier(winnerRay.featureId, '>',
            round((winnerRay.threshold + reducedFeatureVector[thresholdIndex-1]) / 2, digits=2),
            winnerRay.score)
    elseif winnerRay.operator == '≤' && thresholdIndex < length(reducedFeatureVector)
        return BaseClassifier(winnerRay.featureId, '<',
            round((winnerRay.threshold + reducedFeatureVector[thresholdIndex+1]) / 2, digits=2),
            winnerRay.score)
    else
        return winnerRay
    end
end

function find_base_classifier(S::DataFrame, p::Float32)
    """ Compares the base classifiers of all features and returns the optimum. """
    baseClassifiers = Array{BaseClassifier}(undef, ncol(S)-1)
    Threads.@threads for feature in 1:(ncol(S)-1)
        baseClassifiers[feature] = inspect_single_feature(feature, S[!,feature], S.label, p)
    end
    return sort(baseClassifiers, by=(r -> r.score), rev=true)[1]
end

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

        # base classifier needs to classify at least one positive sample as positive and at least one negative sample as negative:
        if !issubset([0,2], dataPredictions .+ S.label)
            break
        end
        # only rows that are classified as 1 by this feature are kept:
        S = S[findall(==(true), dataPredictions), :]

        # needed when recorrecting:
        filter!(bs -> bs.featureId != bc.featureId || bs.operator != bc.operator, selectedBaseClassifiers)
        push!(selectedBaseClassifiers, bc)
    end

    return selectedBaseClassifiers
end

function build_dnf(S::DataFrame, p::Float32, minConjunctionSize::UInt8, sC::UInt8, sD::UInt8)
    """
        Computes a disjunction of up to sD conjunctions.

        Input parameters:
        S  - Set of (up to 2^31 - 1) training samples.
             Those sample each consist of up to 2^31 - 2 features and a label.
        p  - Penalty value for misclassifying a positive sample.
        sC - Max number of features every conjunction is allowed to contain.
        sD - Max number of conjunction the disjunction is allowed to contain.
        minConjunctionSize - Minimum number of positive samples that each conjunction needs to cover.
        
        Output: Vector of conjunction vectors.
    """
    conjunctions = Vector{Vector{BaseClassifier}}()
    while true in S.label && length(conjunctions) < sD
        # Step 1: Create a whole - and maximal big - conjunctive SCM.
        conj = build_conjunction(S, p, sC)
        dataPredictions = map(sample -> classify_sample(conj, sample), eachrow(S))

        # ensure that the conjunction is not null and it covers at least minConjunctionSize
        #   positive samples that have not been covered by any of the previous conjunctions:
        if length(conj) == 0 || sum(length(findall(==(true), dataPredictions .& S.label))) < minConjunctionSize
            break
        end

        # Step 2: Delete all samples (of class 0 OR class 1) that got covered by this SCM.
        S = S[findall(==(false), dataPredictions), :]

        # Step 3: Add the found conjunctive SCM into a disjunction.
        push!(conjunctions, conj)
    end # Step 4: Repeat until all positive samples are covered (or the limit "sD" is reached).
    return conjunctions
end