"""
    Here, the high-level functions "train_on_all_data()" and "cross_validation()"
        are located, together with some helper funtions.
    These access the code of all other classes and therefore link the data laoding, training,
        predicting and plotting routines.
"""

using BenchmarkTools
include("data_types.jl")
include("loading_data.jl")
include("training.jl")
include("predicting.jl")
include("plotting.jl")

function to_string(bc::BaseClassifier, colNames::Vector{String})
    """ Formatting the given baseclassifier as a string, to provide an easy to understand rule. """
    return string(colNames[bc.featureId], " ", bc.operator, " ", bc.threshold)
end

function to_string(conjunction::Vector{BaseClassifier}, colNames::Vector{String})
    """ Joining multiple base classifier strings using AND-symbols. """
    return join(map(bc -> to_string(bc, colNames), conjunction), " AND ")
end

function to_string(disjunction::Vector{Vector{BaseClassifier}}, colNames::Vector{String})
    """ Joining multiple conjunction strings using OR-symbols. """
    return join(map(conjunction -> "($(to_string(conjunction, colNames)))", disjunction), " OR ")
end

function booleanize_labels!(S::DataFrame)
    """
        Enabling efficient vector operations by working on boolean lables.
        The real labels will instead be stored in a "labelTexts" vector with:
            labelTexts[1] = class 0 (false)
            labelTexts[2] = class 1 (true)
    """
    labelTexts = collect(sort(unique(S.label)))
    @assert length(labelTexts) == 2 "The SCM can only process two-class data."
    S.label = map(lab -> lab == labelTexts[1] ? false : true, S.label)
    return string.(labelTexts)
end

function train_on_all_data(S::DataFrame, p::Float32=Float32(2), minConjunctionSize::UInt8=UInt8(1),
        sC::UInt8=typemax(UInt8), sD::UInt8=typemax(UInt8))
    """
        Mainly for testing purposes of the train_dnf() method.
        Creates a classification rule by training on S, outputs this rule, determines its
            fitness values and plots its decision regions.
        Notice that the prediction routine is conducted on the exact same samples, that were used
            for training the model.
        The fitness values are therefore very biased and are not able to give insights on the
            generalization error of the classifier.
    """
    labelTexts = booleanize_labels!(S)
    ruleSet = build_dnf(S, p, minConjunctionSize, sC, sD)
    println("IF ", to_string(ruleSet, names(S)), " THEN class \"", labelTexts[2], "\"")
    fitness = predict(ruleSet, S)
    println("\tsensitivity: ", string(fitness[1]), "\n\tspecificity: ", string(fitness[2]),
        "\n\taccuracy: ", string(fitness[3]))
    # savefig(plot_model_2d(S, ruleSet, false), "2DModel.pdf") # works only on numeric features
end

function calc_pareto_front(models::DataFrame)
    """
        Finding and returning all models, that are not dominated by any other model and
            therefore belong to the first pareto front.
        Taking two dimensions (here: accuracy and noOfUsedFeatures) into account.
        The models will automatically be ordered from optimum(dim1) to optimum(dim2).
    """
    sortedModels = sort(models, [order(:accuracy, rev=true), order(:noOfUsedFeatures, rev=false)])
    paretoFront = sortedModels[1:1, :] # element with max value for first feature
    for row in eachrow(sortedModels)
        if row.noOfUsedFeatures < last(paretoFront).noOfUsedFeatures
            push!(paretoFront, row)
        end
    end
    return paretoFront
end

function evaluate_models(models::DataFrame, labelTexts::Vector{String}, S::DataFrame)
    """ Comparing, evaluating and averaging the results of cross_validation(). """
    noOfClass0Samples = nrow(S)-sum(S.label)
    noOfClass1Samples = sum(S.label)
    baseline = max(noOfClass0Samples, noOfClass1Samples)/(noOfClass0Samples + noOfClass1Samples)

    # evaluating averages
    println("average number of base classifiers per classification rule: ",
        round(mean(models[!,:noOfUsedBaseClassifiers]), digits=2))
    println("average number of features per classification rule: ",
        round(mean(models[!,:noOfUsedFeatures]), digits=2))
    println("average achieved feature compression: ",
        ((1-mean(models[!,:noOfUsedFeatures])/(ncol(S)-1))*100), "%\n")

    println("average accuracy for samples of class \"$(labelTexts[2])\" (sensitivity): ",
        round(mean(models[!,:sensitivity]), digits=3))
    println("average accuracy for samples of class \"$(labelTexts[1])\" (specificity): ",
        round(mean(models[!,:specificity]), digits=3))
    println("average overall accuracy: ", round(mean(models[!,:accuracy]), digits=3))
    println("percentage of models whose accuracies are above the base line: ",
        round((mean(models[!,:accuracy] .> baseline))*100, digits=1), "%\n")
    
    println("average accuracy of only the first conjunction: ",
        round(mean(models[!,:accuracyFirstConj]), digits=3))
    println("average number of conjunctions per classification rule: ",
        round(mean(models[!,:noOfConj]), digits=3))

    # scattering in the accuracy-complexity space
    paretoFront = calc_pareto_front(models)
    savefig((plot_pareto_front(models, paretoFront, baseline)), "paretoFront.pdf")

    println("\nthe models that performed the best in this cross validation are: ")
    foreach(model -> println("$(model.rule)\n\twith a generalization error " *
        "of $(1-model.accuracy) on its test data"), eachrow(paretoFront))
    
    # histograms
    featureFrequency = mapreduce(f -> Dict{String, Int32}(names(S)[f] => 1), mergewith!(+),
        vcat(collect.(models.features)...))
    savefig(plot_histogram(featureFrequency, UInt8(nrow(models)),
        "feature"), "featureHistogram.pdf")

    baseClassifiersForClass0 = models[findall(contains("$(labelTexts[1])"), models.rule), :baseClassifiers]
    if length(baseClassifiersForClass0) > 0&& length(vcat(collect.(baseClassifiersForClass0)...)) > 0
        baseClassifierFrequencyClass0= mapreduce(r -> Dict{String, Int32}(
            to_string(r, names(S)) => 1), mergewith!(+), vcat(collect.(baseClassifiersForClass0)...))
        savefig(plot_histogram(baseClassifierFrequencyClass0, UInt8(length(baseClassifiersForClass0)),
            "base classifier (for classification as \"$(labelTexts[1])\")"), "baseClassifiersClass0Histogram.pdf")
    end

    baseClassifiersForClass1 = models[findall(contains("$(labelTexts[2])"), models.rule), :baseClassifiers]
    if length(baseClassifiersForClass1) > 0 && length(vcat(collect.(baseClassifiersForClass1)...)) > 0
        baseClassifierFrequencyClass1 = mapreduce(r -> Dict{String, Int32}(
            to_string(r, names(S)) => 1), mergewith!(+), vcat(collect.(baseClassifiersForClass1)...))
        savefig(plot_histogram(baseClassifierFrequencyClass1, UInt8(length(baseClassifiersForClass1)),
            "base classifier (for classification as \"$(labelTexts[2])\")"), "baseClassifiersClass1Histogram.pdf")
    end
end

function cross_validation(S::DataFrame, p::Float32=Float32(1), minConjunctionSize::UInt8=UInt8(1),
    sC::UInt8=typemax(UInt8), sD::UInt8=typemax(UInt8), m::UInt8=UInt8(10), n::UInt8=UInt8(10))
    """
        Performing a mxn (usually 10x10) cross-validation.
        Withing each cycle evaluating both, the rule to classify a sample as class 0
            and the rule to classify it as class 1.
        Then using calc_pareto_front() to continue only with the best of the two models.
    """
    labelTexts = booleanize_labels!(S)

    data = Dict("normal" => S, "inverted" => copy(S))
    data["inverted"].label = .!data["inverted"].label
    modelVector = Vector{DataFrame}(undef, m*n)

    Threads.@threads for i in 1:m
        permutation = randperm(nrow(S))
        # creating a vector of n vectors - each consisting out of approximately equal amounts of samples
        chunks = [Vector{Int32}() for _ in 1:n]
        for (index, sampleNo) in enumerate(permutation)
            push!(chunks[mod(index,n)+1],sampleNo)
        end

        Threads.@threads for j in 1:n
            potentialModels = DataFrame(rule = String[], baseClassifiers = Set[], features=Set[],
            noOfUsedBaseClassifiers=Int16[], noOfUsedFeatures=Int16[], noOfConj=Int16[],
            accuracyFirstConj=Float32[], sensitivity=Float32[], specificity=Float32[], accuracy=Float32[])
            for labelDirection in keys(data) # one with normal and one with inverted (flipped) labels
                p_ = labelDirection == "inverted" ? 1/p : p
                model = build_dnf(data[labelDirection][Not(chunks[j]),:], p_, minConjunctionSize, sC, sD)
                fitness = predict(model, data[labelDirection][chunks[j],:])

                if labelDirection == "inverted"
                    fitness = (fitness[2], fitness[1], fitness[3])
                end

                usedBaseClassifiers = vcat(model...)
                usedFeatures = map(bc -> bc.featureId, usedBaseClassifiers)
                ruleAsString = "IF " * to_string(model, names(S)) * " THEN class \"" *
                    (labelDirection == "normal" ? labelTexts[2] : labelTexts[1]) * "\""
                if length(model) > 0 
                    accFirstConjunction = predict([model[1]], data[labelDirection][chunks[j],:])[3]
                else 
                    accFirstConjunction = fitness[3]
                end

                push!(potentialModels, [ruleAsString, Set(usedBaseClassifiers), Set(usedFeatures),
                    length(usedFeatures), length(Set(usedFeatures)), length(model),
                    accFirstConjunction, fitness...])
            end
            # selecting the best or both models, if both are on the pareto front
            modelVector[m*(i-1)+j] = calc_pareto_front(potentialModels)
        end
    end
    evaluate_models(vcat(modelVector...), labelTexts, S)
end

# cross_validation(load_csv("artificial/float/2-dim/cross"), Float32(2))
# cross_validation(load_csv("uci/chess"), Float32(1))
cross_validation(load_csv("uci/german"), Float32(0.25), UInt8(3))
# @btime train_on_all_data(load_csv("artificial/mixed"), Float32(1))
# @btime train_on_all_data(load_rdata("gene-expressions/TCGA_KICH_vs_KIRC"), Float32(1))
# @btime train_on_all_data(load_csv("artificial/float/50k_features"))

# train_on_all_data(load_csv("artificial/float/one_decision_region"), Float32(3))
# train_on_all_data(load_csv("artificial/float/cross"), Float32(1.5))
# train_on_all_data(load_csv("artificial/float/two_decision_regions"), Float32(1.2))
# train_on_all_data(load_csv("artificial/float/complex_patterns"), Float32(2), UInt8(1))
# train_on_all_data(load_csv("artificial/float/small"), Float32(0.1))
# train_on_all_data(load_random_data(50,1000), Float32(1.5))