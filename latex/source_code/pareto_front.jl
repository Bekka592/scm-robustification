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