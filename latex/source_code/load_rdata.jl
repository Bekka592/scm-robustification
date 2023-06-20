function load_rdata(fileName::String)
    """
        Loads the data from the ".RData" file in "/data" with the given file name.
        A dataset usually consists out of: the data matrix "data", the labels "labs"
            and the dataset identifier "name".
        It is possible, yet not necessary, that more attributes are provided.
        Processes and returns the information as a well-formed DataFrame.
    """
    dataset = load("data/$fileName.RData")["dataset"]
    dataMatrix = transpose(dataset["data"])
    S = DataFrame([[dataMatrix[:,i]...] for i in axes(dataMatrix,2)],:auto)

    if haskey(dataset, "featureID")
        rename!(S, dataset["featureID"], makeunique=true)
        for feature in filter(f -> occursin('_', f), names(S))
            @assert S[!, feature] == S[!, feature[1:findfirst("_", feature).stop-1]]
                "There are features with the same names but different value vectors."
        end
        select!(S, unique(dataset["featureID"])) # discarding duplicate features
    end

    S.label = dataset[haskey(dataset, "oldLabs") ? "oldLabs" : "labs"]
    dropmissing!(S)
    return S
end