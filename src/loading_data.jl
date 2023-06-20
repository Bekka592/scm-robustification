"""
    Helper class that provides functions for loading different kinds of data.
    Mainly RData and CSV files are supported.
"""

using DataFrames, CSV, RData, RDatasets, Random

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

function load_csv(fileName::String)
    """
        Loads the data from the ".CSV" file in "/data" with the given file name.
        Processes and returns the information as a well-formed DataFrame.
    """
    S = CSV.read(joinpath(pwd(), "data/$fileName.csv"), DataFrame)

    # parsing fixed width strings into regular strings
    for (index, column) in enumerate(eachcol(S))
        if supertype(typeof(column[1])) in [InlineString, AbstractString]
           S[!, index] = convert.(String, column)
        end
    end

    dropmissing!(S)
    return S
end

function load_iris()
    """
        Loads and processes the R Data Set "iris" from the standard library of R.
        Outputs the processed data as a DataFrame, that can now be used for general testing.
    """
    S = RDatasets.dataset("datasets", "iris")
    rename!(S,:Species => :label)
    S.label = convert.(String,S.label)
    filter!(row -> row.label != "setosa", S)
    return S
end

function load_random_data(rows::Int, columns::Int)
    """
        Generates a matrix with x rows and y columns of random float values.
        Additonally a random boolean label column is added.
        Outputs the generated data as a DataFrame, that can now be used for general testing.
    """
    S = DataFrame(rand(Float64, rows, columns),:auto)
    S.label = rand((0, 1), nrow(S))
    return S
end