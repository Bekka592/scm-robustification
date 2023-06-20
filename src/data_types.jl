""" Storing the needed data structures seperate from the computational logic. """

struct BaseClassifier
    featureId::Int32
    operator::Char
    threshold::Union{Float64, Int64, Bool, AbstractString}
    score::Float64
end

const NumericalFeature = Union{Vector{Float64}, Vector{Int64}}
const NominalFeature = Union{Vector{Bool}, Vector{String}}

const BoolVector = Union{BitVector, Vector{Bool}}