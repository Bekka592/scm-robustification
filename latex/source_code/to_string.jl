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