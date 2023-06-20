function calc_score(threshold::Union{Bool, String}, operator::Char, featureVector::NominalFeature, labelVector::BoolVector, p::Float32)
    """ Calculating the usefulness score of a base classifier by using the basic formula of Marchand and Shawe-Taylor (2002). """
    classifiedAs0 = map(m -> !classify_value(m, operator, threshold), featureVector)
    Q = sum(classifiedAs0 .& .!labelVector)
    R = sum(classifiedAs0 .& labelVector)
    return Q - p * R
end