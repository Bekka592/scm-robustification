for rule in selectedBaseClassifiers
    if rule.feature == feature && rule.operator == operator
        return STC(0, operator, 0, typemin(Float64))
    end
end