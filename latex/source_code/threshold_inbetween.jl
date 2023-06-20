thresholdIndex = findfirst(==(winner.threshold), reducedFeatureVector)
if winner.operator == '≥' && thresholdIndex > 1
    return BaseClassifier(winner.featureId, '>', round((winner.threshold + reducedFeatureVector[thresholdIndex-1]) / 2, digits=2), winner.score)
elseif winner.operator == '≤' && thresholdIndex < length(reducedFeatureVector)
    return BaseClassifier(winner.featureId, '<', round((winner.threshold + reducedFeatureVector[thresholdIndex+1]) / 2, digits=2), winner.score)
else
    return winner
end