# not collecting ties
lowerBorder = find_border(featureId, reducedFeatureVector, posSamples, negSamples, '≥', p)
upperBorder = find_border(featureId, reverse(reducedFeatureVector), posSamples, negSamples, '≤', p)
winnerRay = upperBorder.score > lowerBorder.score ? upperBorder : lowerBorder

# collecting ties
winnerRays = Vector{Ray}()
push!(winnerRays, find_border(featureId, reducedFeatureVector, posSamples, negSamples, '≥', p)...)
push!(winnerRays, find_border(featureId, reverse(reducedFeatureVector), posSamples, negSamples, '≤', p)...)
maxScore = maximum(map(ray -> ray.score, winnerRays))
filter!(ray -> ray.score == maxScore, winnerRays)
