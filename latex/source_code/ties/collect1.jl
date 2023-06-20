# not collecting ties
if score > maxScore
    maxScore = score
    winnerThreshold = threshold
end

# collecting ties
if score > maxScore
    maxScore = score
    winnerThresholds = [threshold]
elseif score == maxScore
    push!(winnerThresholds, threshold)
end