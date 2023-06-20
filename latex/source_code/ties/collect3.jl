# not collecting ties
baseClassifiers = Array{BaseClassifier}(undef, ncol(S)-1)
Threads.@threads for feature in 1:(ncol(S)-1)
    baseClassifiers[feature] = inspect_single_feature(feature, S[!,feature], S.label, p)
end
return sort(baseClassifiers, by=(r -> r.score), rev=true)[1]

# collecting ties
baseClassifiers = Array{Array{BaseClassifier}}(undef, ncol(S)-1)
Threads.@threads for feature in 1:(ncol(S)-1)
    baseClassifiers[feature] = inspect_single_feature(feature, S[!,feature], S.label, p)
end

winners = vcat(baseClassifiers...)
maxScore = maximum(map(bc -> bc.score, winners))
filter!(bc -> bc.score == maxScore, winners)
return (winners[1], winners[2:end])