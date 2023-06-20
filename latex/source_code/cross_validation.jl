function cross_validation(S::DataFrame, p::Float32=Float32(1), minConjSize::UInt8=UInt8(1), sC::UInt8=typemax(UInt8), sD::UInt8=typemax(UInt8), m::UInt8=UInt8(10), n::UInt8=UInt8(10))
    ...
    data = Dict("normal" => S, "inverted" => copy(S))
    data["inverted"].label = .!data["inverted"].label
    modelVector = Vector{DataFrame}(undef, m*n)

    Threads.@threads for i in 1:m
        permutation = randperm(nrow(S))
        chunks = [Vector{Int32}() for _ in 1:n]
        for (index, sampleNo) in enumerate(permutation)
            push!(chunks[mod(index,n)+1],sampleNo)
        end

        Threads.@threads for j in 1:n
            ...
            for labelDirection in keys(data)
                ...
                model = build_dnf(data[labelDirection][Not(chunks[j]),:], p_, minConjSize, sC, sD)
                fitness = predict(model, data[labelDirection][chunks[j],:])
                ...
            end
            ...
        end
    end
    ...
end