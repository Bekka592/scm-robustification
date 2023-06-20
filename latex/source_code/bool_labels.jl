function booleanize_labels!(S::DataFrame)
    """
        Enabling efficient vector operations by working on boolean lables.
        The real labels will instead be stored in a "labelTexts" vector with:
            labelTexts[1] = class 0 (false)
            labelTexts[2] = class 1 (true)
    """
    labelTexts = collect(sort(unique(S.label)))
    @assert length(labelTexts) == 2 "The SCM can only process two-class data."
    S.label = map(lab -> lab == labelTexts[1] ? false : true, S.label)
    return string.(labelTexts)
end