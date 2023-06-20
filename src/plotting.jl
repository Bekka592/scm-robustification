""" Helper class to create 2D plots of various computation and analysis results. """

using PlotlyJS

function get_data_points(data::DataFrame)
    """ Painting the positive and neagtive samples as points in the 2D space. """
    posData = filter(:label => ==(true), data)
    negData = filter(:label => ==(false), data)
    posDataPoints = scatter(x=posData[!, 1], y=posData[!, 2],
        mode="markers", marker_color="darkgreen", name="positive samples")
    negDataPoints = scatter(x=negData[!, 1], y=negData[!, 2],
        mode="markers", marker_color="darkorange", name="negative samples")
    return [posDataPoints, negDataPoints]
end

function get_layout(decisionRegions::Vector{Shape}, colNames::Vector{String})
    """ Multiple dispatch in case there are many decision regions. """
    return Layout(
        legend=attr(x=1, y=1, yanchor="bottom", xanchor="right", orientation="h"),
        xaxis_title = colNames[1],
        yaxis_title = colNames[2],
        autosize = false,
        plot_bgcolor = "gainsboro",
        shapes = decisionRegions
    )
end

function get_layout(decisionRegion::Shape, colNames::Vector{String})
    """ Multiple dispatch in case there is only one decision region. """
    return get_layout([decisionRegion], colNames)
end

function get_decision_region(conjunction::Vector{BaseClassifier}, dataBorders::Tuple, isNegation::Bool, gradient::UInt8=UInt8(1))
    """ Constructs a decision region with the given base classifiers as its borders. """
    minXList = filter(bc -> bc.featureId == 1 && (bc.operator in ['≥', '>']), conjunction)
    minX = isempty(minXList) ? dataBorders[1] : minXList[1].threshold
    minYList = filter(bc -> bc.featureId == 2 && (bc.operator in ['≥', '>']), conjunction)
    minY = isempty(minYList) ? dataBorders[2] : minYList[1].threshold
    maxXList = filter(bc -> bc.featureId == 1 && (bc.operator in ['≤', '<']), conjunction)
    maxX = isempty(maxXList) ? dataBorders[3] : maxXList[1].threshold
    maxYList = filter(bc -> bc.featureId == 2 && (bc.operator in ['≤', '<']), conjunction)
    maxY = isempty(maxYList) ? dataBorders[4] : maxYList[1].threshold

    color = isNegation ? "darkorange" : "darkgreen"
    return rect(xref="x", yref="y", opacity=0.4-gradient*0.04, fillcolor = color, line_color = color, line_width = 5,
        x0=minX*0.999, y0=minY*0.999, x1=maxX*1.001, y1=maxY*1.001) # so plotting is possible, even if "xi <= a AND xi >= a"
end

function get_decision_region(disjunction::Vector{Vector{BaseClassifier}}, dataBorders::Tuple, isNegation::Bool)
    """ Gathers a decision region for every conjunction in the disjunction. """
    return map(((index, c),) -> get_decision_region(c, dataBorders, isNegation, UInt8(index)), enumerate(disjunction))
end

function plot_model_2d(data::DataFrame, model::Vector{Vector{BaseClassifier}}, isNegation::Bool)
    """
        Plotting the given data in the first two feature dimensions.
        Those features need to be numerically.
        Additionally plots the areas that are covered by the trained model.
        Can be used with whole DNFs, as well as with single conjunctions.
    """
    dataBorders = (minimum(data[!, 1]), minimum(data[!, 2]), maximum(data[!, 1]), maximum(data[!, 2]))
    decisionRegions = get_decision_region(model, dataBorders, isNegation)
    return plot(get_data_points(data), get_layout(decisionRegions, names(data)))
end

function plot_histogram(frequency::Dict, noOfModels::UInt8, title::String)
    """
        Plot with measurements in percent about how many perfent of models contain feature/ baseclassifier x.
        Taking all features/ base classifiers into account, that occur in at least 20% of the models.
    """
    frequency = sort(collect(frequency), by = x->x[2], rev = true) # sort features by frequency
    statistics = DataFrame(element = String[], frequency = Float32[])
    foreach(((element, score),) -> score >= noOfModels/5
        && push!(statistics, [element, score/noOfModels]), frequency)

    return plot(statistics, x=:element, y=:frequency, kind="bar",
        marker=attr(color="cadet_blue", line_color="black", line_width=1, opacity=0.75),
        Layout(xaxis_tickangle=-45, yaxis_tickformat=",.0%",
            plot_bgcolor = "gainsboro", xaxis_title=title))
end

function plot_pareto_front(allModels::DataFrame, paretoModels::DataFrame, baseline::Float64)
    """
        Creating a two dimensional accurracy-compactness plot.
        This plot contains all models, the pareto optimal models and the accuracy baseline.
    """
    return plot([
        scatter(x=allModels[!,:noOfUsedFeatures], y=allModels[!,:accuracy],
            mode="markers", marker_color="slategray", marker_size=10, name="other models"),
        scatter(x=paretoModels.noOfUsedFeatures, y=paretoModels.accuracy,
            mode="markers", marker_color="indigo", marker_size=10, name="pareto optimal models"),
        scatter(y=[baseline,baseline], x=[minimum(allModels[!,:noOfUsedFeatures])-0.3,
            maximum(allModels[!,:noOfUsedFeatures])+0.3],
            mode="lines", marker_color="seagreen", line_dash="dash", name="baseline")
    ],
    Layout(
        legend = attr(x=1, y=1, yanchor="bottom", xanchor="right", orientation="h"),
        xaxis_title = "noOfUsedFeatures",
        yaxis_title = "accuracy",
        xaxis = attr(dtick=1),
        yaxis_range = [0,1.05],
        autosize = false,
        plot_bgcolor = "gainsboro"
    ))
end