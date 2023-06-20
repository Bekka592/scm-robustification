#### Bachelor's Thesis on "Robustifying the Set Covering Machine with Disjunctive Normal Forms and Nominal Features"

Extending the set covering machine, a classification algorithm by Marchand and Shawe-Taylor, to improve its accessability and performance on various kinds of data sets, such as data with disjunct behavior or nominal features. This was done by extending the decision rules into disjunctions of multiple conjunctions and allowing the data sets to contain both, numerical and categorical, features simultaneously. The new algorithms were implemented in [Julia](https://julialang.org/) and evaluated on artificial and real-world data using cross-validation and, to a certain extend, pareto fronts. Due to copyrights, the real-world data sets are not provided in this repository, but their raw data can be found in the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php) and [The Cancer Genome Atlas](https://portal.gdc.cancer.gov/).

Executing the Julia code:
1. Open a multithreaded Julia REPL session by running ```julia --threads auto```
2. Run ```include("src/main.jl")```
3. Run a query like for example ```cross_validation(load_csv("artificial/float/cross"))```

[![CC BY 4.0][cc-by-shield]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg