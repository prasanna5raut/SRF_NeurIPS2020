
# A Single Recipe for Online Submodular Maximization with Adversarial or Stochastic Constraints

This repository contains implementation of the numerical experiments from the \textit{NeurIPS 2020} paper titled as above.

The authors are:
- Omid Sadeghi, Department of Electrical and Computer Engineering, University of Washington, Seattle.
- Prasanna Raut Department of Mechanical Engineering, University of Washington, Seattle.
- Maryam Fazel, Department of Electrical and Computer Engineering, University of Washington, Seattle.

## Requirements

The algorithms have been implemented using Python 3.7 and the easiest way to install all dependencies is to install Anaconda first.

To install Anaconda, visit the official website:

```https://docs.anaconda.com/anaconda/install/```

Once Anaconda is installed, you can now install cvxpy. The following command will install cvxpy:

```conda install -c conda-forge cvxpy```

## Experiment Code Summary

### 1) Online Joke Recommendation
- For Figure 1 (a) in paper
- Data set: jester-data-1.xls
- Code: JokeRecommendationNeurIPS.py

### 2) Online job assignment in crowdsourcing markets
- For Figure 1 (b) in paper
- Code: crowdsourcingNeurIPSFinal.py

### 3) Online welfare maximization with production costs
- For Figure 1 (c) in paper
- Code: LogDetNeurIPSFinal.py

### 4) Pareto optimal Plot
- For Figure 1 (a) in supplement
- Code: paretoPlot.py (It just generates the figure and this code doesn’t run any experiment)

### 5) Dynamic regret benchmarks
- For Figure 1 (b) in supplement
- Code: supplementDynamicRegretNeurIPS.py

## Results

For each of the experiment/plot, we recommend opening a new Jupyter notebook and running the code in this notebook. This allows you to play with the parameters and plot the results easily.