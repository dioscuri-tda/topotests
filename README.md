# TopoTests

## Paper
This repo contains a codebase for the paper 

### "Topology-Driven Goodness-of-Fit Tests in Arbitrary Dimensions" 

by Paweł Dłotko, Niklas Hellmer, Łukasz Stettner and Rafał Topolnicki. 

Preprint can be find in [arXiv:2010.14965](https://arxiv.org/pdf/2210.14965.pdf)

## About TopoTests
TopoTests provides a framework for goodness-of-fit (GoF) testing in arbitrary dimensions for one-sample and two-sample problems by adapting Euler Characteristic Curve (ECC) - a tool from computational topology. To the best of our knowledge it is one of the first
attepmts to apply topologically driven approach for goodness of fit testing.
The method is designed to work well with multivariate distributions. 

<img src="images/ecc_example.png" width="800">

## How to use?

### Instalation
Instruction below for work for Linux and MacOSX.

1. Clone the repo
2. (Optional) Create a [python virtual enviroment](https://docs.python.org/3/library/venv.html), activate it and install dependencies
```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Example
Example use and comparison with the Kolmogorov-Smirnov can be found in [OneSampleExample](OneSampleExample.ipynb) and [TwoSampleExample](TwoSampleExample.ipynb) notebooks.

## Results
Simulation study was conducted to address the power of the TopoTest in comparison with
Kolmogorov-Smirnov test. In both, one-sample and two-sample setting, the
TopoTest in many cases yielded better performance than Kolmogorov-Smirnov. 

<img src="images/differential_power.png" width="1000">

Comparison of the power of TopoTest and Kolmogorov-Smirnov one-sample tests for collection of null-alternative distribution pairs in case of univariate, d=1, (left panel) and trivariate, d=3, (right panel) probability distributions. Sample size equals n=250.
In each matrix element a difference between power of TopoTest and Kolmogorov-Smirnov test was given. 
The difference in power was estimated based on K=1000 Monte Carlo realizations. 
The average power (excluding diagonal elements) is

- TopoTest: is 0.832 for d=1 and 0.824 for d=3
- Kolmogorov-Smirnov: 0794 for d=1 and 0.763 for d=3

Please refer to Sections 4 and 5 of [the paper](https://arxiv.org/pdf/2210.14965.pdf) for more results.

## Licence
Code is released under the BSD licence.


