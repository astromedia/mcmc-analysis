# MCMC analysis #

- - - -

Code to perform MCMC sampling of the posterior probability distribution of the model parameter space, given a generic model function. It is prepared to work with large datasets, complex modeling, or very long runs.

* The code allows to start, stop, or load existing chains.

* Statistical chain analysis and plotting functions are included in the notebook.

* Some mock data (with outliers) is provided to test the MCMC analysis code, as well as a mock data generator.

- - - -

To do list:

- [ ] Include in the likelihood calculation any possible covariance in the data.
- [ ] Include some code to show the mean loglikelihood evolution with each step.
- [ ] Include option to select between different loglikelihood functions.
- [ ] Increase the number of prior functions.
- [ ] Generalize the plotting code to work with more than one input value (x-axis).

- - - -

Based on "Data analysis recipes: Fitting a model to data" by Hogg et al.: https://arxiv.org/abs/1008.4686

* MCMC sampling functionality comes from the the `emcee` MCMC sampler by Dan Foreman-Mackey et al.: https://arxiv.org/abs/1202.3665.

* Makes use of the plotting utilities of the `getdist` package by Antony Lewis: https://getdist.readthedocs.io/en/latest/.