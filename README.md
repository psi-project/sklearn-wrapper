# PSI Wrapper Service for scikit-learn

The [_Protocols and Structures for Inference_](http://psi.cecs.anu.edu.au) project aims to develop a general purpose web API for machine learning. This small [web.py](http://webpy.org/) service provides a mechanism for training and obtaining predictions with [scikit-learn](http://scikit-learn.org/stable/) as well as an implementation of a simple ranking algorithm based on logistic regression. _It is not a PSI service_, in that its API does not follow the [PSI specification](http://psi.cecs.anu.edu.au/spec), but it can be used by [an implementation of that specification](https://github.com/psi-project/server) to interact with scikit-learn.

Further documentation may be provided in the future.
