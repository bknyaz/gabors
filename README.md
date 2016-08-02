## 0.30% error on MNIST with 155 Gabor filters

The presented code is NOT a good solution: 
1) it overfits to the test set a lot; 
2) the accuracy is very sensitive to the parameters of filters.
But still, it was quite surprising to obtain such a low test error given the simplicity of the model compared
to other works with similar result (without data augmentation and cropping).

## Model
1. 155 complex valued filters of size 18x18 were heuristically selected from a larger set of about 1000 filters;
2. All images are standardized using batches of 100;
3. All images are then convolved with the filters;
4. Absolute value rectification;
5. Max pooling over 2x2 regions;
6. Feature standardization (one sample at a time);
7. RBF-SVM classification (one-vs-one).

Heuristic selection is the most difficult part and will not be discussed here. 
Just a few rather obvious comments:
- there are about 10^182 possible unique combinations of 155 filters from a set of 1000, 
so there is no (trivial) way to brute force this task;
- there is no (again, trivial) way to find single "strong" (in terms of classification error) and "weak" filters as
could be done using boosting algorithms, so only combinations of filters can be strong or weak.

## Dependencies
- Matlab (tested on R2015b)
- Matconvnet
- LIBSVM
- Several functions from
[Autoconvolution for Unsupervised Feature Learning] (https://github.com/bknyaz/autocnn_unsup)
