\[ \text{Error} = \text{Bias}^2 + \text{irreducible error} \]

| Learner                                                                       | Strengths                                  | Notes                                                 |
| ----------------------------------------------------------------------------- | ------------------------------------------ | ----------------------------------------------------- |
| 🌳 **Decision Trees**                                                         | High variance (perfect for bagging)        | Most common in Random Forests and Gradient Boosting   |
| ➕ **Linear Classifiers** (e.g. logistic regression, SVMs with linear kernels) | Low variance, interpretable                | Often used in stacking or boosting                    |
| 🎛️ **Naive Bayes**                                                           | Simple, fast, high bias                    | Useful in ensembles to add diversity                  |
| 🧠 **Neural Nets** (shallow)                                                  | Flexible, can be weak or strong            | Rare in bagging but used in stacking or hybrid models |
| 📐 **k-NN**                                                                   | Simple, nonlinear, local                   | Can diversify the ensemble                            |
| 🌐 **SVMs**                                                                   | Robust, good decision boundaries           | Often computationally heavier in ensembles            |
| 📦 **Random Subspaces or PCA-reduced models**                                 | Learners trained on different feature sets | Great for increasing independence                     |

