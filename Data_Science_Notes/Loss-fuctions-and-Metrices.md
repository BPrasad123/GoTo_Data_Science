# Cost Function vs Loss Function vs Objective Function vs Metrices #

In general Cost, Loss and Objective functions are interchangeably used. However to differentiate them technically here are the definitions.

* **Loss Function**  
It is a differentiable function that calculates loss for a training record.

* **Cost Function**  
It is the average of errors for all the records in the training set calculated by the loss function.

* **Objective Function**  
Reduction of the cost is not the only purpose. That will make the model very well trained on the training data, that is overfitting. However, it has to be well generalized, for that we add regularization to the cost function that constitutes the objective function. However, in lot many cases, the loss function is considered as objective function as well.

* **Metrices**  
It is used to evaluate the performance of the models on training as well as validation and test datasets.

**References**:  
- General idea about loss, cost, metrices - [Link](https://www.baeldung.com/cs/cost-vs-loss-vs-objective-function)
- Loss functions and their properties - [Link](http://www.cs.cornell.edu/courses/cs4780/2015fa/web/lecturenotes/lecturenote10.html)

### **Regression (can be applied for Time Series as well)** ###
#### **Regression: Loss Functions** ####

* **MSE (Mean Squared Error)**  
Also called as L2 loss, this quadratic function although amplifies the error between prediction and actual value when they are close, but is not robust to presence of outliers in the data. It tends to favour mean of the target data. If a dataset is without outliers, it tends to outperform the model trained with L1 loss. However, the performance is impacted if there are outliers in the dataset.

  When to use: If we you can remove undesired outliers in the dataset and want a stable solution, then you should try L2 or MSE loss function.  

  Algorithm examples with L2 loss:  
    - sklearn.linear_model.SGDRegressor with loss='squared_error'
    - GradientBoostingRegressor(loss='ls')

  Comparison of L1 and L2 losses with and without outliers: [Link](http://rishy.github.io/ml/2015/07/28/l1-vs-l2-loss/)
  
* **MAE (Mean Absolute Error)**  
Also known as L1 loss, this linear loss function is robust to outliers. Since it is not a differentiable function at 0, finding gradients involve more complicated techniques such as approximation of derivatives.

  When to use: If we need outliers present in the dataset, then we should go for L1 loss. Moreover, if the target data follows multimodal distribution, then we can experiment with L1 loss.  
  
  Algorithm examples with L1 loss:
    - GradientBoostingRegressor(loss='lad')

  References:  
  - Short overview about MAE: [Link](https://peltarion.com/knowledge-center/documentation/modeling-view/build-an-ai-model/loss-functions/mean-absolute-error)
  - Comparison of L1 and L2 losses with and without outliers: [Link](http://rishy.github.io/ml/2015/07/28/l1-vs-l2-loss/)
  - Some algorithm implementations use approximate derivative or subgradients for that point where the derivative is not defined: [Link](https://datascience.stackexchange.com/questions/61743/when-using-absolute-error-in-gradient-descent-how-to-calculate-the-derivative)
  - More details on subgradients: [Link](https://people.csail.mit.edu/dsontag/courses/ml16/slides/notes_convexity16.pdf)

* **Poisson Loss**  
Poisson distribution:  
A Poisson distribution helps us to predict the probability of certain events happening when you know how often the event has occurred. It gives us the probability of a given number of events happening in a fixed interval of time.

  It is best used for rare events, as these tend to follow a Poisson distribution (as opposed to more common events which tend to be normally distributed). For example:
    - Number of colds contracted on airplanes.
    - Number of bacteria found in a petri dish.
    - Number of customer churning next week.

  If the target data follows poisson distribution, that it represents count (positive integer) within a time frame and following assumptions are met, then consider experimenting with poisson loss in regression.  
    - Y-values are counts. If your response variables aren’t counts, Poisson regression is not a good method to use.
    - Counts must be positive integers (i.e. whole numbers) 0 or greater (0,1,2,3…k). The technique will not work with fractions or negative numbers, because the Poisson distribution is a discrete distribution.
    - Counts must follow a Poisson distribution. Therefore, the mean and variance should be the same.
    - Explanatory variables must be continuous, dichotomous or ordinal.
    - Observations must be independent.

  Algorithm examples with Poisson loss:
  - sklearn.linear_model.PoissonRegressor  
  - HistGradientBoostingRegressor(loss="poisson", max_leaf_nodes=128)  
  
  Reference:  
  - Sklearn examples [Link](https://scikit-learn.org/stable/auto_examples/linear_model/plot_poisson_regression_non_normal_loss.html)
  - Article from Statisticshowto on poisson regression: [Link](https://www.statisticshowto.com/poisson-regression/)
  - Another article on poisson regression: [Link](https://peijin.medium.com/the-poisson-deviance-for-regression-d469b56959ce)  

* **Huber Loss**  
This function takes the good of both squared and absolute error loss functions. With a threshold error value, the loss is quadratic (or L2) for small error and linear (or L1) for larger error.  

  One big problem with using MAE is its constantly large gradient when using gradient decent for training. This can lead to missing minima at the end of training using gradient descent. While with MSE, gradient decreases as the loss gets close to its minima, making it more precise. Huber loss can be helpful here, as it curves around the minima which decreases the gradient.  

  Pros:
    - It is differentiable at zero.
    - Outliers are handled properly due to the linearity above delta.
    - The hyperparameter, 𝛿 can be tuned to maximize model accuracy.

  Cons:
    - The additional conditionals and comparisons make Huber loss computationally expensive for large datasets.
    - In order to maximize model accuracy, 𝛿 needs to be optimized and it is an iterative process.
    - It is differentiable only once. Hence, it cannot be used in XGBOOST that differentiates twice.

  Algorithm examples using Huber loss for training:
    - sklearn.linear_model.HuberRegressor
    - sklearn.linear_model.SGDRegressor with loss='huber'
    - sklearn.ensemble.GradientBoostingRegressor (loss{‘squared_error’, ‘absolute_error’, **‘huber’**, ‘quantile’}, default=’squared_error’)

* **LogCosh**  
  Log(cosh(x)) is approximately equal to (x ** 2) / 2 for small x and to abs(x) - log(2) for large x. This means that ‘logcosh’ works similar to the mean squared error, but will not be strongly affected by the occasional wildly incorrect prediction. Along with the advantages of Huber loss, it’s twice differentiable everywhere, unlike Huber loss. [Ref](https://www.xpertup.com/blog/deep-learning/types-of-loss-functions-part-2/)

  Log Cosh Loss addresses the small number of problems that can arise from using Mean Absolute Error due to its sharpness. Log(cosh(x)) is a way to very closely approximate Mean Absolute Error while retaining a 'smooth' function. [Ref](https://orchardbirds.github.io/bokbokbok/tutorials/log_cosh_loss.html)

  Due to numerical instability we can rewrite this function in terms of exponentials. Moreover, large y-values can cause issues, which is why the y-values can be scaled to experiment. [Link](https://jiafulow.github.io/blog/2021/01/26/huber-and-logcosh-loss-functions/)

  Algorithm examples using log cosh loss function -  
    - LightGBM and XGBOOST with loss function from bokbokbok: [Link](https://orchardbirds.github.io/bokbokbok/tutorials/log_cosh_loss.html)
    - tf.keras.losses.LogCosh - [Link1](https://www.tensorflow.org/api_docs/python/tf/keras/losses/LogCosh), [Link2](https://github.com/christianversloot/machine-learning-articles/blob/main/how-to-use-logcosh-with-keras.md)

* **Quantile Loss**  
  Let us say, we are ok with over prediction but the lower prediction will cost us more in business. We could use a loss function that is minimized at the desired quantile. For example, a prediction for quantile 0.9 should over-predict 90% of the times.

  Examples of algorithms with Quantile Loss:  
    * sklearn.linear_model.QuantileRegressor
    * ensemble.GradientBoostingRegressor(loss='quantile', alpha=q)
    * Tensorflow Keras

  References -  
  - Blog on quantile loss function - [Link](https://www.evergreeninnovations.co/blog-quantile-loss-function-for-machine-learning/)
  - Lecture Note - [Link](https://artowen.su.domains/courses/305a/lec18.pdf)
  - Detailed blog - [Link](https://towardsdatascience.com/quantile-regression-from-linear-models-to-trees-to-deep-learning-af3738b527c3)

* **Special Loss functions**  
  * **Fair Loss**  
    [ *TBD - Short explanation to be added* ]  
    - Why fair loss - [Link](https://www.aurelielemmens.com/debiasing-algorithms-fair-machine-learning/) 
    - For classification - [Link](https://andrewpwheeler.com/2021/12/22/learning-a-fair-loss-function-in-pytorch/)
    - Fair loss used in XGBOOST - [Link](https://github.com/alno/kaggle-allstate-claims-severity)  
    - Fair loss for pytorch - [Link](http://vi.le.gitlab.io/fair-loss/) 


  * **Custom loss functions:**  
    Based on the business problem we might want to change the loss function so as to penalize more for certain data records during training. We can build a custom loss function and pass that to the model while fitting on the training data. However, please note that the loss function has to be differentiable and compatible with the algorithm it will be used with. Moreover, not all the algorithms support custom loss functions, especially sklearn algorithms.
  
    * Sklearn algorithms -   
    It is difficult to implement. For example most of the regressor models such as RandomForestRegressor in sklearn are inherited from the base class *RegressorMixin* and that uses *r-square* as the loss function.  
    Ref: [Stackoverflow](https://stackoverflow.com/questions/54267745/implementing-custom-loss-function-in-scikit-learn), [Github](https://github.com/scikit-learn/scikit-learn/issues/3071), [Kaggle](https://www.kaggle.com/questions-and-answers/172754)
  
    * LightGBM  
    We can choose from a variety of loss functions that are by default supported.  
    Ref: [LightGBM docs](https://lightgbm.readthedocs.io/en/latest/Parameters.html)

      For the problem at hand we can define a custom loss function as well and pass that as objective function to pass that as a parameter to lightgbm algorithm.  
      Ref: [Notebook on Github](https://github.com/manifoldai/mf-eng-public/blob/master/notebooks/custom_loss_lightgbm.ipynb)
    
    * XGBOOST  
    Templates for creating custom loss function for XGBOOST.  
    Ref: [Stackoverflow](https://stackoverflow.com/questions/45006341/xgboost-how-to-use-mae-as-objective-function)

    * Tensorflow Keras
    It provides options to use custom loss functions in algorithms. Even, the keras implementation of randomforest model support custom loss functions.  
    Ref: [Tensorflow Doc](https://www.tensorflow.org/decision_forests/api_docs/python/tfdf/keras/RandomForestModel)
    
    * Pytorch supports custom loss functions.

    Examples of custom loss function implementations:  
    - [TowardsDataScience Blog](https://towardsdatascience.com/custom-loss-functions-for-gradient-boosting-f79c1b40466d)
    - [Blog by Alex Miller](https://alex.miller.im/posts/linear-model-custom-loss-function-regularization-python/)
    - [Blog by Kiwi Damien](https://kiwidamien.github.io/custom-loss-vs-custom-scoring.html)


* **Algorithm Specific Loss functions:**  
  **XGBOOST** supports following loss functions -  
    - reg:squarederror: regression with squared loss.
    - reg:squaredlogerror: regression with squared log loss. But because log function is employed, rmsle might output nan when prediction value is less than -1. Hence be careful for using this.
    - reg:logistic: logistic regression
    - reg:pseudohubererror: regression with Pseudo Huber loss, a twice differentiable alternative to absolute loss.
    - reg:gamma: gamma regression with log-link. Output is a mean of gamma distribution. It might be useful, e.g., for modeling insurance claims severity, or for any outcome that might be gamma-distributed.
    - reg:tweedie: Tweedie regression with log-link. It might be useful, e.g., for modeling total loss in insurance, or for any outcome that might be Tweedie-distributed.
    - binary:logistic: logistic regression for binary classification, output probability
    - binary:logitraw: logistic regression for binary classification, output score before logistic transformation
    - binary:hinge: hinge loss for binary classification. This makes predictions of 0 or 1, rather than producing probabilities.
    - count:poisson: Poisson regression for count data, output mean of Poisson distribution
    - multi:softmax: set XGBoost to do multiclass classification using the softmax objective, you also need to set num_class(number of classes)
    - multi:softprob: same as softmax, but output a vector of ndata * nclass. The result contains predicted probability of each data point belonging to each class.
    - rank:pairwise, rank:ndcg, rank:map: For rank related use cases
    - survival:cox, survival:aft, aft_loss_distribution: For survival related use cases  
  Reference: [XGBOOST Docs](https://xgboost.readthedocs.io/en/stable/parameter.html)  

#### **Regression: Evaluation Metrics** ####
The evaluation metrics as used to measure the performance of the model. They do not need to be differentiable as during the evaluation the gradients are not calculated. And of course the loss function can be used as an evaluation metric to judge the performance of the model. Depending upon the problem scenario and context the corresponding metric can be used to compare the model performance as explained below.

* MSE
* MEA
* RMSE  
  The RMSE gives more importance to the large deviation of forecasting error, resulting in a higher RMSE value. Using the RMSE metric when a few large incorrect predictions from a model on some items can be very costly to the business.  

  You should use RMSE with caution, because a few large deviations in forecasting errors can severely punish an otherwise accurate model. For example, if one item in a large dataset is severely under-forecasted or over-forecasted, the error in that item skews the entire RMSE metric drastically, and may make you reject an otherwise accurate model prematurely. For use cases where a few large deviations are not of importance, consider using wQL or WAPE.  

* R-square (also known as coefficient of determination)
  How does the model perform against a simple prediction by taking average. With addition of new variable, the sum of squares that is in the denominator increases, hence the value of R square increases. However that does not mean the predictive power of the model increases. This will be a misleading conclusion.  
  Ref:  
  https://www.researchgate.net/post/Why_does_R_squared_increase_with_the_inclusion_of_an_interaction_term#:~:text=When%20you%20add%20another%20variable,increases%20your%20R%2Dsquared%20value.

* Adjusted R-square  
  To rectify misleading behavior, R² is adjusted with the number of independent variables.
  
* MAPE (Mean Absolute Percentage Error)  
  It measures the error compared to the absolute value of the target value. However, if the actual value is zero, then it is impossible to calculate. Moreover, if the actual value is less and error is more, MAPE value is higher. The MAPE equally penalizes for under-forecasting or over-forecasting.  
  
  You can use MAPE for datasets where forecasting for all SKUs should be equally weighted regardless of sales volume. For example, a retailer may prefer to use the MAPE metric to equally emphasize forecasting errors on both items with low sales and items with high sales.  

* MASE (Mean Absolute Scaled Error)  
  Ratio of MAE and average of naive error (if predicted number is same as the last instance). If less than 1 then it performs better than naive method. The MASE is a scale-free metric, which makes it useful for comparing models from different datasets. It is recommended to use the MASE metric when you are interested in measuring the impact of seasonality on your model. If your dataset does not have seasonality, we recommend using other metrics.

* Quantile Loss  
  Go with weighted QL measure at different quantiles when the costs of under-forecasting and over-forecasting differ. If the difference in costs is negligible, you may consider forecasting at the median quantile of 0.5 (P50) or use the WAPE metric, which is evaluated using the mean forecast. We want to prioritize over-forecasting and penalize under-forecasting.

  
Note: Although R-square, MSE, MEA are used to explain the performance of the model, they are subject to the number of features, magnitude of the target variable.

More details:  
  https://datascience.stackexchange.com/questions/37168/high-rmse-and-mae-and-low-mape  
  https://www.analyticsvidhya.com/blog/2021/10/evaluation-metric-for-regression-models/  

When to use which metric in forecasting -   
https://aws.amazon.com/blogs/machine-learning/measuring-forecast-model-accuracy-to-optimize-your-business-objectives-with-amazon-forecast/#:~:text=0.75%5D)%2F2.-,Weighted%20absolute%20percentage%20error%20(WAPE),to%20calculate%20the%20absolute%20error.
  
Difference between MAPE and MASE -  
https://www.andreaperlato.com/tspost/statistica-background-for-time-series/#:~:text=The%20Mean%20Absolute%20Percentage%20Error,makes%20MASE%20a%20favor%20metrics.  
Choosing between MAD (Or MAE) and MAPE -  
https://towardsdatascience.com/mad-over-mape-a86a8d831447


### **Classification** ###

Before we dive into loss functions used in classification, we need to understand what Entropy is, as many loss functions are built using entropy.  

**What is Entropy?**  
    Simply put, it is basically the amount of information needed to reduce the uncertainty about a sample outcome drawn from a probability distribution.

    Say,  

    [Case 1]  
    There is a chance of 50% rainy day and 50% sunny day, and you just need to tell me 0 or 1 to confirm. That is just 1 bit of data. Earlier I had two possible outcomes and your confirmation reduced my uncertainty by 2.  

    [Case 2]  
    There is a 75% chance of rainy day and 25% chance of sunny day. If you confirm it is going to be sunny day, then you are reducing my uncertainty by 1/0.25, that is a factor of 4.  

    [Case 3]  
    If there are 8 different possible weather conditions with all equal probabilities, then 3 bits (001, 101, 011, 010, 100, 110, 010, 111) are required to communicate to confirm the outcome. If you observe the pattern, then the mathematical formula for the information required to reduce uncertainty in this case, is log (8) with base 2 = 3  

    [Case 4]  
    Extending it to instances where possible outcomes have different probabilities. With 75% chance of rainy day and 25% chance of sunny day, information received to reduce uncertainty on an average 
    = 0.75*log(1/0.75) + 0.25*log (1/0.25) 
    = -0.75*log(0.75) - 0.25*log(0.25) 
    = - sum of p * log(p) 
    where p is the probability of true occurrence.  
    Note: Probability is multiplied with log of the probability is to average out.  

    [Case 5]  
    Let us say our prediction distribution is q and true distribution is p. Now the function of cross entropy will be - sum of p * log(q). If the prediction probability distribution is equal to the true probability distribution then cross entropy value will be equal to the entropy. If the prediction probability distribution is different, then the value of cross entropy will be more than the true probability distribution. That difference is called KL divergence.  

    Ref: https://www.youtube.com/watch?v=ErfnhcEV1O8


#### **Classification: Loss Functions** ####

**Multi classification**  
* Cross entropy (Same as Log loss, aka logistic loss)

* Kullback Leibler Divergence Loss (KL Loss)  
  Can be used for multi class classification

**Multi classification with classes one hot encoded**  
* Categorical Cross Entropy
  * Cross entropy where target is one hot encoded

**Binary classification**   
* Binary Cross Entropy
  * Special case of cross entropy where target values are 0 and 1. If prediction is 0.6 is for class 1, then by default prediction for class 0, is 1-0.6=0.4

**Multi class multi label classification**  
* Treat the problem as 1 vs rest. And then binary cross entropy is applied.  

**SVM classifier**  
* Hinge Loss
  Used with SVM that penalises model for wrong classifications as well weak predictions. Classes are defined as -1 and 1 instead of 0 and 1.  
  Can support multi-classification as well.  

  Reference:  
  https://medium.com/analytics-vidhya/loss-functions-multiclass-svm-loss-and-cross-entropy-loss-9190c68f13e0

**Focal Loss**  


Gini impurity in Random Forest
Hinge loss

* **Special Loss functions**  
  * **Fair Loss**  
    [*TBD - Short explanation to be added*]  
    For classification -  
    https://andrewpwheeler.com/2021/12/22/learning-a-fair-loss-function-in-pytorch/  
    Fair loss for pytorch -  
    http://vi.le.gitlab.io/fair-loss/  

  * RandomForestClassifier  
    It is difficult to implement. All the classifiers in sklearn are inherited from the base class *ClassiferMixin* and that uses accuracy as the loss function.  







References:  
https://towardsdatascience.com/cross-entropy-for-classification-d98e7f974451

https://www.enjoyalgorithms.com/blog/loss-and-cost-functions-in-machine-learning

https://www.analyticsvidhya.com/blog/2019/08/detailed-guide-7-loss-functions-machine-learning-python-code/  

https://machinelearningmastery.com/how-to-choose-loss-functions-when-training-deep-learning-neural-networks/


### Metrices ###

  * Binary Classification

  * Multi classification

  * Multi label multi classification

  * Imbalanced classification
    * Cohen's Kappa score  
      - It basically tells you how much better your classifier is performing over the performance of a classifier that simply guesses at random according to the frequency of each class. In other words denotes the degree of agreement between true values and predicted values.
      - Value ranges from -1 to 1.
      - Applicable to multi class classification as well.
      - Kappa value is not as easy to interpret as accuracy.
      - Reference:  
        https://thedatascientist.com/performance-measures-cohens-kappa-statistic/  
        
        https://thenewstack.io/cohens-kappa-what-it-is-when-to-use-it-and-how-to-avoid-its-pitfalls/
        
        https://towardsdatascience.com/multi-class-metrics-made-simple-the-kappa-score-aka-cohens-kappa-coefficient-bdea137af09c  
        
        https://stats.stackexchange.com/questions/82162/cohens-kappa-in-plain-english  
    * Mathew's Correlation Coefficient (MCC)  
      - Unlike F1-score, MCC takes all the four components including True Negatives into the account. Hence it is a good measure of class imbalanced dataset.  
      - Value ranges from -1 to 1
      - Can be used for multi class classification
      - Reference:  
        https://bmcgenomics.biomedcentral.com/articles/10.1186/s12864-019-6413-7

        https://lettier.github.io/posts/2016-08-05-matthews-correlation-coefficient.html

        https://towardsdatascience.com/matthews-correlation-coefficient-when-to-use-it-and-when-to-avoid-it-310b3c923f7e

    * F1 score
      - F1 score is highly influenced by which class is labeled as positive. It is necessary to assign lower class as positive.
      - Value range from 0 to 1  

https://www.upgrad.com/blog/top-dimensionality-reduction-techniques-for-machine-learning/



Perplexity
NLP and CV related metrices


Algorithm specific loss functions:  

XGBOOST  
https://campus.datacamp.com/courses/extreme-gradient-boosting-with-xgboost/regression-with-xgboost?ex=3

https://xgboost.readthedocs.io/en/stable/parameter.html


Random Forest  
https://www.reddit.com/r/learnmachinelearning/comments/t8254n/what_is_the_loss_function_for_trees_and_random/

Decision Tree  
https://buggyprogrammer.com/how-to-calculate-the-decision-tree-loss-function/  


MAP (Mean average precision)  
https://blog.paperspace.com/mean-average-precision/amp/  
https://www.v7labs.com/blog/mean-average-precision#:~:text=Mean%20Average%20Precision%20(mAP)%20is,to%20evaluate%20the%20their%20models.



Blogs to check:  
https://orchardbirds.github.io/bokbokbok/tutorials/log_cosh_loss.html
https://www.cs.cmu.edu/~mgormley/courses/10701-f16/slides/lecture4.pdf  
https://www.evergreeninnovations.co/blog-quantile-loss-function-for-machine-learning/  


Keras losses  
https://keras.io/api/losses/regression_losses/


Find out SVM loss


https://medium.com/@ShreyaG0127/gini-vs-entropy-how-do-they-find-the-optimum-split-e98acf48caa1


https://towardsdatascience.com/a-tale-of-two-macro-f1s-8811ddcf8f04


https://towardsdatascience.com/multi-class-metrics-made-simple-part-ii-the-f1-score-ebe8b2c2ca1

https://h2o.ai/blog/regression-metrics-guide/

https://medium.com/the-rise-of-unbelievable/what-is-evaluation-metrics-and-when-to-use-which-metrics-23d16446d690


https://peltarion.com/knowledge-center/documentation/modeling-view/build-an-ai-model/loss-functions/focal-loss

https://peltarion.com/knowledge-center/documentation/modeling-view/build-an-ai-model/loss-functions/squared-hinge


https://www.baeldung.com/cs/mape-vs-wape-vs-wmape

https://www.ibm.com/docs/en/cognos-analytics/11.1.0?topic=forecasting-statistical-details


https://www.andreaperlato.com/tspost/statistica-background-for-time-series/#:~:text=The%20Mean%20Absolute%20Percentage%20Error,makes%20MASE%20a%20favor%20metrics.


why note MAPE:
https://towardsdatascience.com/forecast-kpi-rmse-mae-mape-bias-cdc5703d242d


https://towardsdatascience.com/why-not-mse-as-a-loss-function-for-logistic-regression-589816b5e03c

https://towardsdatascience.com/multi-class-metrics-made-simple-the-kappa-score-aka-cohens-kappa-coefficient-bdea137af09c

