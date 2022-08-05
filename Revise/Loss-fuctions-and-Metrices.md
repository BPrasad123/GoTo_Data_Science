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

**Reference**:  
https://www.baeldung.com/cs/cost-vs-loss-vs-objective-function


## Loss Functions ##

Loss functions and their properties:  
http://www.cs.cornell.edu/courses/cs4780/2015fa/web/lecturenotes/lecturenote10.html



### Regression ###

* **MSE (Mean Squared Error)**  
This quadratic function although amplifies the error between prediction and actual value when they are close, but is not robust to presence of outliers in the data. It tends to favour mean of the target data.  
When to use: When we want to penalize larger error more and the response (target) data follows normal distribution.  

* **MAE (Mean Absolute Error)**  
Although this linear loss function is robust to outliers but finding gradients involve more complicated techniques. Moreover, handling the absolute or modulus operator in mathematical equations is not easy. It tends to favour median of the target data.  

  When to use: When the target data follows multimodal distribution, but not necessarily.  
  Reference:  
  https://peltarion.com/knowledge-center/documentation/modeling-view/build-an-ai-model/loss-functions/mean-absolute-error

  It is not a differentiable function. However some algorithm implementations use approximate derivative for that point where the derivative is not defined.  
  https://datascience.stackexchange.com/questions/61743/when-using-absolute-error-in-gradient-descent-how-to-calculate-the-derivative



  https://people.csail.mit.edu/dsontag/courses/ml16/slides/notes_convexity16.pdf

* **Poisson Loss**  
If the target data is count (positive integer) within a time frame, then considering experimenting with poisson loss in regression.

  Reference:  
  https://peijin.medium.com/the-poisson-deviance-for-regression-d469b56959ce

* **Huber Loss**  
This function takes the good both both squared and absolute error loss functions. With a threshold error value, the loss is quadratic for small error and linear for larger error.

* **Fair Loss**  
  https://andrewpwheeler.com/2021/12/22/learning-a-fair-loss-function-in-pytorch/

  Fair loss used in XGBOOST  
  https://github.com/alno/kaggle-allstate-claims-severity


* **LogCosh**  
  Due to numerical instability we can rewrite this function in terms of exponentials.  
  https://jiafulow.github.io/blog/2021/01/26/huber-and-logcosh-loss-functions/
  
    

* **Custom loss functions:**  


  * RandomForestRegressor   
    It is difficult to implement. All the classifiers in sklearn are inherited from the base class *RegressorMixin* and that uses *r-square* as the loss function.
  * RandomForestClassifier  
    It is difficult to implement. All the classifiers in sklearn are inherited from the base class *ClassiferMixin* and that uses accuracy as the loss function.  
    Ref:  
    https://stackoverflow.com/questions/54267745/implementing-custom-loss-function-in-scikit-learn  
    https://github.com/scikit-learn/scikit-learn/issues/3071
  * LightGBM  
    We can choose from a variety of loss functions that are by default supported.  
    Ref:  
    https://lightgbm.readthedocs.io/en/latest/Parameters.html

    For the problem at hand we can define a custom loss function as well and pass that as objective function to pass that as a parameter to lightgbm algorithm.  
    Ref:  
    https://github.com/manifoldai/mf-eng-public/blob/master/notebooks/custom_loss_lightgbm.ipynb
  * XGBOOST  
    Templates for creating custom loss function for XGBOOST.
    https://stackoverflow.com/questions/45006341/xgboost-how-to-use-mae-as-objective-function


  https://towardsdatascience.com/custom-loss-functions-for-gradient-boosting-f79c1b40466d

  https://alex.miller.im/posts/linear-model-custom-loss-function-regularization-python/

  https://kiwidamien.github.io/custom-loss-vs-custom-scoring.html


### Classification ###

**What is Entropy?**  
One needs to understand **Entropy** first before diving into the loss functions built around entropy.

Simply put, it is basically the amount of information needed to reduce the uncertainty about a sample drawn from a probability distribution.

Say,  

[Case 1]  
There is a chance of 50% rainy day and 50% sunny day. You just need to tell me 0 0r 1 to confirm, that is 1 bit of data. Earlier I had two options and your confirmation reduced my uncertainty by 2.  

[Case 2]  
There is 75% chance of rainy day and 25% chance of sunny day. If you confirm it is going to be sunny day, then you are reducing my uncertainty by 1/0.25, that is a factor of 4.  

[Case 3]  
If there are 8 different possible weather conditions with all equal probabilities, then 3 bits (001, 101, 011, 010, 100, 110, 010, 111) are required to communicate to confirm the outcome. If you observe the pattern, then the mathematical formula for the information required to reduce uncertainty in this case, is log (8) with base 2 = 3  

[Case 4]  
Extending it to instances where possible outcomes have different probabilities. With 75% chance of rainy day and 25% chance of sunny day, information received to reduce uncertainty on an average =   
0.75*log(1/0.75) + 0.25*log (1/0.25) = -0.75*log(0.75) - 0.25*log(0.25) = - sum of p * log(p) where p is the probability of true occurrence.  
Note: Probability is multiplied with log of the probability is to average out.  

[Case 5]  
Let us say our prediction distribution is q and true distribution is p. Now the function of cross entropy will be - sum of p * log(q). If the prediction probability distribution is equal to the true probability distribution then cross entropy value will be equal to the entropy. If the prediction probability distribution is different, then the value of cross entropy will be more than the true probability distribution. That difference is called **KL divergence**.  


Reference:  
https://www.youtube.com/watch?v=ErfnhcEV1O8


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





References:  
https://towardsdatascience.com/cross-entropy-for-classification-d98e7f974451

https://www.enjoyalgorithms.com/blog/loss-and-cost-functions-in-machine-learning

https://www.analyticsvidhya.com/blog/2019/08/detailed-guide-7-loss-functions-machine-learning-python-code/  

https://machinelearningmastery.com/how-to-choose-loss-functions-when-training-deep-learning-neural-networks/


## Metrices ##

*  **Regression**  
Although R-square, MSE, MEA are used to explain the performance of the model, they are subject to the number of features, magnitude of the target variable.

    MAPE (Mean Absolute Percentage Error), however measures the error compared to the absolute value of the target value.  

    Depending upon the problem scenario context the corresponding metric can be used to compare the model performance.  

    Reference:  
    https://datascience.stackexchange.com/questions/37168/high-rmse-and-mae-and-low-mape

* **Time Series**    
    MAD or MAE  
    MSE  
    RMSE  
    MASE  
      Ratio of MAE and average of naive error (if predicted number is same as the last instance)  
      If less than 1 then it performs better than naive method.  
    MAPE  
      Ratio of error to the actual value. However, if the actual value is zero, then it is impossible to calculate. Moreover, if the actual value is less and error is more, MAPE value is higher.  

    Difference between MAPE and MASE: https://www.andreaperlato.com/tspost/statistica-background-for-time-series/#:~:text=The%20Mean%20Absolute%20Percentage%20Error,makes%20MASE%20a%20favor%20metrics.

    https://towardsdatascience.com/mad-over-mape-a86a8d831447

* **Classification**  

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
