
## Sampling Techniques ##

If the dataset if highly imbalanced we can use several upsampling, downsampling or even hybrid techniques to have sizable amount of records for minority classes. Following is the list of techniques we can leverage and experiment with.  

* Oversampling/Upsampling
  * **SMOTE**  
    By **default uses KNN** to generate synthetic data. If K=2, it randomly finds an instance and two of its nearest neighbours and then create new data doing the interpolation between the initial record and nearest records. However there is chance that the space where the interpolated record was created might belong to other class, hence that would result in incorrect dataset.

    There are many other variants of SMOTE.  
    * **BorderlineSMOTE**  
        Create a two sets of records during interpolation - noise and border. If all of the surrounding neighbours are of different classes, then it is called noise. If those classes are of mixed classes, then that is called border. Only the border instances are considered for synthetic data generation.
    * **K-means SMOTE**  
        K-means cluster technique is applied to find the clusters and only those with certain portion of minority classes are considered for data generation.
    * **SVM SMOTE**  
        SVM is applied to find the support vectors. Synthetic data is created along the support vectors by doing interpolation or extrapolation between support vectors for minority classes and nearest neighbours.
    * **SMOTE-NC**  
        If we have mix of numerical and categorical data in the independent features, then we go with SMOTE-NC approach.
    * **SMOTE-N**  
        If all the predictor variables are categorical, then we go with SMOTE-N technique
  * **ADASYN**  
    Uses **density distribution** to generate synthetic data. Adaptively changes decision boundaries for classes difficult to learn.
* Undersampling/Downsampling  
  * **NearMiss**  
    Finds records close to each other and removes the records belong to majority class.
* Oversampling + undersampling  
  * **SMOTE+TOMEK**  
    Clears overlapping records between classes thus results in clean separation along decision boundaries.
  * **SMOTE+ENN**  
    First implements SMOTE and then removes records resulting in wrong classification.

* Imbalance problem in regression

Need to transform the target variable to normal distribution for better result.

https://towardsdatascience.com/data-imbalance-in-regression-e5c98e20a807#:~:text=Data%20imbalance%20is%20not%20only,variable%20can%20boost%20the%20performance  
https://towardsdatascience.com/strategies-and-tactics-for-regression-on-imbalanced-data-61eeb0921fca  



**Refences:**  

Different sampling techniques:  
https://www.analyticsvidhya.com/blog/2020/10/overcoming-class-imbalance-using-smote-techniques/

SMOTE variants:  
https://medium.com/analytics-vidhya/handling-imbalanced-data-by-oversampling-with-smote-and-its-variants-23a4bf188eaf


## Class Weights ##

Assigning higher weight to under-represented classes result in higher error for wrong prediction during the training.

Although scikit-learn library supports class weights but does not compute class weights automatically for one hot encoded classes nor multi-label classifications. For that a custom function can be created to calculate the weights as per the class frequencies and passed to the model directly as a dictionary.

**Reference**:  
https://towardsdatascience.com/dealing-with-imbalanced-data-in-tensorflow-class-weights-60f876911f99


## Algorithm ##

Try tree based models. That work well with imbalanced datasets compared to linear models.

## Check anomaly detection and change detection ##
There might be some malfunction or change in the user behaviour amounting to such less represented data. Check for that.

Reference:  
https://machinelearningmastery.com/tactics-to-combat-imbalanced-classes-in-your-machine-learning-dataset/

## Different Metrices to measure ##
Try

* **cappa (or cohen's cappa)**  
    Signifies how the model is performing against random guesses based on class frequencies.
* **MCC (Mathew's Correlation Coefficient)**
    Summarises the confusion matrix well with single score. Not influenced by what class if positive and what class if negative, unlike f1-score.

**Note**: If confusion matrix threshold is not known then check for Precision Recall AUC and ROC - AUC with more preference to the former. If the threshold is known, then go for MCC over f1.


**Reference**:  
https://bmcgenomics.biomedcentral.com/articles/10.1186/s12864-019-6413-7

https://towardsdatascience.com/matthews-correlation-coefficient-when-to-use-it-and-when-to-avoid-it-310b3c923f7e

https://towardsdatascience.com/the-best-classification-metric-youve-never-heard-of-the-matthews-correlation-coefficient-3bf50a2f3e9a

