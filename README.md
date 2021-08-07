# Bayesian Inference: *An application to Kinect data*

*by Javier Rico Reche (jvirico@gmail.com)*

[*Online version*](https://www.notion.so/rico-projects/Bayesian-Inference-5eb7ac4f3b85418390ad0ee0ab68c9ea)

# Introduction

During this work we perform the classification of sequences of body positions of skeletal body movements recorded from a kinect device.

First we train two Bayesian models, Naïve Bayes and Linear Gaussian Model, the later considering dependencies between the positions of different parts of the skeleton.

Second, we classify a set of new instances using both methods, and finally we evaluate the results giving an Accuracy estimate using Stratified K-Fold Cross-Validation.


### Data

We use a fraction of the [Kinect Gesture Dataset](https://www.microsoft.com/en-us/download/details.aspx?id=52283&from=http%3A%2F%2Fresearch.microsoft.com%2Fen-us%2Fum%2Fcambridge%2Fprojects%2Fmsrc12%2F) from Microsoft Research Cambridge, that consists of sequences of human movements, represented as body-part locations, and the associated gesture to be recognized by a system. Although the original dataset contains more than 6h of recordings of 30 people, our subset has 2045 instances of body positions for 4 classes, 'arms lifted', 'right arm extended to one side', 'crouched' and 'right arm extended to the front'.

The data is provided in a .mat Matlab file and loaded in python using *scipy.io.loadmat* function.

It is organized in three data structures:

1. **data**: a matrix (20x3x2045) with information of joints and their corresponding 3D coordinates of 2045 instances.
2. **labels**: a vector (2045x1) with the class label for each instance.
3. **individuals:** a vector (2045x1) with the identification of each individual.


### Functions to compute in Log Space

To avoid underflow numerical issues caused by repeated product operations of small numbers we perform the Posterior's computation in Log Space.

To implement this we use the following functions:

1. ***log_normpdf***: Computes the natural logarithm of the normal probability.
2. ***compute_logprobs***: Computes the log probabilities of an instance.
3. ***normalize_logprobs***: Normalizes log probabilities.


# Evaluation

An evaluation of both, Naïve Bayes and Linear Gaussian Model have been performed using the following methods:

1. **Sequential Holdout** 80/20 split.
2. **Random Holdout** 80/20 split.
3. **Stratified 4-Fold Cross-validation**.


### Sequential Holdout

We use a split of 80% instances to build the training dataset and 20% for the testing dataset. The separation is done sequentially, therefore it does not consider that the instances of the original dataset have a specific order, and that instances of a specific class are mostly grouped together.

Since the method does not split in train and test datasets with data representing equally all classes, this method is the weakest one as we will see in the Results section.


### Random Holdout

For the generation of Random index vectors without repetitions we use the function ***RandomVect:***

We use again a split of 80% instances to build the training dataset and 20% for the testing dataset. 

This time the separation of instances is done randomly using ***RandomVect***, therefore we minimize the previous problem of not representing all clases in the train and test datasets equally.

No repetitions of instances is used in the process of splitting into train and test datasets.

### Stratified 4-Fold Cross-validation

To solve the problem of not representing all clases equally for train and test datasets we introduce Stratification, and a 4 Fold Cross-validation, assuring that the folds preserve the percentage of samples for each class.

To control the Cross-validation iterations we use the function ***GetPerformance***

As we can observe in the table '*Accuracy Table*' below, **Linear Gaussian Model outperforms Naïve Bayes** **Model** with all evaluation methods used. Being the former a Discriminative model that considers conditional probabilities (parent joints), and the later a Generative model with a more simplistic view of the data (strong variables independence), the results are congruent.

Regarding the method of validation, **Stratified Cross-validation** is clearly the **more robust** method we have used and therefore more reliable than Random Holdout. Although the results are similar on both methods, Cross-validation is usually the preferred because it gives the model the opportunity to train on multiple train-test splits.

This gives us a better indication of how well the model will perform on unseen data. Hold-out, on the other hand, is dependent on just one train-test split.

Regarding Sequential or Random Holdout, as we have mentioned before, splitting the data randomly produces a more 'fair' model than a sequential split, specially considering that the instances in the original dataset are grouped by classes. We can see, comparing Sequential Holdout and Random Holdout, that the results clearly support this. 

Finally, **Stratification** ensures even greater 'fairness' regarding to equal representation of classes in all datasets and folds.


# Conclusion
Regardless the naïve aspect of Naïve Bayes, and having into account that strong conditional independence assumptions rarely hold true, we observe good performance for this model, achieving a good balance between accuracy and simplicity of implementation.

Nevertheless, introducing dependencies between the positions of different parts of the skeleton makes Linear Gaussian Model outperform the Naïve Bayes approach.

Notice the importance to perform the calculation of the posteriors in Log Space to avoid floating point underflows, and ending up dealing with numbers of smaller absolute value than the computer can actually represent.


## Cite this work
    J. Rico (2019) Bayesian Inference: An Application to Kinect data
    [Source code](https://github.com/jvirico/bayesian_inference_linear_gaussian_model)
    [Report](https://rico-projects.notion.site/Bayesian-Inference-5eb7ac4f3b85418390ad0ee0ab68c9ea)