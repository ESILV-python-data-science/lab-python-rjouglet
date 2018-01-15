# Digits classification with the MNIST database

## Question 5 : K-Nearest Neighbors algorithm (KNN)

_Test the k-NN with k= 1 on both the training and the test set. Print the score produced by metrics.accuracy_score_

The best value for this parameter depends on the classification task and has to be found by trying different values and selecting the one with the best accuracy. for k= 1 / accruacy score : 0.924 for k= 2 / accruacy score : 0.921 for k= 3 / accruacy score : 0.931 for k= 4 / accruacy score : 0.927 for k= 5 / accruacy score : 0.934 for k= 6 / accruacy score : 0.932

## Question 6 : KNN best value for k

_Create three sets : train set (60%), validation set (20%) and test set (20%), using twice train_test_split Train a kNN classifier with different values of k and report the train/valid/test accuracy. Select the is best value for k according to the accuracy on the dev set. Report the best performance of the classifier on the test set for this value of k._

Best value of K=6 Classification: precision recall f1-score support

      0       0.97      0.98      0.97      1237
      1       0.93      0.98      0.96      1347
      2       0.96      0.95      0.95      1157
      3       0.93      0.90      0.92      1178
      4       0.94      0.93      0.93      1150
      5       0.94      0.92      0.93      1113
      6       0.97      0.98      0.97      1131
      7       0.90      0.93      0.91      1283
      8       0.92      0.87      0.90      1171
      9       0.88      0.89      0.89      1233
      
Average score : 0.931458
 
For second session test, best result is obtain for k = 5
snapshops for k=1 to 9 are availables in answersImg folder
So the best value for k depend on the training set and the validation set will choose the best k for the specific training

## Question 7 : Logistic regression

_Train a Logistic Regression classifier on 80% and test on 20% of the samples. Report the accuracy and compare the the best result of the kNN classifier._

For first session test, Logistic Regression comparing to Knn with best k (between 1 and 9) result on the same test set.
For first session test, Logistic Regression comparing to Knn with best k (between 1 and 9) result on the same test set.

## Question 8 : Increasing Training Set

_We train the classifier on an increasing training set corresponding to 1%, 10%, 20%, 40%, 60%, 80% and 100% of the original training set. We will always be tested on the same test set (the original one). Study the impact of a growing training set on the accuracy as explained before. Report the training and test set accuracies for the 1NN, 2NN, kNN (k being the best value for k you previously found) and the Logisitic Regresstion._

Comparing Knn et Logistic Regression in a training set of 1300 samples (with 1%,10%,20%,40%,60%,80%,100% of total samples), [Knn with best k (between 1 and 9) result of the kNN classifier.]

## Question 10 : 

We can see that the errors bars are getting shorter. It means that we have less errors ad the estimation is better when test size increase.

## Question 11 :

SVM would be better for handwriiten digit recognition than Kernel.




