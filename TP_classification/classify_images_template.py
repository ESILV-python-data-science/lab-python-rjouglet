# -*- coding: utf-8 -*-
"""
Classify digit images
C. Kermorvant - 2017
"""


import argparse
import logging
import time
import sys

from tqdm import tqdm
import pandas as pd
from PIL import Image, ImageFilter
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn import svm, metrics, neighbors
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import numpy as np
import pickle


# Setup logging
logger = logging.getLogger('classify_images.py')
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(ch)

#For a given image, this function compute a 8x8 subresolution image
#and return a feature vector with pixel values.
# 0 must correspond to white and 255 to black.
def extract_features_subresolution(img,img_feature_size = (8, 8)):
    """
    Compute a 8x8 subresolution image
	:param img: the original image (can be color or gray)
	:type img: pillow image
	:return: pixel values of the image in subresolution
	:rtype: list of int in [0,255]
    """

    # convert color images to grey level
    gray_img = img.convert('L')
    # find the min dimension to rotate the image if needed
    min_size = min(img.size)
    if img.size[1] == min_size:
        # convert landscape  to portrait
        rotated_img = gray_img.rotate(90, expand=1)
    else:
        rotated_img = gray_img

    # reduce the image to a given size
    reduced_img = rotated_img.resize(
        img_feature_size, Image.BOX).filter(ImageFilter.SHARPEN)

    # return the values of the reduced image as features
    return [255 - i for i in reduced_img.getdata()]

# Train and print metrics for a given classifier with training set and test test
# Return train accuracy score and test accuracy score
def classifier_train_test(classifier_name, classifier,X_train, X_test, y_train, y_test) :
    # Do Training@
    t0 = time.time()
    classifier_result=classifier.fit(X_train, y_train)
    logger.info("Training done in %0.3fs" % (time.time() - t0))

    # Do testing
    logger.info("Testing "+classifier_name)
    t0 = time.time()
    # predicted_validation = neigh.predict(X_validation)
    predicted_test = classifier.predict(X_test)

    # Print score produced by metrics.classification_report and metrics.accuracy_score
    logger.info("Testing  done in %0.3fs" % (time.time() - t0))
    print('Score on training : %f' % classifier.score(X_train, y_train))
    print('Score on testing : %f' % classifier.score(X_test, y_test))
    print(metrics.classification_report(y_test,predicted_test))
    return classifier.score(X_train, y_train),metrics.accuracy_score(y_test,predicted_test)

# Split the initial train set into 10 test set (1%,10%,20%,40%,60%,80%,100%) of original training set
def learning_impact(classifier_name,classifier,X_train,y_train):
    samples_train_rate=[0.01,0.10,0.20,0.40,0.60,0.80,1.00]
    train_accuracy = [0,0,0,0,0,0,0]
    test_accuracy = [0,0,0,0,0,0,0]
    for i in range(0,7): # for i=0 to i=6
        print('\n'+classifier_name+' :')
        X_train_set, X_unused, y_train_set, y_unused = train_test_split(X_train, y_train, train_size=samples_train_rate[i],shuffle=False)
        logger.info("Train set size is {}".format(X_train_set.shape))
        logger.info("Train size is {} % from original train set".format(samples_train_rate[i]*100))

        # Do Training and Testing
        train_accuracy[i],test_accuracy[i] = classifier_train_test(classifier_name,classifier,X_train_set, X_test, y_train_set, y_test)
        print(classifier_name+' Test accuracy score :', test_accuracy[i])
    # Display Training curve
    samples_train=X_train_set.shape[0]*np.array(samples_train_rate)
    display_learning_curve(classifier_name+" Learning curves",samples_train,train_accuracy,test_accuracy)
    plt.show()

# Display graph for training and testing accuracy for a variable train size
def display_learning_curve(title,samples_train,train_accuracy,test_accuracy) :
    plt.figure()
    plt.title(title)
    plt.xlabel("Training set size")
    plt.ylabel("Accuracy")
    plt.grid()
    plt.plot(samples_train,train_accuracy,linewidth=2.5,label="Train accuracy ")
    plt.plot(samples_train,test_accuracy,linewidth=2.5,label="Test accuracy ")
    plt.legend(loc="best")
    return plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract features, train a classifier on images and test the classifier')
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--images-list',help='file containing the image path and image class, one per line, comma separated')
    input_group.add_argument('--load-features',help='read features and class from pickle file')
    parser.add_argument('--save-features',help='save features in pickle format')
    parser.add_argument('--limit-samples',type=int, help='limit the number of samples to consider for training')
    parser.add_argument('--learning-curve',action='store_true',help='study the impact of a growing training set [(1%,10%,20%,40%,60%,80%,100%) of original training set] on the accuracy')
    classifier_group = parser.add_mutually_exclusive_group(required=True)
    classifier_group.add_argument('--nearest-neighbors',type=int)
    classifier_group.add_argument('--nearest-neighbors-logistic-regression', action='store_true', help='train knn with 60% training, 20% validation select the best k for k=1 to k=9, 20% test then do logistic regression')
    classifier_group.add_argument('--logistic-regression', action='store_true')
    classifier_group.add_argument('--features-only', action='store_true', help='only extract features, do not train classifiers')
    args = parser.parse_args()

    if args.load_features:
        # read features from to_pickle
        dataframe = pd.read_pickle(args.load_features)
        if args.limit_samples:
            dataframe = dataframe.sample(n=args.limit_samples)
        y = list(dataframe['class'])
        X = dataframe.drop(columns='class')
        pass
    else:
        # Load the image list from CSV file using pd.read_csv
        # see the doc for the option since there is no header ;
        file_list = []
        file_list = pd.read_csv(args.images_list, header=None, names=['filename', 'class']) # return a DataFrame
        #logger.info('Loaded {} images in {}'.format(all_df.shape,args.images_list))

        # Extract the feature vector on all the pages found
        # Modify the extract_features from TP_Clustering to extract 8x8 subresolution values
        # white must be 0 and black 255
        data = [] # images with features
        y = file_list['class'] # number to predict
        for i_path in file_list['filename']:
            page_image = Image.open(i_path)
            data.append(extract_features_subresolution(page_image))

        # check that we have data
        if not data:
            logger.error("Could not extract any feature vector or class")
            sys.exit(1)

        # convert to np.array
        X = np.array(data)


    # save features
    if args.save_features:
        # convert X to dataframe with pd.DataFrame and save to pickle with to_pickle
        dataframe = pd.DataFrame(X)
        dataframe['class'] = y
        dataframe.to_pickle(args.save_features + '.pkl')
        logger.info('Saved {} features and class to {}'.format(dataframe.shape,args.save_features))

    # Train classifier
    logger.info("Training Classifier")

    if args.features_only:
        logger.info('No classifier to train, exit')
        sys.exit()

    if args.nearest_neighbors:
        # create KNN classifier with args.nearest_neighbors as a parameter
        neigh = neighbors.KNeighborsClassifier(n_neighbors=args.nearest_neighbors)
        logger.info('Use kNN classifier with k= {}'.format(args.nearest_neighbors))

        # Use train_test_split to create train and validation/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)
        logger.info("Train set size is {}".format(X_train.shape))
        logger.info("Test set size is {}".format(X_test.shape))

        # If we want to study Increasing training set impact
        if args.learning_curve:
            learning_impact("KNN",neigh,X_train,y_train);
        else:
            # Do Training and testing
            train_accuracy, test_accuracy = classifier_train_test("KNN",neigh,X_train, X_test, y_train, y_test)
            print('Knn Train accuracy score : ',train_accuracy)
            print('Knn Test accuracy score :', test_accuracy)


    elif args.logistic_regression:
        # create logistic regression classifier with args.logistic_regression as a parameter
        logistic_reg = LogisticRegression()
        logger.info('Use logistic_regression classifier with k= {}'.format(args.logistic_regression))

        # Use train_test_split to create train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)
        logger.info("Train set size is {}".format(X_train.shape))
        logger.info("Test set size is {}".format(X_test.shape))

        # If we want to study Increasing training set impact
        if args.learning_curve:
            learning_impact("Logistic Regression",logistic_reg,X_train,y_train);
        else:
            # Do Training and testing
            train_accuracy, test_accuracy = classifier_train_test("Logistic Regression",logistic_reg,X_train, X_test, y_train, y_test)
            print('Logistic Train accuracy score :', train_accuracy)
            print('Logistic Test accuracy score :', test_accuracy)


    elif args.nearest_neighbors_logistic_regression:
        # Use train_test_split to create train and validation/test split
        X_train_validation, X_test, y_train_validation, y_test = train_test_split(X, y, train_size=0.8)
        logger.info("Test set size is {}".format(X_test.shape))

        # Use train_test_split to create train/validation split
        X_train, X_validation, y_train, y_validation = train_test_split(X_train_validation, y_train_validation, train_size=0.8)
        logger.info("Training set size is {}".format(X_train.shape))
        logger.info("Validation set size is {}".format(X_validation.shape))

        # Select best KNN classifier k for k=0 to 9
        best_k = [0,0] # k, accuracy_score
        for i in range(1,10):
            neigh = neighbors.KNeighborsClassifier(n_neighbors=i)
            logger.info('Use kNN classifier with k= {}'.format(i))
            # Do Training and Testing
            train_accuracy, test_accuracy = classifier_train_test("KNN",neigh,X_train, X_validation, y_train, y_validation)
            print('Knn accuracy score :', test_accuracy)
            if (test_accuracy>best_k[1]):
                best_k[0] = i
                best_k[1] = test_accuracy

        # Execute classifier
        print('Best k is = {}'.format(best_k[0]))
        print('Accuracy score from validation = {}'.format(best_k[1]))
        neigh = neighbors.KNeighborsClassifier(n_neighbors=best_k[0])
        logistic_reg = LogisticRegression()

        # Display Training curve
        if args.learning_curve:
            # Split the initial train set into 10 test set (1%,10%,20%,40%,60%,80%,100%) of original training set
            samples_train_rate=[0.01,0.10,0.20,0.40,0.60,0.80,1.00]
            train_accuracy_knn = [0,0,0,0,0,0,0]
            test_accuracy_knn = [0,0,0,0,0,0,0]
            train_accuracy_log = [0,0,0,0,0,0,0]
            test_accuracy_log = [0,0,0,0,0,0,0]
            for i in range(0,7): # for i=0 to i=6
                X_train_set, X_unused, y_train_set, y_unused = train_test_split(X_train_validation, y_train_validation, train_size=samples_train_rate[i],shuffle=False)
                logger.info("Train set size is {}".format(X_train_set.shape))
                logger.info("Train size is {} % from original train set".format(samples_train_rate[i]*100))

                # Do Training and Testing
                print('Knn :')
                train_accuracy_knn[i],test_accuracy_knn[i] = classifier_train_test("KNN",neigh,X_train_set, X_test, y_train_set, y_test)
                print('Logistic Regression :')
                train_accuracy_log[i],test_accuracy_log[i] = classifier_train_test("Logistic Regression",logistic_reg,X_train_set, X_test, y_train_set, y_test)
                print('Knn Test accuracy score :', test_accuracy_knn[i])
                print('Logistic Test accuracy score :', test_accuracy_log[i])

            samples_train=X_train_set.shape[0]*np.array(samples_train_rate)
            display_learning_curve("Nearest Neighbors Learning curves",samples_train,train_accuracy_knn,test_accuracy_knn)
            display_learning_curve("Logistic Regression Learning curves",samples_train,train_accuracy_log,test_accuracy_log)
            plt.show()

        else:
            # Do Training and testing
            print('Knn :')
            train_accuracy_knn,test_accuracy_knn = classifier_train_test("KNN",neigh,X_train_validation, X_test, y_train_validation, y_test)
            print('Logistic Regression :')
            train_accuracy_log,test_accuracy_log = classifier_train_test("Logistic Regression",logistic_reg,X_train_validation, X_test, y_train_validation, y_test)
            print('Knn Test accuracy score :', test_accuracy_knn)
            print('Logistic Test accuracy score :', test_accuracy_log)

    else:
        logger.error('No classifier specified')
        sys.exit()