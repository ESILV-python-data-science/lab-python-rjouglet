# -*- coding: utf-8 -*-
"""
Cluster images based on visual similarity

C. Kermorvant - 2017
"""


import argparse
import logging
import os
import shutil
import multiprocessing
import sys

from joblib import Parallel, delayed
from tqdm import tqdm
from PIL import Image, ImageFilter
from sklearn.cluster import KMeans
import numpy as np

# default sub-resolution
IMG_FEATURE_SIZE = (12, 16)

# Setup logging
logger = logging.getLogger('cluster_images.py')
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(ch)


def extract_features(img):
    """
    Compute the subresolution of an image and return it as a feature vector

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
        IMG_FEATURE_SIZE, Image.BOX).filter(ImageFilter.SHARPEN)

    # return the values of the reduced image as features
    return [255 - i for i in reduced_img.getdata()]


def copy_to_dir(images, clusters, cluster_dir):
    """
    Move images to a directory according to their cluster name

    :param images: list of image names (path)
    :type images: list of path
    :param clusters: list of cluster values (int), such as given by cluster.labels_, associated to each image
    :type clusters: list
    :param cluster_dir: prefix path where to copy the images is a directory corresponding to each cluster
    :type images: path
    :return: None
    """

    for img_path, cluster in zip(images, clusters):
        # define the cluster path : for example "CLUSTERS/4" if the image is in cluster 4
        clst_path = os.path.join(cluster_dir, str(cluster))
        # create the directory if it does not exists
        if not os.path.exists(clst_path):
            os.mkdir(clst_path)
        # copy the image into the cluster directory
        shutil.copy(img_path, clst_path)


def get_jpg_files():
    """ Returns a list of jpeg files in the input directory """
    files = []
    print(os.path.dirname(os.path.realpath(__file__)) + "/" + args.images_dir)
    for file in os.listdir(os.path.dirname(os.path.realpath(__file__)) + "/" + args.images_dir):
        if ".jpg" not in file:
            continue
        files.append(os.path.dirname(os.path.realpath(__file__)) + "/" + args.images_dir + file)
    return files


def image_loader(image):
    """" Extract the features from the image"""
    my_image = Image.open(image)
    return extract_features(my_image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract features, cluster images and move them to a directory')
    parser.add_argument('--images-dir', required=True)
    parser.add_argument('--move-images')
    args = parser.parse_args()
    CLUSTER_DIR = ""
    if args.move_images:
        CLUSTER_DIR = args.move_images
        # Clean up
        if os.path.exists(CLUSTER_DIR):
            shutil.rmtree(CLUSTER_DIR)
            logger.info('remove cluster directory %s' % CLUSTER_DIR)
        os.mkdir(CLUSTER_DIR)

    # find all the pages in the directory
    images_path_list = []
    data = []
    if args.images_dir:
        SOURCE_IMG_DIR = args.images_dir
        images_path_list = get_jpg_files()

    if not images_path_list:
        logger.warning("Did not found any jpg image in %s" % args.images_dir)
        sys.exit(0)

    # Multi-core feature extraction.
    data = Parallel(n_jobs=multiprocessing.cpu_count())(
        delayed(image_loader)(image) for image in tqdm(images_path_list))

    if not data:
        logger.error("Could not extract any feature vector")
        sys.exit(1)


    # convert to np array (default format for scikit-learn)
    X = np.array(data)
    logger.info("Running clustering")

    # in the directory corresponding to its cluster

    kmeans_model = KMeans(n_clusters=11, random_state=1).fit(X)

    if not args.images_dir:
        logger.info(msg="Cluster image directory was not specified, exiting")
        sys.exit(1)

    copy_to_dir(images_path_list, kmeans_model.labels_, CLUSTER_DIR)


# Setup logging
logger = logging.getLogger('classify_images.py')
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(ch)


def extract_features_subresolution(img,img_feature_size = (8, 8)):
    """
    Compute the subresolution of an image and return it as a feature vector

    :param img: the original image (can be color or gray)
    :type img: pillow image
    :return: pixel values of the image in subresolution
    :rtype: list of int in [0,255]

    """
    # reduce the image to a given size
    reduced_img = img.resize(
        img_feature_size, Image.BOX).filter(ImageFilter.SHARPEN)
    # return the values of the reduced image as features
    return [i for i in reduced_img.getdata()]


def hyperoptize_knn(X_train,y_train,X_test,y_test,test_on_train_flag):
    # Create validation set so that train = 60% , validation = 20% and test =  20%
    X_train_hyper, X_valid_hyper, y_train_hyper, y_valid_hyper = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

    for k in [1,2,3,4,5,6,7,8,9,10]:
        logger.info("k={}".format(k))
        clf = neighbors.KNeighborsClassifier(k)
        clf.fit(X_train_hyper,y_train_hyper)


        for _name,_train_set,_test_set in [('train',X_train_hyper,y_train_hyper),('valid',X_valid_hyper,y_valid_hyper),('test',X_test,y_test)]:

            _predicted = clf.predict(_train_set)
            _accuracy = metrics.accuracy_score(_test_set, _predicted)
            logger.info("{} accuracy : {}".format(_name,_accuracy))



def hyperoptimze_rbf_svm(X_train,y_train,X_test,y_test,test_on_train_flag):
    C_range = np.logspace(-2, 10, 13)
    gamma_range = np.logspace(-9, 3, 13)
    param_grid = dict(gamma=gamma_range, C=C_range)
    cv = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    grid = GridSearchCV(svm.SVC(), param_grid=param_grid, cv=cv)
    grid.fit(X_train, y_train)

    logger.info("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))


def train_test_classifier(clf,X_train,y_train,X_test,y_test,test_on_train_flag):
    # Do classification
    t0 = time.time()
    logger.info("Training...")
    clf.fit(X_train,y_train)
    logger.info("Done in %0.3fs" % (time.time() - t0))

   # test on train set
    if test_on_train_flag:
        predicted_train = clf.predict(X_train)

    # test on test set
    predicted = clf.predict(X_test)

    if test_on_train_flag:
        train_acc = metrics.accuracy_score(y_train, predicted_train)
    else:
        train_acc = None
    test_acc = metrics.accuracy_score(y_test, predicted)
    return (train_acc,test_acc)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract features, train a classifier on images and test the classifier')
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--images-list',help='file containing the image path and image class, one per line, comma separated')
    input_group.add_argument('--load-features',help='read features and class from pickle file')
    parser.add_argument('--save-features',help='save feautures in pickle format')
    parser.add_argument('--limit-samples',type=int, help='limit the number of samples to consider for training')
    parser.add_argument('--limit-test-samples',type=int, help='limit the number of samples to consider for testing  ')
    parser.add_argument('--subresolution',type=int, default=8,help='value for square subresolution extaction (value x value)')
    parser.add_argument('--run-number',type=int, default=1, help='run N times the experiment with different sampling')
    parser.add_argument('--scale-features',action='store_true',default=False)
    parser.add_argument('--test-on-train',action='store_true',default=False)
    classifier_group = parser.add_mutually_exclusive_group()
    classifier_group.add_argument('--nearest-neighbors',type=int)
    classifier_group.add_argument('--optimize-nearest-neighbors',action='store_true')
    classifier_group.add_argument('--logistic-regression',action='store_true')
    classifier_group.add_argument('--svm-linear',action='store_true')
    classifier_group.add_argument('--svm-rbf',action='store_true')
    classifier_group.add_argument('--optimize-svm-rbf',action='store_true')

    action_group = parser.add_mutually_exclusive_group()
    #action_group.add_argument('--features-only', action='store_true', help='only extract features, not classification')
    action_group.add_argument('--classify', action='store_true')
    action_group.add_argument('--learning-curve', action='store_true')
    action_group.add_argument('--testing-curve', action='store_true')

    args = parser.parse_args()


    if args.load_features:
        df_features = pd.read_pickle(args.load_features)
        logger.info('Loaded {} features'.format(df_features.shape))
        if args.limit_samples:
            df_features = df_features.sample(n=args.limit_samples)

        # define X (features) and y (target)
        if 'class' in df_features.columns:
            X = df_features.drop(['class'], axis=1)
            y = df_features['class']
        else:
            logger.error('Can not find classes in pickle')
            sys.exit(1)
    else:


        all_df = pd.read_csv(args.images_list,header=None,names=['filename','class'])
        logger.info('Loaded {} images in {}'.format(all_df.shape,args.images_list))



        data = []



        if args.limit_samples:
            file_list = all_df['filename'][:args.limit_samples]
            y = all_df['class'][:args.limit_samples]
        else:
            file_list = all_df['filename']
            y = all_df['class']


        for i_path in tqdm(file_list):
            page_image = Image.open(i_path)
            data.append(extract_features_subresolution(page_image,(args.subresolution,args.subresolution)))

        # check that we have data
        if not data:
            logger.error("Could not extract any feature vector or class")
            sys.exit(1)


        if not len(data) == len(y):
            logger.error('number of features and classes are different')
            sys.exit()

        # convert to np.array
        X = np.array(data)

        # save features in pickle format
        if args.save_features:
            df_features = pd.DataFrame(X)
            df_features['class'] = y
            df_features.to_pickle(args.save_features)
            logger.info('Saved {} features and class to {}'.format(df_features.shape, args.save_features))

        # Train classifier

        if args.classify or args.learning_curve or args.testing_curve:

            # 1-NN
            if args.nearest_neighbors:
                clf = neighbors.KNeighborsClassifier(args.nearest_neighbors)
                clf_name = "{}NN".format(args.nearest_neighbors)
            elif args.logistic_regression:
                clf = linear_model.LogisticRegression()
                clf_name = "LogReg"
            elif args.svm_linear:
                clf = svm.LinearSVC()
                clf_name = "LinearSVM"
            elif args.svm_rbf:
                clf = svm.SVC(kernel='rbf', C=2.8, gamma=.0073)
                clf_name = "RBFSVM"
            elif args.optimize_nearest_neighbors:
                clf_name = "NN"
            elif args.optimize_svm_rbf:
                clf_name = "RBFSVM"
            else:
                logger.error('No classifier specified')
                sys.exit()
            logger.info("Classifier is {}".format(clf_name))




    if args.scale_features:
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

    run_train_scores = []
    run_test_scores =[]
    if args.classify:

        if args.optimize_nearest_neighbors:
            hyperoptize_knn(X_train, y_train, X_test, y_test,args.test_on_train)

        elif args.optimize_svm_rbf :
            hyperoptimze_rbf_svm(X_train, y_train, X_test, y_test,args.test_on_train)

        else:

            for run in range(args.run_number):

                train_acc,test_acc = train_test_classifier(clf,X_train, y_train, X_test, y_test,args.test_on_train)
                trainset_size = X_train.shape[0]

                if args.test_on_train:
                    logger.info("{} train size: {} train accuracy: {}".format(clf_name,trainset_size,train_acc))
                    run_train_scores.append(train_acc)
                logger.info("{} train size: {} test accuracy: {}".format(clf_name,trainset_size,test_acc))

                run_train_scores.append(train_acc)
                run_test_scores.append(test_acc)

                if args.run_number>1:
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

            if args.run_number>1:
                if args.test_on_train:
                    logger.info('Train scores : {} +/- {}'.format(np.mean(run_train_scores),np.std(run_train_scores)))
                logger.info('Test scores :: {} +/- {}'.format(np.mean(run_test_scores),np.std(run_test_scores)))


plt.show()