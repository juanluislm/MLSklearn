import pickle
import os
 
import numpy as np
import sklearn
from sklearn import cross_validation, grid_search
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib

from sklearn.multiclass import OneVsRestClassifier

try:
    from scripts.MacUtils import *
except:
    from MacUtils import *

import datetime
import argparse

def find_accuracy(model, X_test, y_test, tests, classes):

    print("entered")

    predictions = model.predict(X_test)

    results = np.zeros((tests+1, classes), dtype=float)

    # print(predictions)

    # we don't really need to store all of it, but just in case

    for i in range(0,tests):
        for j in range(0, classes):
            if predictions[i][j] == y_test[i][j]:
                # 1.0 means we classified the label test correctly. 0.0 means we did not
                results[i][j] = 1.0
                # This is what really matters. I am storing the rest of the info just in case
                results[tests][j]+=1.0

    # Time to find the accuracy for each label!
    # Storing the results in the last row

    overall = 0.0

    for i in range(0, classes):
        results[tests][i] = results[tests][i]/tests
        overall += results[tests][i]

    overall = overall/classes

    return overall, results

def find_best_model(path, labels, features, out_dir):

    scaler = StandardScaler()
    features = scaler.fit(features).transform(features)
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(features, labels, test_size=0.25)

    paths = os.listdir(path)

    classes = len(labels[0])

    tests = len(X_test)

    best_acc = 0.0
    best_model = paths[0]

    for file in paths:

        print(file)

        results = np.zeros((tests+1, classes), dtype=float)

        model = pickle.load( open(path+file, 'rb') )

        summary_file = out_dir+file + '_summary.p'

        try:

            overall, results = find_accuracy(model, X_test, y_test, tests, classes)

            if overall > best_acc:
                best_acc = overall
                best_model = file

            # write the summary file . It won't hurt

            sum_file = open(summary_file, 'wb')
            pickle.dump(results, sum_file)

            print(overall)

            sum_file.close()
        except:
            print("SOmething went wrong. Skipping ",file)

    decision_fname = out_dir+'best_model.p'

    decision_file = open(decision_fname, 'wb')

    pickle.dump( [best_model, best_acc], decision_file)

    print("done")

def find_best_model_load(data_path, path, out_dir):

    all_the_data = pickle.load(open(data_path, 'rb'))

    paths = os.listdir(path)

    X_test = all_the_data['x_test']

    y_test = all_the_data['y_test']

    classes = len(y_test[0])

    tests = len(X_test)

    best_acc = 0.0
    best_model = paths[0]

    for file in paths:

        print(file)

        model = pickle.load( open(path+file, 'rb') )

        summary_file = out_dir+file + '_summary.p'

        try:

            overall, results = find_accuracy(model, X_test, y_test, tests, classes)

            if overall > best_acc:
                best_acc = overall
                best_model = file

            # write the summary file . It won't hurt

            sum_file = open(summary_file, 'wb')
            pickle.dump(results, sum_file)

            print(overall)

            sum_file.close()
        except:
            print("SOmething went wrong. Skipping ",file)

    decision_fname = out_dir+'best_model.p'

    decision_file = open(decision_fname, 'wb')

    pickle.dump( [best_model, best_acc], decision_file)

    print("done")

def load_bottlenecks(path, unpickled, postfix):
    """

    :param path:
    :param filename:
    :return:
    """

    bottle_necks = []

    for idx in range(0, len(unpickled)):
        person = unpickled[idx][0].split()
        fname = '_'.join(person)+'_'+str(unpickled[idx][1]).zfill(4)+postfix
        m_data = open(path+'/'+fname, 'r')
        data = m_data.read().split(',')
        numbers = [float(data[i]) for i in range(0, len(data))]
        bottle_necks.append(numbers)

    return bottle_necks


def load_ground_truth_cache(labels, lines):
    class_count = len(labels)
    ground_truth_all = []
    for line in lines:
        ground_truth = [0]*class_count
        for i in range(class_count):
            if float(line[i]) > 0:
                ground_truth[i] = 1

        ground_truth_all.append(ground_truth)

    return ground_truth_all


def train_svm_classifer(features, labels, model_output_path):
    """
    train_svm_classifer will train a SVM, saved the trained and SVM model and
    report the classification performance
 
    features: array of input features
    labels: array of labels associated with the input features
    model_output_path: path for storing the trained svm model
    """
    # save 20% of data for performance evaluation
    scaler = StandardScaler()
    features = scaler.fit(features).transform(features)
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(features, labels, test_size=0.05)

    model_to_set = OneVsRestClassifier(SVC(probability=True))

    parameters = {
        "estimator__C": [1, 10, 100],
        "estimator__kernel": ["rbf"],
        "estimator__gamma": [1e-2, 1e-3, 1e-4],
    }

    model_tunning = grid_search.GridSearchCV(model_to_set, param_grid=parameters, scoring='accuracy', n_jobs=2)

    model_tunning.fit(X_test, y_test)

    print(model_tunning.best_score_)
    print(model_tunning.best_params_)



    # clf = OneVsRestClassifier(SVC(probability=True))
    # svm = SVC(probability=True, max_iter=100)
    # clf = OneVsRestClassifier(grid_search.GridSearchCV(svm, param, cv=10, verbose=0, n_jobs=6) )
    # clf.fit(X_test, y_test)

    # print(clf.n_jobs)


 
    # request probability estimation
    # svm = SVC(probability=True)
 
    # 10-fold cross validation, use 4 thread as each fold and each parameter set can be train in parallel
    # clf = grid_search.GridSearchCV(svm, param,
    #         cv=10, n_jobs=2, verbose=3)
    #
    # clf.fit(X_train, y_train)
 
    if os.path.exists(model_output_path):
        joblib.dump(model_tunning.best_estimator_, model_output_path)
    else:
        print("Cannot save trained svm model to {0}.".format(model_output_path))

    print("doing good!")

    return model_tunning
    # print("\nBest parameters set:")
    # print(clf.best_params_)
    #
    # y_predict=clf.predict(X_test)
    #
    # labels=sorted(list(set(labels)))
    # print("\nConfusion matrix:")
    # print("Labels: {0}\n".format(",".join(labels)))
    # print(confusion_matrix(y_test, y_predict, labels=labels))
    #
    # print("\nClassification report:")
    # print(classification_report(y_test, y_predict))


def train_svm_classifer2(features, labels, model_output_path, t_size, njobs, verbose):
    """
    train_svm_classifer will train a SVM, saved the trained and SVM model and
    report the classification performance

    features: array of input features
    labels: array of labels associated with the input features
    model_output_path: path for storing the trained svm model
    """
    # save 20% of data for performance evaluation
    scaler = StandardScaler()
    features = scaler.fit(features).transform(features)
    #print(features[0])
    # scaler = MinMaxScaler()
    # features = scaler.fit_transform(features)#scaler.fit(features).transform(features)
    # #
    # # #print(features[0])
    # #
    # # # X_train, X_test, y_train, y_test = cross_validation.train_test_split(features, labels, test_size=0.20)
    # #
    # for i in range(0, len(features)):
    #     for j in range(0, len(features[0])):
    #         if(features[i][j] > 0.9999 or features[i][j] < -1.000):
    #             print(i,j,features[i][j],"Aaaaaaaa")
    #             features[i][j] = 0.999999

    X_train, X_test, y_train, y_test = cross_validation.train_test_split(features, labels, test_size=t_size)

    out_data = open(model_output_path+'data_.p','wb')
    data_to_save = {'x_train': X_train, 'y_train':y_train, 'x_test':X_test, 'y_test':y_test}

    pickle.dump(data_to_save, out_data)
    out_data.close()

    best_model = None
    best_acc = 0
    kernels = ['linear']

    gammas = [1e-2, 1e-3, 1e-4]
    Cs = [1, 10, 100, 1000, 10000]

    classes = len(y_test[0])
    tests = len(X_test)

    for mC in Cs:
        for mgamma in gammas:
            for mkernel in kernels:
                model_to_set = OneVsRestClassifier(
                    SVC( probability=False,
                         kernel='rbf',
                         C=mC, gamma=mgamma,
                         verbose=verbose),
                    n_jobs=njobs)

                print("gamma", mgamma, "C", mC)
                model_to_set.fit(X_train, y_train)
                print("gamma", mgamma, "C",mC)

                overall, results = find_accuracy(model_to_set, X_test, y_test, tests, classes)

                print('overall_acc_',overall)

                model_name = model_output_path +'_gamma_'+str(mgamma)+'_C_'+str(mC)+'_kernel_'+mkernel+'_overall_acc_'+str(overall)+'.p'
                # outfile = open(model_name,'wb')
                # pickle.dump(model_to_set,outfile)
                # outfile.close()
                pickle_dump(model_to_set, model_name)

                summary_file = model_output_path +'_gamma_'+str(mgamma)+'_C_'+str(mC)+'_kernel_'+mkernel+'_results.p'

                pickle_dump(results, summary_file)

def prune_data2(keys, lines, excluded_keys={}):
    new_keys = keys[2:]
    to_remove = []

    for i in range(len(new_keys) - 1, 0, -1):
        if (new_keys[i] in excluded_keys):
            new_keys.pop(i)
            to_remove.append(i)

    new_lines = []
    for i in range(0, len(lines)):
        new_line = lines[i][2:]

        for rem in range(len(to_remove) - 1, 0, -1):
            new_line.pop(rem)

        new_lines.append(new_line)

    return new_keys, new_lines

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--jobs',
        type=int,
        default=12,
        help='Number of threads to be fired during training.'
    )
    parser.add_argument(
        '--test_size',
        type=float,
        default=0.8,
        help='% of images used for testing.'
    )
    parser.add_argument(
        '--bottleneck_path',
        type=str,
        default='/home/withme/dev/tensorflow-for-poets-2/tf_files/bottlenecks_funneled/lfw_all',
        help='Path to the bottlenecks '
    )
    parser.add_argument(
        '--output_prefix',
        type=str,
        default='/home/withme/dev/tensorflow-for-poets-2/scripts/svm_',
        help='Prefix of the folder where the models and data will be saved.'
    )

    parser.add_argument(
        '--header_lines',
        type=str,
        default='lfw_header_lines.p',
        help='Path to the header lines pickle file.'
    )
    parser.add_argument(
        '--verbose',
        type=bool,
        default=True,
        help='Verbose? Are you sure?.'
    )

    FLAGS = parser.parse_args()

    fname = FLAGS.header_lines
    keys_lines = pickle.load(open(fname, 'rb'))
    keys = keys_lines['header']
    lines = keys_lines['lines']
    # included_keys = ['male', 'asianindian', 'eastasian', 'african', 'latino', 'caucasian']
    # keys, lines = prune_data(keys, lines, excluded_keys = ['Male']) # duplicate keys

    # print('=================================================',len(keys))
    # print('=================================================', len(lines))
    #
    # lines_trans = np.array(lines).transpose()
    #
    # lines = lines_trans.tolist()
    # keys2 = [x for x in range(0,len(keys))]
    # path1 = '/data/lfw/lfw_all_funneled_face_crop_l_0.3_r_0.3_t_0.4_d_0.2'
    # path = '/home/withme/dev/tensorflow-for-poets-2/tf_files/bottlenecks_funneled/lfw_all'
    path = FLAGS.bottleneck_path
    postfix = '.jpg_mobilenet_0.50_224.txt'

    bottlenecks = load_bottlenecks(path, lines, postfix)

    # new_keys, new_lines = prune_data(keys, lines, excluded_keys=['Male'])

    # keys, lines = prune_data(keys, lines, excluded_keys=['Male'])  # duplicate keys

    keys, lines = prune_data2(keys, lines, {'Male':1})

    ground_truth = load_ground_truth_cache(keys, lines)
    test_size = FLAGS.test_size

    out_path = FLAGS.output_prefix+str(test_size)+'_'+str(datetime.datetime.now())

    os.makedirs(out_path)
    jobs = 32


    clf = train_svm_classifer2(np.array(bottlenecks), np.array(ground_truth),
                               out_path+'/svm_', test_size, FLAGS.jobs, FLAGS.verbose)

    print("done")

