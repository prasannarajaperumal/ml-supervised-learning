# Load libraries
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import export_graphviz
from six import StringIO
from IPython.display import Image
from timeit import default_timer as timer
from sklearn.model_selection import cross_val_score
import pydotplus


feature_cols = ['Employment',
                'Education',
                'Marital Status',
                'Position',
                'Family Status',
                'Ethnicity',
                'Sex',
                'Country']

# Define the column names
col_names = ['Employment',
             'Education',
             'Marital Status',
             'Position',
             'Family Status',
             'Ethnicity',
             'Sex',
             'Country',
             'Salary']


def prepare_dataset():
    # load dataset
    salary_df = pd.read_csv("data/adult-salary-data.csv", header=None, names=col_names)

    # Clean dataset - Drop all the rows with unknown (?) value
    salary_df['Country'] = salary_df['Country'].replace(' ?', np.nan)
    salary_df['Employment'] = salary_df['Employment'].replace(' ?', np.nan)
    salary_df['Position'] = salary_df['Position'].replace(' ?', np.nan)
    salary_df['Salary'] = salary_df['Salary'].replace(' ?', np.nan)
    salary_df.dropna(how='any', inplace=True)

    # Ensure the column values are encoded into float - most columns are very low dimension and hence fits well with Label encoder
    for col_name in col_names:
        str_column = salary_df[col_name]
        lb = LabelEncoder()
        salary_df[col_name] = lb.fit_transform(str_column)

    return salary_df


def split_dataset(test_size, random_state):
    # Define the feature columns
    feature_cols = ['Employment',
                    'Education',
                    'Marital Status',
                    'Position',
                    'Family Status',
                    'Ethnicity',
                    'Sex',
                    'Country']
    feature_values = salary_df[feature_cols]
    feature_output = salary_df["Salary"]

    return train_test_split(feature_values, feature_output, test_size=test_size, random_state=random_state)


def get_decision_tree_classifiers(only_default):
    default_clf = DecisionTreeClassifier()
    if only_default:
        return [("default_dt_classifier", default_clf)]  # used to find the splits with default classifier
    max_depth_3_prune_10_clf = DecisionTreeClassifier(min_samples_leaf=10, max_depth=3)
    max_depth_3_prune_100_clf = DecisionTreeClassifier(min_samples_leaf=100, max_depth=3)
    max_depth_5_prune_10_clf = DecisionTreeClassifier(min_samples_leaf=10, max_depth=5)
    max_depth_5_prune_100_clf = DecisionTreeClassifier(min_samples_leaf=100, max_depth=5)
    max_depth_10_prune_10_clf = DecisionTreeClassifier(min_samples_leaf=10, max_depth=10)
    max_depth_10_prune_100_clf = DecisionTreeClassifier(min_samples_leaf=100, max_depth=10)
    return [
        ("max_depth_3_prune_10", max_depth_3_prune_10_clf),
        ("max_depth_3_prune_100", max_depth_3_prune_100_clf),
        ("max_depth_5_prune_10", max_depth_5_prune_10_clf),
        ("max_depth_5_prune_100", max_depth_5_prune_100_clf),
        ("max_depth_10_prune_10", max_depth_10_prune_10_clf),
        ("max_depth_10_prune_100", max_depth_10_prune_100_clf)]


def get_neural_network_classifiers(only_default):
    default_clf = MLPClassifier()
    if only_default:
        return [("default_nn_classifier", default_clf)]  # used to find the splits with default classifier
    wide_001_alpha_clf = MLPClassifier(alpha=0.01, hidden_layer_sizes=(100,))
    wide_01_alpha_clf = MLPClassifier(alpha=0.1, hidden_layer_sizes=(100,))
    wide_1_alpha_clf = MLPClassifier(alpha=1.0, hidden_layer_sizes=(100,))
    narrow_001_alpha_clf = MLPClassifier(alpha=0.01, hidden_layer_sizes=(10, 10, 10))
    narrow_01_alpha_clf = MLPClassifier(alpha=0.1, hidden_layer_sizes=(10, 10, 10))
    narrow_1_alpha_clf = MLPClassifier(alpha=1.0, hidden_layer_sizes=(10, 10, 10))
    return [
        ("wide_001_alpha", wide_001_alpha_clf),
        ("wide_01_alpha", wide_01_alpha_clf),
        ("wide_1_alpha", wide_1_alpha_clf),
        ("narrow_001_alpha", narrow_001_alpha_clf),
        ("narrow_01_alpha", narrow_01_alpha_clf),
        ("narrow_1_alpha", narrow_1_alpha_clf)]


def get_boosting_classifiers(only_default):
    default_clf = AdaBoostClassifier()
    if only_default:
        return [("default_boosting_classifier", default_clf)]  # used to find the splits with default classifier
    estimator_1_rate_1_clf = AdaBoostClassifier(n_estimators=1, learning_rate=1)
    estimator_1_rate_10_clf = AdaBoostClassifier(n_estimators=1, learning_rate=10)
    estimator_10_rate_1_clf = AdaBoostClassifier(n_estimators=10, learning_rate=1)
    estimator_10_rate_10_clf = AdaBoostClassifier(n_estimators=10, learning_rate=10)
    estimator_100_rate_1_clf = AdaBoostClassifier(n_estimators=100, learning_rate=1)
    estimator_100_rate_10_clf = AdaBoostClassifier(n_estimators=100, learning_rate=10)
    return [
        ("estimator_1_rate_1", estimator_1_rate_1_clf),
        ("estimator_1_rate_10", estimator_1_rate_10_clf),
        ("estimator_10_rate_1", estimator_10_rate_1_clf),
        ("estimator_10_rate_10", estimator_10_rate_10_clf),
        ("estimator_100_rate_1", estimator_100_rate_1_clf),
        ("estimator_100_rate_10", estimator_100_rate_10_clf)]


def get_svm_classifiers(only_default):
    default_clf = SVC()
    if only_default:
        return [("default_svm_classifier", default_clf)]  # used to find the splits with default classifier
    linear_c_1_clf = SVC(C=1.0, kernel='linear')
    linear_c_5_clf = SVC(C=5.0, kernel='linear')
    linear_c_10_clf = SVC(C=10.0, kernel='linear')
    rbf_c_1_clf = SVC(C=1.0, kernel='rbf')
    rbf_c_5_clf = SVC(C=5.0, kernel='rbf')
    rbf_c_10_clf = SVC(C=10.0, kernel='rbf')
    return [
        ("linear_c_1", linear_c_1_clf),
        ("linear_c_5", linear_c_5_clf),
        ("linear_c_10", linear_c_10_clf),
        ("rbf_c_1", rbf_c_1_clf),
        ("rbf_c_5", rbf_c_5_clf),
        ("rbf_c_10", rbf_c_10_clf)]


def get_nearest_k_classifiers(only_default):
    default_clf = KNeighborsClassifier()
    if only_default:
        return [("default_classifier", default_clf)]  # used to find the splits with default classifier
    k_3_leaf_30_algo_kd_clf = KNeighborsClassifier(n_neighbors=3, leaf_size=30)
    k_3_leaf_100_algo_kd_clf = KNeighborsClassifier(n_neighbors=3, leaf_size=100)
    k_3_leaf_500_algo_kd_clf = KNeighborsClassifier(n_neighbors=3, leaf_size=500)
    k_5_leaf_30_algo_kd_clf = KNeighborsClassifier(n_neighbors=5, leaf_size=30)
    k_5_leaf_100_algo_kd_clf = KNeighborsClassifier(n_neighbors=5, leaf_size=100)
    k_5_leaf_500_algo_kd_clf = KNeighborsClassifier(n_neighbors=5, leaf_size=500)
    k_10_leaf_30_algo_kd_clf = KNeighborsClassifier(n_neighbors=10, leaf_size=30)
    k_10_leaf_100_algo_kd_clf = KNeighborsClassifier(n_neighbors=10, leaf_size=100)
    k_10_leaf_500_algo_kd_clf = KNeighborsClassifier(n_neighbors=10, leaf_size=500)

    return [
        ("k_3_leaf_30_algo_kd_clf", k_3_leaf_30_algo_kd_clf),
        ("k_3_leaf_100_algo_kd_clf", k_3_leaf_100_algo_kd_clf),
        ("k_3_leaf_500_algo_kd_clf", k_3_leaf_500_algo_kd_clf),
        ("k_5_leaf_30_algo_kd_clf", k_5_leaf_30_algo_kd_clf),
        ("k_5_leaf_100_algo_kd_clf", k_5_leaf_100_algo_kd_clf),
        ("k_5_leaf_500_algo_kd_clf", k_5_leaf_500_algo_kd_clf),
        ("k_10_leaf_30_algo_kd_clf", k_10_leaf_30_algo_kd_clf),
        ("k_10_leaf_100_algo_kd_clf", k_10_leaf_100_algo_kd_clf),
        ("k_10_leaf_500_algo_kd_clf", k_10_leaf_500_algo_kd_clf)]


def timed_classifier_fit(classifier, training_features, training_output, method_name):
    start = timer()
    classifier = classifier.fit(training_features, training_output)
    end = timer()
    print(method_name + ":training_time:" + str(end - start))
    return end - start, classifier


def timed_classifier_predict(classifier, testing_features, method_name):
    start = timer()
    prediction = classifier.predict(testing_features)
    end = timer()
    print(method_name + ":testing_time:" + str(end - start))
    return end - start, prediction


def cross_validate(classifier, training_features, training_output, folds, method_name):
    scores = cross_val_score(classifier, training_features, training_output, cv=folds)
    print(method_name + ":%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
    return classifier, scores.mean(), scores.std()


def calculate_classifier_score(clf, mean, std, classifier_name):
    fit = timed_classifier_fit(clf, training_features, training_output, classifier_name)
    prediction = timed_classifier_predict(fit[1], testing_features, classifier_name)
    # manually plugin the normalization and weights. Training time with default / Prediction time with default.
    # TODO - automate this
    # return 0.6 * mean + (0.1 * (1 - (fit[0] / 0.005))) + (0.3 * (1 - (prediction[0] / 0.36))) # K-neighbours
    # return 0.6 * mean + (0.3 * (1 - (fit[0] / 5))) + (0.1 * (1 - (prediction[0] / 0.0044)))  # Neural network
    # return 0.8 * mean + (0.1 * (1 - (fit[0] / 0.5))) + (0.1 * (1 - (prediction[0] / 0.04)))  # ADA Boosting
    return 0.6 * mean + (0.3 * (1 - (fit[0] / 14))) + (0.1 * (1 - (prediction[0] / 3)))  # SVM


def run_best_classifier(classifiers, training_features, training_output):
    best_classifier = None
    best_classifier_score = 0
    best_classifier_name = None
    for classifier in classifiers:
        clf, mean, std = cross_validate(classifier[1], training_features, training_output, 5, classifier[0])
        classifier_score = calculate_classifier_score(clf, mean, std, classifier[0])
        print(classifier[0] + ":classifier_score:" + str(classifier_score))
        if classifier_score > best_classifier_score:
            print("new best classifier " + classifier[0])
            best_classifier = clf
            best_classifier_name = classifier[0]
            best_classifier_score = classifier_score
    return best_classifier, best_classifier_name, best_classifier_score


def find_best_split(classifiers, training_features, testing_features, training_output, testing_output):
    for classifier in classifiers:
        classifier_fit = timed_classifier_fit(classifier[1], training_features, training_output, classifier[0])[1]
        prediction = timed_classifier_predict(classifier_fit, testing_features, classifier[0])[1]
        print(classifier[0] + ":accuracy:", metrics.accuracy_score(testing_output, prediction))


def decision_tree(training_features, testing_features, training_output, testing_output, export_image, only_default):
    classifiers = get_decision_tree_classifiers(only_default)
    # find_best_split(classifiers, training_features, testing_features, training_output, testing_output)
    best_classifier, best_classifier_name, best_classifier_score = run_best_classifier(classifiers, training_features, training_output)
    best_classifier = timed_classifier_fit(best_classifier, training_features, training_output, best_classifier_name)[1]
    prediction = timed_classifier_predict(best_classifier, testing_features, best_classifier_name)[1]
    print("accuracy:", metrics.accuracy_score(testing_output, prediction))
    if export_image:
        export_as_image(best_classifier, feature_cols, "data/" + str(best_classifier_name + "-tree.png"))


def neural_network(training_features, testing_features, training_output, testing_output, only_default):
    classifiers = get_neural_network_classifiers(only_default)
    # find_best_split(classifiers, training_features, testing_features, training_output, testing_output)
    best_classifier, best_classifier_name, best_classifier_score = run_best_classifier(classifiers, training_features, training_output)
    best_classifier = timed_classifier_fit(best_classifier, training_features, training_output, best_classifier_name)[1]
    prediction = timed_classifier_predict(best_classifier, testing_features, best_classifier_name)[1]
    print("accuracy:", metrics.accuracy_score(testing_output, prediction))


def boosting(training_features, testing_features, training_output, testing_output, only_default):
    classifiers = get_boosting_classifiers(only_default)
    # find_best_split(classifiers, training_features, testing_features, training_output, testing_output)
    best_classifier, best_classifier_name, best_classifier_score = run_best_classifier(classifiers, training_features, training_output)
    best_classifier = timed_classifier_fit(best_classifier, training_features, training_output, best_classifier_name)[1]
    prediction = timed_classifier_predict(best_classifier, testing_features, best_classifier_name)[1]
    print("accuracy:", metrics.accuracy_score(testing_output, prediction))


def svm(training_features, testing_features, training_output, testing_output, only_default):
    classifiers = get_svm_classifiers(only_default)
    # find_best_split(classifiers, training_features, testing_features, training_output, testing_output)
    best_classifier, best_classifier_name, best_classifier_score = run_best_classifier(classifiers, training_features, training_output)
    best_classifier = timed_classifier_fit(best_classifier, training_features, training_output, best_classifier_name)[1]
    prediction = timed_classifier_predict(best_classifier, testing_features, best_classifier_name)[1]
    print("accuracy:", metrics.accuracy_score(testing_output, prediction))


def nearest_k(training_features, testing_features, training_output, testing_output, only_default):
    classifiers = get_nearest_k_classifiers(only_default)
    # find_best_split(classifiers, training_features, testing_features, training_output, testing_output)
    best_classifier, best_classifier_name, best_classifier_score = run_best_classifier(classifiers, training_features, training_output)
    print("Best classifier:" + best_classifier_name)
    best_classifier = timed_classifier_fit(best_classifier, training_features, training_output, best_classifier_name)[1]
    prediction = timed_classifier_predict(best_classifier, testing_features, best_classifier_name)[1]
    print("accuracy:", metrics.accuracy_score(testing_output, prediction))


def export_as_image(classifier, feature_cols, file_name):
    dot_data = StringIO()
    export_graphviz(classifier,
                    out_file=dot_data,
                    filled=True,
                    rounded=True,
                    special_characters=True,
                    feature_names=feature_cols,
                    class_names=['0', '1'])

    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_png(file_name)
    Image(graph.create_png())


if __name__ == '__main__':
    salary_df = prepare_dataset()

    # Find the best split
    # for split in [0.2, 0.25, 0.3, 0.35, 0.4, 0.5]:
    #     training_features, testing_features, training_output, testing_output = split_dataset(split, 1)
    #     print("Split:" + str(split))
    #     # decision_tree(training_features, testing_features, training_output, testing_output, False, True)
    #     # nearest_k(training_features, testing_features, training_output, testing_output, True)
    #     # neural_network(training_features, testing_features, training_output, testing_output, True)
    #     boosting(training_features, testing_features, training_output, testing_output, True)
    #     # svm(training_features, testing_features, training_output, testing_output, True)

    chosen_split = 0.35
    training_features, testing_features, training_output, testing_output = split_dataset(chosen_split, 1)
    # decision_tree(training_features, testing_features, training_output, testing_output, False, False)
    # nearest_k(training_features, testing_features, training_output, testing_output, False)
    # neural_network(training_features, testing_features, training_output, testing_output, False)
    # boosting(training_features, testing_features, training_output, testing_output, False)
    # svm(training_features, testing_features, training_output, testing_output, False)
