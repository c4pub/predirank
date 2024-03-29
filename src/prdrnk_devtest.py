"""
predirank dev test

"""

#   c4pub@git 2024
#

import datetime
import random
import sys
import traceback

import predirank


# >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def GetSelClassifierListSk() :

    import sklearn.calibration
    import sklearn.discriminant_analysis
    import sklearn.ensemble
    import sklearn.gaussian_process
    import sklearn.linear_model
    import sklearn.mixture
    import sklearn.naive_bayes
    import sklearn.neighbors
    import sklearn.neural_network
    import sklearn.semi_supervised
    import sklearn.svm
    import sklearn.tree

    classifier_lst = [
                        # First term is classifier, second: alias name/id
                        [sklearn.calibration.CalibratedClassifierCV(), None],
                        [sklearn.discriminant_analysis.LinearDiscriminantAnalysis(), None],
                        [sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis(), None],
                        [sklearn.ensemble.AdaBoostClassifier(), None],
                        [sklearn.ensemble.BaggingClassifier(), None],
                        [sklearn.ensemble.ExtraTreesClassifier(), None],
                        [sklearn.ensemble.GradientBoostingClassifier(), None],
                        [sklearn.ensemble.HistGradientBoostingClassifier(), None],
                        [sklearn.ensemble.RandomForestClassifier(), None],
                        # [sklearn.ensemble.VotingClassifier(), None],
                        [sklearn.gaussian_process.GaussianProcessClassifier(), None],
                        [sklearn.linear_model.LogisticRegression(), None],
                        [sklearn.linear_model.LogisticRegressionCV(), None],
                        [sklearn.linear_model.PassiveAggressiveClassifier(), None],
                        [sklearn.linear_model.Perceptron(), None],
                        [sklearn.linear_model.RidgeClassifier(), None],
                        [sklearn.linear_model.RidgeClassifierCV(), None],
                        [sklearn.linear_model.SGDClassifier(), None],
                        [sklearn.mixture.BayesianGaussianMixture(), None],
                        [sklearn.mixture.GaussianMixture(), None],
                        # [sklearn.multiclass.OneVsOneClassifier(), None],
                        # [sklearn.multiclass.OneVsRestClassifier(), None],
                        # [sklearn.multiclass.OutputCodeClassifier(), None],
                        # [sklearn.multioutput.ClassifierChain(), None],
                        # [sklearn.multioutput.MultiOutputClassifier(), None],
                        [sklearn.naive_bayes.BernoulliNB(), None],
                        [sklearn.naive_bayes.GaussianNB(), None],
                        [sklearn.naive_bayes.MultinomialNB(), None],
                        [sklearn.neighbors.KNeighborsClassifier(), None],
                        [sklearn.neighbors.NearestCentroid(), None],
                        [sklearn.neighbors.RadiusNeighborsClassifier(), None],
                        [sklearn.neural_network.MLPClassifier(), None],
                        [sklearn.semi_supervised.LabelPropagation(), None],
                        [sklearn.semi_supervised.LabelSpreading(), None],
                        [sklearn.svm.LinearSVC(), None],
                        [sklearn.svm.NuSVC(), None],
                        [sklearn.svm.OneClassSVM(), None],
                        [sklearn.svm.SVC(), None],
                        [sklearn.tree.DecisionTreeClassifier(), None],
                        [sklearn.tree.ExtraTreeClassifier(), None],
                    ]

    return classifier_lst

# >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def GetClassifierListExtra() :

    import sklearn.neighbors
    # import deodel
    # import deodel2

    classifier_lst = [
                        # First term is classifier, second: alias name/id
                        [sklearn.neighbors.KNeighborsClassifier(n_neighbors=1), None],
                        # [deodel.DeodataDelangaClassifier(), None],
                        # [deodel2.DeodelSecond(), None],
                        [predirank.RandPredictor({'rand_seed': 42}), '<<< Random Baseline >>>'],
    ]
    return classifier_lst

# >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def GetCsvDataList() :

    csv_file_lst = [
        # First term is file name, second: index of target (-1/last by default)
        ["data/basic-abalone-sex.csv"],
        ["data/basic-adult-reduced.csv"],
        ["data/basic-breast-cancer-wisconsin.csv"],
        ["data/basic-cardiotocography.csv"],
        ["data/basic-credit-approval.csv"],
        ["data/basic-data_banknote_authentication.csv"],
        ["data/basic-diabetes.csv"],
        ["data/basic-digits.csv"],
        ["data/basic-ionosphere.data.csv"],
        ["data/basic-iris.csv"],
        ["data/basic-pima-indians-diabetes.csv"],
        ["data/basic-sonar.all-data.csv", -1],
        ["data/basic-titanic.csv", 1],
        ["data/basic-wheat-seeds.csv"],
        ["data/basic-wine.csv"],
        ["data/basic-winequality-white.csv"],

        ["data/dset-abalone.csv"],
        ["data/dset-acute-inflammation.csv"],
        ["data/dset-acute-nephritis.csv"],
        ["data/dset-adult_train-reduced.csv"],
        ["data/dset-annealing_train.csv"],
        ["data/dset-arrhythmia.csv"],
        ["data/dset-audiology-std_train.csv"],
        ["data/dset-audiology-std_train-patrons_repetidos.csv"],
        ["data/dset-balance-scale.csv"],
        ["data/dset-balloons.csv"],
        ["data/dset-bank.csv"],
        ["data/dset-blood.csv"],
        ["data/dset-breast-cancer.csv"],
        ["data/dset-breast-cancer-wisc.csv"],
        ["data/dset-breast-cancer-wisc-diag.csv"],
        ["data/dset-breast-cancer-wisc-prog.csv"],
        ["data/dset-breast-tissue.csv"],
        ["data/dset-car.csv"],
        ["data/dset-cardiotocography-10clases.csv"],
        ["data/dset-cardiotocography-3clases.csv"],
        ["data/dset-chess-krvk.csv"],
        ["data/dset-chess-krvkp.csv"],
        ["data/dset-congressional-voting.csv"],
        ["data/dset-conn-bench-sonar-mines-rocks.csv"],
        ["data/dset-conn-bench-vowel-deterding_train.csv"],
        ["data/dset-connect-4-reduced.csv"],
        ["data/dset-contrac.csv"],
        ["data/dset-credit-approval.csv"],
        ["data/dset-cylinder-bands.csv"],
        ["data/dset-dermatology.csv"],
        ["data/dset-echocardiogram.csv"],
        ["data/dset-ecoli.csv"],
        ["data/dset-energy-y1.csv"],
        ["data/dset-energy-y2.csv"],
        ["data/dset-fertility.csv"],
        ["data/dset-flags.csv"],
        ["data/dset-glass.csv"],
        ["data/dset-haberman-survival.csv"],
        ["data/dset-hayes-roth_train.csv"],
        ["data/dset-heart-cleveland.csv"],
        ["data/dset-heart-hungarian.csv"],
        ["data/dset-heart-switzerland.csv"],
        ["data/dset-heart-va.csv"],
        ["data/dset-hepatitis.csv"],
        ["data/dset-hill-valley_train.csv"],
        ["data/dset-horse-colic_train.csv"],
        ["data/dset-ilpd-indian-liver.csv"],
        ["data/dset-image-segmentation_test.csv"],
        ["data/dset-ionosphere.csv"],
        ["data/dset-iris.csv"],
        ["data/dset-led-display.csv"],
        ["data/dset-lenses.csv"],
        ["data/dset-letter-reduced.csv"],
        ["data/dset-libras.csv"],
        ["data/dset-low-res-spect.csv"],
        ["data/dset-lung-cancer.csv"],
        ["data/dset-lymphography.csv"],
        ["data/dset-magic.csv"],
        ["data/dset-mammographic.csv"],
        ["data/dset-miniboone-reduced.csv"],
        ["data/dset-molec-biol-promoter.csv"],
        ["data/dset-molec-biol-splice.csv"],
        ["data/dset-monks-1_test.csv"],
        ["data/dset-monks-2_test.csv"],
        ["data/dset-monks-3_test.csv"],
        ["data/dset-mushroom.csv"],
        ["data/dset-musk-1.csv"],
        ["data/dset-musk-2-reduced.csv"],
        ["data/dset-nursery.csv"],
        ["data/dset-oocytes_merluccius_nucleus_4d.csv"],
        ["data/dset-oocytes_merluccius_states_2f.csv"],
        ["data/dset-oocytes_trisopterus_nucleus_2f.csv"],
        ["data/dset-oocytes_trisopterus_states_5b.csv"],
        ["data/dset-optical_train.csv"],
        ["data/dset-ozone.csv"],
        ["data/dset-page-blocks.csv"],
        ["data/dset-parkinsons.csv"],
        ["data/dset-pendigits_train.csv"],
        ["data/dset-pima.csv"],
        ["data/dset-pittsburg-bridges-MATERIAL.csv"],
        ["data/dset-pittsburg-bridges-REL-L.csv"],
        ["data/dset-pittsburg-bridges-SPAN.csv"],
        ["data/dset-pittsburg-bridges-T-OR-D.csv"],
        ["data/dset-pittsburg-bridges-TYPE.csv"],
        ["data/dset-planning.csv"],
        ["data/dset-plant-margin.csv"],
        ["data/dset-plant-shape.csv"],
        ["data/dset-plant-texture.csv"],
        ["data/dset-post-operative.csv"],
        ["data/dset-primary-tumor.csv"],
        ["data/dset-ringnorm.csv"],
        ["data/dset-seeds.csv"],
        ["data/dset-semeion-reduced.csv"],
        ["data/dset-soybean_test.csv"],
        ["data/dset-spambase.csv"],
        ["data/dset-spect_test.csv"],
        ["data/dset-spectf_test.csv"],
        ["data/dset-statlog-australian-credit.csv"],
        ["data/dset-statlog-german-credit.csv"],
        ["data/dset-statlog-heart.csv"],
        ["data/dset-statlog-image.csv"],
        ["data/dset-statlog-landsat_train.csv"],
        ["data/dset-statlog-shuttle_train-reduced.csv"],
        ["data/dset-statlog-vehicle.csv"],
        ["data/dset-steel-plates.csv"],
        ["data/dset-synthetic-control.csv"],
        ["data/dset-teaching.csv"],
        ["data/dset-thyroid_train.csv"],
        ["data/dset-tic-tac-toe.csv"],
        ["data/dset-titanic.csv"],
        ["data/dset-trains.csv"],
        ["data/dset-twonorm.csv"],
        ["data/dset-vertebral-column-2clases.csv"],
        ["data/dset-vertebral-column-3clases.csv"],
        ["data/dset-vertebral-datos_orixinais-column_2C_weka.csv"],
        ["data/dset-vertebral-datos_orixinais-column_3C_weka.csv"],
        ["data/dset-wall-following.csv"],
        ["data/dset-waveform.csv"],
        ["data/dset-waveform-noise.csv"],
        ["data/dset-wine.csv"],
        ["data/dset-wine-quality-red.csv"],
        ["data/dset-wine-quality-white.csv"],
        ["data/dset-yeast.csv"],
        ["data/dset-zoo.csv"],
    ]
    return csv_file_lst

# >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def predirank_devtest() :

    print("- - - - - - - - - - - - - - - - - - ")
    print("- - - - - - - - - - - - - - - - - - ")
    print("- - - predirank_devtest")
    print()
    print("- - - - - - - - - - - - - - - - - - ")
    print("- - - Begin")
    print("- - - - - - - - - - - - - - - - - - ")
    print()

    classifier_sk_lst = GetSelClassifierListSk()
    classifier_extra_lst = GetClassifierListExtra()
    classifier_lst = classifier_sk_lst + classifier_extra_lst

    predictor_list = classifier_lst
    file_data_list = GetCsvDataList()

    # >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    data_location = "../"

    # >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    iter_no = 3

    # >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    random_seed = 42

    # >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    test_fraction = 0.5

    # >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    array_limit_row_max = 25
    array_limit_row_min = 12
    array_limit_col_max = 30

    # >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    data_process_mode = 'numeric'

    print("- - - - - - - - - - - - - - - - - - ")
    print("- - - data_process_mode:", data_process_mode)
    print("- - - - - - - - - - - - - - - - - - ")
    print()

    ret_data = predirank.BatchCsvAccuracyTest(predictor_list, file_data_list, data_location,
                                    iter_no, random_seed, test_fraction,
                                    data_process_mode, array_limit_row_max,
                                    array_limit_row_min, array_limit_col_max)
    # >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    data_process_mode = 'categ_sim_a'

    print("- - - - - - - - - - - - - - - - - - ")
    print("- - - data_process_mode:", data_process_mode)
    print("- - - - - - - - - - - - - - - - - - ")
    print()

    ret_data = predirank.BatchCsvAccuracyTest(predictor_list, file_data_list, data_location,
                                    iter_no, random_seed, test_fraction,
                                    data_process_mode, array_limit_row_max,
                                    array_limit_row_min, array_limit_col_max)
    # >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    data_process_mode = 'categ_sim_b'

    print("- - - - - - - - - - - - - - - - - - ")
    print("- - - data_process_mode:", data_process_mode)
    print("- - - - - - - - - - - - - - - - - - ")
    print()

    ret_data = predirank.BatchCsvAccuracyTest(predictor_list, file_data_list, data_location,
                                    iter_no, random_seed, test_fraction,
                                    data_process_mode, array_limit_row_max,
                                    array_limit_row_min, array_limit_col_max)
    # >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    print()
    print("- - - - - - - - - - - - - - - - - - ")
    print("- - - End")
    print("- - - - - - - - - - - - - - - - - - ")
    print()

    # import sys
    import sklearn

    print("- - - - - - - - - - - - - - - - - - - - - - - - - - - - ")
    print("- - - - python version:", str(sys.version))
    print("- - - - scikit-learn version:", str(sklearn.__version__))
    print("- - - - - - - - - - - - - - - - - - - - - - - - - - - - ")
    print()

    # >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    return ret_data

# >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

if __name__ == "__main__":

    ret = predirank_devtest()

# >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# >- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
