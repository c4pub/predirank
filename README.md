# Predirank: Ranking Accuracy of Predictive Algorithms

## Description

This project ranks classifiers based on their accuracy across a selection of datasets. The primary focus is on classifiers from the sklearn/scikit-learn library and their application to popular tabular datasets. Sklearn classifiers require numerical data for operation. Consequently, categorical data must be 'numerized', or converted into numerical form using one-hot encoding.

The resulting rankings are only indicative; there can't be an absolute ranking. One classifier may perform better than others depending on the dataset/problem domain. The tool has many limitations, most of which are out of necessity to keep the task manageable. Among them:

- The classifiers are used with their default parameters.
- The size of the datasets is limited; those that are too large are randomly truncated to save space and execution time.

The datasets used in this project are derived from a selection of UCI datasets used in the paper:

> Fernandez-Delgado, Manuel, et al. "Do we need hundreds of classifiers to solve real world classification problems?" The Journal of Machine Learning Research 15.1 (2014): 3133-3181.

The format has been converted to "CSV". The large ones have been randomly truncated.

The rank score attempts to be a more robust indicator than the average accuracy. It limits the influence of outlier results. A rank score of 1.0 corresponds to a competitor that was better than all other competitors in all competitions. A competition ranks the accuracy of the prediction for a given training and test dataset. The number of competitions is given by the number of datasets multiplied by the number of iterations. Each dataset is randomly split into two halves for each iteration.

The ranking tracks exceptions: if a classifier generates an exception during the training or predictions, it is penalized with an accuracy of zero for that competition.

## Usage

The tool transforms tabular data (CSV files) into a numerical format that is compatible with sklearn classifiers. Additional classifiers can be added, provided they conform to the fit()/predict() class interface. The notebooks contain examples of how to wrap classifiers that do not conform to the sklearn interface.

* Run the Jupyter notebooks to see the classifiers in action
* Use `prdrnk_devtest.py` for a demo run in a console. Needs to be run from the `/src` directory of the project.

## Project Structure

* `/notebooks` - This directory contains the Jupyter notebooks which test the performance of the classifiers.
* `/src` - This directory contains the Python code necessary to run and rank the algorithms.
* `/data` - This directory contains the tabular datasets used.

## Results

Results of the classifier rankings are presented in the Jupyter notebooks.

* predirank_test_numerized.ipynb

    This notebook compares the sklearn classifiers and a few extra ones on the collection of tabular datasets. The attributes of the datasets are overwhelmingly numerical. The few that are categorical are 'numerized' using the one-hot encoding method.

* predirank_test_discretized.ipynb

    This notebook compares the same classifiers and datasets, but has an extra preprocessing step to simulate an execution with categorical/discretized data. The extra step consists in transforming numerical elements into zeros or ones, depending on whether the value is below or above the median for that attribute.

## Output

The following ranking resulted in a run performed on March 5, 2024 (predirank_test_numerized.ipynb):

```

--------------------------------------------------------------------------------
- - - - aggregate rank score

no  rank aggregate avg  stddev    excpt   classifier
--------------------------------------------------------------------------------
1   0.8166353383458647  0.149861> 0       CatBoostClassifier
2   0.7868421052631579  0.164188> 0       ExtraTreesClassifier()
3   0.7769736842105264  0.149596> 0       RandomForestClassifier()
4   0.7137531328320802  0.194009> 0       XGBClassifierWrap({})
5   0.7120300751879699  0.168401> 0       MLPClassifier()
6   0.6983709273182958  0.171396> 0       SVC()
7   0.6981516290726817  0.216143> 0       HistGradientBoostingClassifier()
8   0.6960839598997494  0.178327> 0       GradientBoostingClassifier()
9   0.6804511278195489  0.153307> 0       LogisticRegression()
10  0.6596491228070176  0.212654> 27      LogisticRegressionCV()
11  0.643170426065163   0.177667> 0       BaggingClassifier()
12  0.6278822055137845  0.186693> 0       DeodelSecond({})
13  0.6114974937343358  0.192538> 0       GaussianProcessClassifier()
14  0.6111842105263158  0.179586> 2       CalibratedClassifierCV()
15  0.6038533834586466  0.182666> 0       LinearSVC()
16  0.6016604010025063  0.211758> 0       LinearDiscriminantAnalysis()
17  0.6010651629072682  0.210583> 0       RidgeClassifierCV()
18  0.5890664160401002  0.209220> 0       RidgeClassifier()
19  0.5388471177944862  0.171131> 0       KNeighborsClassifier()
20  0.4980889724310777  0.219233> 0       DecisionTreeClassifier()
21  0.4887531328320802  0.230914> 0       AdaBoostClassifier()
22  0.48154761904761906 0.180675> 0       DeodataDelangaClassifier({})
23  0.4800125313283208  0.212593> 0       LabelSpreading()
24  0.47963659147869675 0.216330> 0       LabelPropagation()
25  0.4579573934837093  0.160635> 0       SGDClassifier()
26  0.42775689223057645 0.334809> 160     NuSVC()
27  0.4213032581453634  0.245235> 25      QuadraticDiscriminantAnalysis()
28  0.4184210526315789  0.223452> 0       BernoulliNB()
29  0.4106203007518797  0.229202> 0       GaussianNB()
30  0.4087406015037594  0.167799> 0       PassiveAggressiveClassifier()
31  0.4010651629072682  0.171779> 0       ExtraTreeClassifier()
32  0.3893170426065163  0.146816> 0       Perceptron()
33  0.3711152882205514  0.210556> 0       NearestCentroid()
34  0.19104010025062657 0.198842> 0       GaussianMixture()
35  0.19104010025062657 0.198842> 0       BayesianGaussianMixture()
36  0.15103383458646616 0.109409> 0       <<< Random Baseline >>>
37  0.06851503759398496 0.032042> 0       OneClassSVM()
38  0.05260025062656641 0.110408> 381     MultinomialNB()
39  0.04426691729323308 0.101659> 403     RadiusNeighborsClassifier()
--------------------------------------------------------------------------------

```
