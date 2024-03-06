# Predirank: Ranking Accuracy of Predictive Algorithms

## Description

This project ranks classifiers based on their accuracy across a selection of datasets. The primary focus is on classifiers from the sklearn/scikit-learn library and their application to popular tabular datasets. Sklearn classifiers require numerical data for operation. Consequently, categorical data must be 'numerized', or converted into numerical form using one-hot encoding.

The resulting rankings are only indicative; there can't be an absolute ranking. One classifier may perform better than others depending on the dataset/problem domain. The tool has many limitations, most of which are out of necessity to keep the task manageable. Among them:

- The classifiers are used with their default parameters.
- The size of the datasets is limited; those that are too large are randomly truncated to save space and execution time.

The notebook examples in this project are based on a selection of UCI datasets used in the paper:

> Fernandez-Delgado, Manuel, et al. "Do we need hundreds of classifiers to solve real world classification problems?" The Journal of Machine Learning Research 15.1 (2014): 3133-3181.

The format has been converted to "CSV". The larger ones have been randomly truncated.

The rank score attempts to be a more robust indicator than the average accuracy. It limits the influence of outlier results. A rank score of 1.0 corresponds to a competitor that was better than all other competitors in all competitions. A competition consists in ranking the accuracy of the prediction for a given training and test dataset. The number of competitions is given by the number of datasets multiplied by the number of iterations. The dataset is randomly split into two halves for each iteration.

The ranking tracks exceptions: if a classifier generates an exception during the training or predictions, it is penalized with an accuracy of zero for that competition.

## Usage

The tool transforms tabular data (CSV files) into a numerical format that is compatible with sklearn classifiers. Additional classifiers can be added, provided they conform to the fit()/predict() class interface. The notebooks contain examples of how to wrap classifiers that do not conform to the sklearn interface.

* Run the Jupyter notebooks to see the classifiers in action
* Use `prdrnk_devtest.py` for a demo run in a console. Needs to be run from the `/src` directory of the project.

## Project Structure

* `/notebooks` - This directory contains the Jupyter notebooks which demonstrate the usage and performance of the classifiers.
* `/src` - This directory contains the Python code necessary to run and rank the algorithms.
* `/data` - This directory contains the tabular datasets used.

## Results

Results of the classifier rankings are presented in the Jupyter notebooks.

* predirank_test_numerized.ipynb

    This notebook compares the sklearn classifiers and a few extra ones on the collection of tabular datasets. The attributes of the datasets are overwhelmingly numerical. The few that are categorical are 'numerized' using the one-hot encoding method.

* predirank_test_discretized.ipynb

    This notebook compares the same classifiers and datasets, but has an extra preprocessing step to simulate an execution with categorical/discretized data. The extra step consists in transforming numerical elements into zeros or ones, depending on whether the value is below or above the median for that attribute.
