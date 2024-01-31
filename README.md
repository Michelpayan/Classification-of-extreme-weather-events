# Classification-of-extreme-weather-events

This project presents the results of the [Kaggle I competition for IFT6390 Fall 2023 course at Université de Montréal](https://www.kaggle.com/competitions/classification-of-extreme-weather-events-udem). The competition is about detection of extreme
weather events from atmospherical data and it's goal is to design a machine learning algorithm that can automatically classify whether a set of climate variables corresponding to a
time point and location (latitude and longitude) is associated to i) standard (background)
conditions, ii) a tropical cyclone or iii) an atmospheric river.

For this specific project, we were asked to build a logistic regression classifier from scratch to and beat the baselines highlighted in
the leaderboard. These baselines are:
• a dummy classifier initialized with random parameters.
• a logistic regression classifier.
• the TA’s best baseline.

Besides the logistic regression, we have to compare our results and performance using other models. All the scripts are contained in the "Code" folder, where you will see the following notebooks and python files:

1. Visualizations: it is a jupyter notebook that created and saved the plots that are included in the report.

2. Feature_modifications: it is a python file that contains two functions that are used in the "Final code" notebook which creates new columns, such as the season column and applies the non-linear transformation on the given columns using degree 2 polynomial.

3. Multiclass_LogisticReg and MultiLR_compatible: each one is a python file that contains the logistic regression model built from scratch. The main difference is that the last one inherits from the BaseEstimator, ClassifierMixin classes of scikit-learn in order to make it compatible with other libraries such as gridSearchCV. Therefore, the MultiLR_compatible was the one used in the final project.

4.  Final code: this is a jupyter notebook that contains all the process followed in order to get the best classifier. It uses the Feature_modifications and MultiLR_compatible. Furthermore it saves the predictions made by each estimator and the model itself.

In order to run the Final code notebook, it was important to install some libraries, outside of the packages included by default on sklearn such as:

* pip install xgboost
* pip install imbalanced-learn

Another important thing to note is that the cells in the Final code notebook must be run in sequence, since there are many transformations in the data set that depends of the previous modification made on it.
