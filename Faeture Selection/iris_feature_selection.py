'''
    Univariate feature selection works by selecting the best features
     based on univariate statistical tests. It can be seen as a
     preprocessing step to an estimator.

     Random Forests is that they make it easy to measure
    the relative importance of each feature. from Random foreast feature_importances_
    we get the quality of important ::
            petal width > petal length > sepal length > sepal width
    so for better esimation or reduce the dimensionality, we can take it very
    special care.

    Using SelectKBest, we can select most import feature which have higher variance
    or higher information. Here the parameter K determine the number important feature
    you want as order of higher to lower variance.
    or Select features according to the k highest scores.
'''


import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import LabelEncoder

# make column_names
column_names = ['sepal length', 'sepal width', 'petal length', 'petal width', 'target']

'''
    Here Import data from csv file, but you can also use sklearn.dataset load_iris
'''

#import data from csv file
iris = pd.read_csv(r"C:\Users\JAHID\Desktop\ML DATA\iris_data.csv", header=None, names=column_names)

# create pandas dataframe
iris_data = pd.DataFrame(iris)

# feature columns
feature_cols = ['sepal length', 'sepal width', 'petal length', 'petal width']

# create instance of  feature and target
X = iris_data[feature_cols]
y = iris_data.target

# make target value(string) to  discreate value(int value)
y = LabelEncoder().fit_transform(y)

rf_clf = RandomForestClassifier(n_estimators=500, n_jobs=-1)
rf_clf.fit(X, y)

# determine  feature importance  of iris dataset
important_feature = rf_clf.feature_importances_
# predict a new instance
predict = rf_clf.predict([[5.1, 3.5, 1.4, 0.2]])



# create instance of SelectKBest with K = 2
KBest = SelectKBest(chi2, k=2)
# trnsform the iris feature with SelectKBest to get most two important feature
X_new = KBest.fit_transform(X, y)

# print shape of iris feature
print('before feature_selection : ',X.shape)  # before transform (original data)
print('before feature_selection : ',X_new.shape)  # after transform (transform data)
