#!/usr/bin/python

#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")
import tester

import tester
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from time import time
from sklearn.cross_validation import train_test_split
from sklearn import linear_model
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from tabulate import tabulate
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedShuffleSplit

from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier  #GBM algorithm
from sklearn.linear_model import LogisticRegression
from xgboost.sklearn import XGBClassifier
import xgboost as xgb
#%matplotlib inline

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
target_list = ['poi']
report_list = ['salary','deferral_payments', 'total_payments',
               'loan_advances', 'bonus', 'restricted_stock_deferred', 
               'deferred_income', 'total_stock_value', 'expenses',
               'exercised_stock_options', 'other', 'long_term_incentive', 
               'restricted_stock', 'director_fees','to_messages',
               'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 
               'shared_receipt_with_poi']
features_list =  target_list + report_list

### Store to my_dataset for easy export below.
my_dataset = data_dict
counter = 0
for entry in my_dataset:
    if my_dataset[entry]['poi']:
        counter += 1

print 'Number of data points in this dataset: %f' %len(my_dataset)
print 'Number of POI in this dataset: %f' %counter
print 'Number of feature in this dataset: %f' %len(features_list)

### Task 2: Remove outliers
### Determine missing values in Enron report
value_report_list = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
dict_report_list = dict(zip(report_list,value_report_list))
for entry in my_dataset:
    for name in report_list:
        val = my_dataset[entry][name]
        if val == 'NaN':
            dict_report_list[name] += 1
NaN_table = pd.DataFrame(dict_report_list.items(), columns=['Feature Name', 'No of Missing Values'])
print NaN_table.sort(columns = 'No of Missing Values',ascending=False)

outlier_list = []
#### Next script is used to remove observation which have all value 'NaN' in Enron report. 
for entry in my_dataset:
    for name in report_list:
        val = my_dataset[entry][name]
        if val == 'NaN':
            if entry not in outlier_list:
                outlier_list.append(entry)
        elif val != 'NaN':
            try:
                outlier_list.remove(entry)
                break
            except ValueError:
                break

outlier_list.append('TOTAL') #insert 'TOTAL' observation
outlier_list.append('THE TRAVEL AGENCY IN THE PARK')
print 'The outlier list is: '
print outlier_list

#### Remove outlier
for name in outlier_list:
    my_dataset.pop(name,0)
print 'Number of data points after outlier removing in this dataset: %f' %len(my_dataset)


### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
#### Making new feature based on other features.
def computeFraction( poi_messages, all_messages ):
    """ given a number messages to/from POI (numerator) 
        and number of all messages to/from a person (denominator),
        return the fraction of messages to/from that person
        that are from/to a POI
   """
   
    ### you fill in this code, so that it returns either
    ###     the fraction of all messages to this person that come from POIs
    ###     or
    ###     the fraction of all messages from this person that are sent to POIs
    ### the same code can be used to compute either quantity

    ### beware of "NaN" when there is no known email address (and so
    ### no filled email features), and integer division!
    ### in case of poi_messages or all_messages having "NaN" value, return 0.
    if poi_messages == "NaN":
        fraction = 0.
    elif all_messages == "NaN":
        fraction = 0.
    else:
        fraction = poi_messages*1.0/all_messages

    return fraction

for name in my_dataset:
    data_point = my_dataset[name]
    from_poi_to_this_person = data_point["from_poi_to_this_person"]
    to_messages = data_point["to_messages"]
    fraction_from_poi = computeFraction( from_poi_to_this_person, to_messages )
    my_dataset[name]["fraction_from_poi"] = fraction_from_poi

    from_this_person_to_poi = data_point["from_this_person_to_poi"]
    from_messages = data_point["from_messages"]
    fraction_to_poi = computeFraction( from_this_person_to_poi, from_messages )
    my_dataset[name]["fraction_to_poi"] = fraction_to_poi

### Feature list with addtional features
features_list = features_list + ['fraction_from_poi','fraction_to_poi']
print features_list

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Using SelectKBest to determine number of selection feature.
SKB = SelectKBest()
SKB.fit(features,labels)
scores = SKB.scores_
unsorted_pairs = zip(features_list[1:], scores)
sorted_pairs = list(reversed(sorted(unsorted_pairs, key=lambda x: x[1])))
KBest_table = pd.DataFrame(sorted_pairs,columns=['Feature','Score'])
print KBest_table

sns.barplot(KBest_table['Feature'],KBest_table['Score'])
_, labels = plt.xticks()
plt.setp(labels, rotation=90)
plt.ylabel('SKBest score')
plt.title('Classifier')
plt.show()

### Validate the original feature list in Naive Bayes classifier
old_features_list = target_list + list(KBest_table['Feature'].iloc[:6])
old_features_list.remove('fraction_to_poi')
print old_features_list
# Re-create the data for model fitting
data = featureFormat(my_dataset, old_features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
# Example starting point. Create train and test set from original dataset. 30% of data goes into test set.
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)
    
clf_NB = GaussianNB()
clf_NB.fit(features_train,labels_train)
labels_pred = clf_NB.predict(features_test)
print classification_report(labels_test, labels_pred)

### Validate the engineered feature list in Naive Bayes classifier
new_features_list = target_list + list(KBest_table['Feature'].iloc[:6])
print new_features_list
# Re-create the data for model fitting
data = featureFormat(my_dataset, new_features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
# Example starting point. Create train and test set from original dataset. 30% of data goes into test set.
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

print tabulate([['Original List', 0.33, 0.33], ['Engineered list', 0.60, 0.50]], headers=['Name', 'Precision','Recall'])


data = featureFormat(my_dataset, new_features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
# Example starting point. Create train and test set from original dataset. 30% of data goes into test set.
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

#### Reshape the training and test sets.
features_train = np.array(features_train)
labels_train = np.array(labels_train)
features_test = np.array(features_test)
labels_test = np.array(labels_test)


### Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

#### Checking the final result with all ML that are used for classification:
#### Using GridSearchCV to optimize number of features + SelectKBest
#### Create cross-validation  
sss = StratifiedShuffleSplit(labels_train, 50, test_size = 0.3, random_state = 42)
mm_scaler = MinMaxScaler()

name_list = ['Naive Bayes',
             'Support Vector Machine',
             'Decision Tree',
             'KNN',
             'RandomForest','AdaBoost'
            ]
clf_list = [GaussianNB(),
            SVC(),
            DecisionTreeClassifier(),
            KNeighborsClassifier(),
            RandomForestClassifier(),
            AdaBoostClassifier()
           ]

for name, clf in zip(name_list,clf_list):
    t0 = time() 
    pipe = Pipeline(steps =[('scaler',mm_scaler),('SKB',SelectKBest(k = 6)),('CLF',clf)])
    pipe.fit(features_train,labels_train)
    labels_pred = pipe.predict(features_test)
    clf_final = pipe
    #clf_final = pipe
    print '\t'
    print name
    print 'Fit to training dataset'
    print classification_report(labels_test, labels_pred)    
    print '\t'
    print 'Test on tester.py'
    f1 = tester.test_classifier(clf_final, my_dataset, new_features_list)
    print f1
    print "done in %0.3fs" % (time() - t0)


### Create array for f1
f1_score = [0.39825,0.07246,0.28096,0.15336,0.18340,0.27638]

name_list = ('Naive Bayes','Support Vector Machine',
             'Decision Tree','K-NN',
             'RandomForest','AdaBoost')

plt.bar(np.arange(len(name_list)),f1_score,align='center', color = 'rgbkymc',alpha = 0.5)
plt.xticks(np.arange(len(name_list)), name_list, rotation = 90)
plt.ylabel('F1 score')
plt.title('Classifier')
plt.show()


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
### Create cross-validation
sss = StratifiedShuffleSplit(labels, 50, random_state = 42)

### Naive Bayes optimization
t0 = time()
# PCA
pca = PCA()
#Pipeline
pipeline = Pipeline([('scale',mm_scaler),('SKB',SelectKBest()),('PCA',PCA()),('NB', GaussianNB())])
# clf's parameters
parameters = {'SKB__k':[5,6],
              'PCA__n_components':[2,3,4]
             }
#GridSearchCV
gs = GridSearchCV(pipeline, parameters, cv=sss, scoring='f1')
gs.fit(features, labels)

clf_NB = gs.best_estimator_
tester.test_classifier(clf_NB, my_dataset, new_features_list)
print "done in %0.3fs" % (time() - t0)

### Decision Tree optimization
t0 = time()
# PCA
pca = PCA()
#Pipeline
pipeline = Pipeline([('scale',mm_scaler),('SKB',SelectKBest()),('PCA',PCA()),('DT', DecisionTreeClassifier())])
# clf's parameters
parameters = {'DT__criterion': ['gini','entropy'],
              'DT__splitter':['best','random'],
              'DT__min_samples_split':[2,10,20],
              'DT__max_depth':[10,15,20,25,30],
              'DT__max_leaf_nodes':[5,10,30],
              'SKB__k':[5,6],
              'PCA__n_components':[2,3,4]}
#GridSearchCV
gs = GridSearchCV(pipeline, parameters, cv=sss, scoring='f1')
gs.fit(features, labels)

clf_DT = gs.best_estimator_
tester.test_classifier(clf_DT, my_dataset, new_features_list)
print "done in %0.3fs" % (time() - t0)

clf = clf_NB
dump_classifier_and_data(clf, my_dataset, new_features_list)