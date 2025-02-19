'''
Monthly savings rule: 50/30/20
50% spent on needs
30% savings
20% spent on wants not needs - including entertainment, apparel

Median yearly household income of 90171
Monthly household income of 7514.25
50% spent on needs gives budget of 3757.13 per month

https://www.apartmentlist.com/renter-life/cost-of-living-in-miami
400.67 food, 702.12 transportation, 728 insurance, 200.08 health care = 2030.87 monthly
166.94 electricity, 32.84 nat gas, 45.89 water, 65.47 internet, 58.08 trash = 369.22 utilities monthly
Total costs = 2400.09 monthly
Leaving 1357.03 per month for housing

An average mortgage loan has:
    term: 30 years
    interest rate: 6.811% compounded monthly
calculate loan amount https://www.calculator.net/loan-calculator.html?cloanamount=207%2C750&cloanterm=30&cloantermmonth=0&cinterestrate=6.811&ccompound=monthly&cpayback=month&x=Calculate&type=1#monthlyfixedr
At 1355.90 per month, an amortized loan of 207750 can be made

1 year of savings gives (90171 * 0.3) 27051.30
2 years of savings gives 54102.60
3 years of savings gives 81153.90
'''
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score

import joblib

#read data from file
file = pd.read_csv('STA 4103\\Project 1\\data\\miami-housing.csv') # original file reading; https://www.kaggle.com/datasets/deepcontractor/miami-housing-dataset
content = file # copy of file reading, used for preprocessing

loan_amt = 207750
savings = 27051.30 # 1 year of savings
available_pay = loan_amt + savings

affordable = []
for i in range(0,len(file)):
    if content.loc[i]['SALE_PRC'] >= available_pay: affordable.append(0)    # set 0 for unaffordable
    else: affordable.append(1)                                              # set 1 for affordable

content['AFFORD'] = affordable
content.pop('month_sold')

# extract x and y from data
x = (content.iloc[:,4:16]).values.tolist()
y = (content.iloc[:,16]).values.tolist()

# split x and y into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state = 0)
ss_train = StandardScaler()
x_train = ss_train.fit_transform(x_train)
ss_test = StandardScaler()
x_test = ss_test.fit_transform(x_test)

# create classification models
models = {}
models['Logistic Regression'] = LogisticRegression()    # Logistic Regression
models['Support Vector Machines'] = LinearSVC()         # Support Vector Machines
models['Decision Trees'] = DecisionTreeClassifier()     # Decision Trees
models['Random Forest'] = RandomForestClassifier()      # Random Forest
models['Naive Bayes'] = GaussianNB()                    # Naive Bayes
models['K-Nearest Neighbor'] = KNeighborsClassifier()   # K-Nearest Neighbors

accuracy, precision, recall = {}, {}, {}

i = 0
model_file = ["LogReg.pkl","SVC.pkl","DecTree.pkl","RanFor.pkl","NaiBayes.pkl","KNear.pkl"]
for key in models.keys():
    models[key].fit(x_train, y_train)           # Fit the models
    predictions = models[key].predict(x_test)   # Make predictions using test data
    
    # Calculate metrics; comparing predictions to the test data
    accuracy[key] = accuracy_score(predictions, y_test)     
    precision[key] = precision_score(predictions, y_test)
    recall[key] = recall_score(predictions, y_test)

    # Save model
    joblib.dump(models[key], model_file[i]); i += 1

df_model = pd.DataFrame(index=models.keys(), columns=['Accuracy', 'Precision', 'Recall'])
df_model['Accuracy'] = accuracy.values()
df_model['Precision'] = precision.values()
df_model['Recall'] = recall.values()

print(df_model)