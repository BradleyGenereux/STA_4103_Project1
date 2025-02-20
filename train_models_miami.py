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

from sklearn.metrics import accuracy_score, precision_score, recall_score, mean_squared_error as mean_sq_err, r2_score

import joblib

#read data from file
file = pd.read_csv('STA 4103\\Project 1\\data\\miami-housing.csv') # original file reading; https://www.kaggle.com/datasets/deepcontractor/miami-housing-dataset
content = file # copy of file reading, used for preprocessing


# budget from above calculations in comments
loan_amt = 207750
savings = 27051.30 # 1 year of savings
budget = loan_amt + savings

affordable = []
for i in range(0,len(file)):
    if content.loc[i]['SALE_PRC'] >= budget: affordable.append(0)    # set 0 for unaffordable
    else: affordable.append(1)                                       # set 1 for affordable

content['AFFORD'] = affordable  # create new column 'AFFORD' to replace 'SALE_PRC' as a binary 0 for 'Unaffordable' and 1 for 'Affordable'
content.pop('month_sold')       # remove unnecessary column 'month_sold'
content.pop('PARCELNO')         # remove unnecessary column 'PARCELNO'
content.pop('SALE_PRC')         # remove unnecessary column 'SALE_PRC'


# extract x and y from data
x = (content.iloc[:,0:14]).values.tolist()
y = (content.iloc[:,14]).values.tolist()

# split x and y into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state = 0)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# create classification models
models = {}
models['Logistic Regression'] = LogisticRegression()    # Logistic Regression
models['Support Vector Machines'] = LinearSVC()         # Support Vector Machines
models['Decision Trees'] = DecisionTreeClassifier()     # Decision Trees
models['Random Forest'] = RandomForestClassifier()      # Random Forest
models['Naive Bayes'] = GaussianNB()                    # Naive Bayes
models['K-Nearest Neighbor'] = KNeighborsClassifier()   # K-Nearest Neighbors

accuracy, precision, recall, meanSE, r2 = {}, {}, {}, {}, {}

i = 0
model_file = ["LogReg.pkl","SVC.pkl","DecTree.pkl","RanFor.pkl","NaiBayes.pkl","KNear.pkl"]
for key in models.keys():
    models[key].fit(x_train, y_train)           # Fit the models
    predictions = models[key].predict(x_test)   # Make predictions using test data
    
    # Calculate metrics; comparing predictions to the test data
    accuracy[key] = accuracy_score(y_test, predictions)     
    precision[key] = precision_score(y_test, predictions)
    recall[key] = recall_score(y_test, predictions)
    meanSE[key] = mean_sq_err(y_test, predictions)
    r2[key] = abs(r2_score(y_test, predictions))

    # Save model
    joblib.dump(models[key], model_file[i]); i += 1


# print coeficients
coef = []
ind = ['Logistic Regression', 'Support Vector Machines', 'Decision Trees', 'Random Forest', 'Naive Bayes']
col = ["LATITUDE","LONGITUDE","LND_SQFOOT","TOT_LVG_AREA",
    "SPEC_FEAT_VAL","RAIL_DIST","OCEAN_DIST","WATER_DIST","CNTR_DIST","SUBCNTR_DI","HWY_DIST","age","avno60plus","structure_quality"]

coef.append(models[ind[0]].coef_[0])
coef.append(models[ind[1]].coef_[0])
coef.append(models[ind[2]].feature_importances_)
coef.append(models[ind[3]].feature_importances_)
nb_coef = models['Naive Bayes'].var_

# prepare dataframe for print
df = pd.DataFrame(index=ind, columns=col)
for i in range(0,len(col)):
    arr = []
    for j in range(0,len(ind)-1):
        arr.append(coef[j][i])
    
    arr2 = []
    for k in range(0,2):
        arr2.append(nb_coef[k][i])
    arr.append(arr2)
    df[col[i]] = arr

print(f'{'---' * 65}\n{'   ' * 29}Coefficients\n{'---' * 65}')
print(f'{df[col[:4]]}\n{'---' * 65}')
print(f'{df[col[4:8]]}\n{'---' * 65}')
print(f'{df[col[8:12]]}\n{'---' * 65}')
print(f'{df[col[12:]]}\n{'---' * 65}')
print(f'K-Nearest Neighbors Parameters: {models['K-Nearest Neighbor'].weights}\n{'---' * 65}')

# print metrics
df_model = pd.DataFrame(index=models.keys(), columns=['Accuracy', 'Precision', 'Recall', 'Mean Squared Error', 'R-Squared'])
df_model['Accuracy'] = accuracy.values()
df_model['Precision'] = precision.values()
df_model['Recall'] = recall.values()
df_model['Mean Squared Error'] = meanSE.values()
df_model['R-Squared'] = r2.values()

print(f'{'---' * 65}\n{'---' * 65}\n{'   ' * 29}Model Metrics\n{'---' * 65}')
print(f'{df_model}\n{'---' * 65}')