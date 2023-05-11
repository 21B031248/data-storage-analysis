import pandas as pd
import matplotlib.pyplot as mp
import seaborn as sns

dataset = pd.read_excel("HousePricePrediction.xlsx")
#1--for printing the first 5 records of the dataset

#print(dataset.head(5))

#2--for showing the shape of the dataset
print(dataset.shape)

#DATA PREPROCESSING - an important step in machine learning , where the data cleaning, transforming, integrating takes part in order to make it ready for analysis.

#3--categorizing the features depending on their datatypes(int, float, object) and then calculate the number of them

obj = (dataset.dtypes == 'object')
object_cols = list(obj[obj].index)
print("Categorical variables:", len(object_cols))

int = (dataset.dtypes == 'int')
num_cols = list(int[int].index)
print("Integer variables:", len(num_cols))

fl = (dataset.dtypes == 'float')
fl_cols = list(fl[fl].index)
print("Float variables:", len(fl_cols))

#EXPLORATORY DATA ANALYSIS refers to the critical process of perfoming initial investigations on data so as to discover patterns,
#to spot anomalies, to test hypothesis and to check assumptions with the help of summary statistics 
#and graphical representations.
#4--refers to the deep analysis of data so as to discover different patterns and spot anomalies. 
# Before making inferences from data it is essential to examine all your variables.

mp.figure(figsize=(12,6))
sns.heatmap(dataset.corr(),
            cmap = 'BrBG',
            fmt = '.2f',
            linewidths = 2,
            annot = True)

mp.show()

#5--to analyze the different categorical features we draw a barplot
unique_values = []
for col in object_cols:
    unique_values.append(dataset[col].unique().size)
mp.figure(figsize = (10,6))
mp.title('Number of Unique Values of Categorical Features')
mp.xticks(rotation=90)
sns.barplot(x = object_cols, y = unique_values)
#The plot shows that Exterior1st has around 16 unique categories 
# and other features have around  6 unique categories. 
# To findout the actual count of each category we can plot the bargraph of each four features separately.

mp.show()

mp.figure(figsize = (18, 36))
mp.title('Categorical Features: Distribution')
mp.xticks(rotation = 90)
index = 1

for col in object_cols:
    y = dataset[col].value_counts()
    mp.subplot(11, 4, index)
    mp.xticks(rotation = 90)
    sns.barplot(x = list(y.index), y = y)
    index +=1

mp.show()

#DATA CLEANING
#droping the Id column, bc we dont need it
dataset.drop(['Id'], 
              axis = 1,
              inplace = True)

dataset['SalePrice'] = dataset['SalePrice'].fillna(dataset['SalePrice'].mean())

#dropna method deletes all ros that contains NULL values
new_dataset = dataset.dropna()

#isnull function returns a specified value if the expression is NULL
#print(new_dataset.isnull().sum())

#ONEHOTENCODER - FOR LABEL CATEGORICAL FEATURES

from sklearn.preprocessing import OneHotEncoder

s = (new_dataset.dtypes == 'object')
object_cols = list(s[s].index)
print("Categorical variables:")
print(object_cols)
print('Number of categorical features:', len(object_cols))

OH_encoder = OneHotEncoder(sparse=False)
OH_cols = pd.DataFrame(OH_encoder.fit_transform(new_dataset[object_cols]))
OH_cols.index = new_dataset.index
OH_cols.columns = OH_encoder.get_feature_names_out()
df_final = new_dataset.drop(object_cols, axis=1)
df_final = pd.concat([df_final, OH_cols], axis=1)


#SPLITTING DATASET INTO TRAINING AND TESTING
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
 
X = df_final.drop(['SalePrice'], axis=1)
Y = df_final['SalePrice']
 
# Split the training set into
# training and validation set
X_train, X_valid, Y_train, Y_valid = train_test_split(
    X, Y, train_size=0.8, test_size=0.2, random_state=0)

#3 REGRESSION MODELS - SVM, RANDOM FOREST REGRESSOR, LINEAR REGRESSOR

#SVM  can be used for both regression and classification model.
#It finds the hyperplane in the n-dimensional plane.
#To read more about svm refer this.

from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import mean_absolute_percentage_error
 
model_SVR = svm.SVR()
model_SVR.fit(X_train,Y_train)
Y_pred = model_SVR.predict(X_valid)
 
print(mean_absolute_percentage_error(Y_valid, Y_pred))

#Random Forest Regression
from sklearn.ensemble import RandomForestRegressor
 
model_RFR = RandomForestRegressor(n_estimators=10)
model_RFR.fit(X_train, Y_train)
Y_pred = model_RFR.predict(X_valid)
 
print(mean_absolute_percentage_error(Y_valid, Y_pred))

#Linear Regression
from sklearn.linear_model import LinearRegression
 
model_LR = LinearRegression()
model_LR.fit(X_train, Y_train)
Y_pred = model_LR.predict(X_valid)
 
print(mean_absolute_percentage_error(Y_valid, Y_pred))
