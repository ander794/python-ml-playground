import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, classification_report
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import RFE
from sklearn.metrics import f1_score

import matplotlib.pyplot as plt
import numpy as np

def calculate_c_parameter(model,svc):
  accuracy_list=[]
  c_parameter_list=[]
  c_parameter_optimalization_range = range(1,15,1)
  max_accuracy = 0
  best_c_par = 0
  bestfscore = 0
  for c_parameter in c_parameter_optimalization_range:
    c_value=float('10e-{}'.format(c_parameter))
    print(c_value)
    if svc:
      model = Pipeline((
        ("poly_features", PolynomialFeatures(degree=3)),
        ("scaler", StandardScaler()),
        ("svm_clf", LinearSVC(C=c_value, loss="hinge"))))
    else:
      model.C=c_value
    model.fit(train_X,train_y)
    predicted_subscriptions = model.predict(val_X)
    calculated_accuracy = metrics.accuracy_score(val_y, predicted_subscriptions)
    accuracy_list.append(calculated_accuracy)
    c_parameter_list.append(c_parameter)
    fscore = f1_score(val_y, predicted_subscriptions, average='micro')
    print("F1 Score:",fscore)
    if(max_accuracy<calculated_accuracy):
      max_accuracy = calculated_accuracy
      best_c_par=c_value
    print("Accuracy:",calculated_accuracy)
  
  plt.plot(c_parameter_list,accuracy_list)
  plt.show()
  return best_c_par

file_path='bank-additional/bank-additional-full.csv'

#use separator!
bank_data = pd.read_csv(file_path,sep=';')
#print(bank_data.info())
#print(bank_data.describe(percentiles=[.25]))
#print(bank_data.columns)
#bank_data.head()
#print(bank_data['job']=='student')
#personal_data = bank_data.loc[:,['job','education']]

# Drop NA values
bank_data = bank_data.dropna()
#bank_data.drop(bank_data.columns[[0,3,4,5,6,9,10,11,12,13,14]],axis=1,inplace=True)

#print(bank_data.dtypes)
#Get the features and labels
#columns = ["age","balance"]
#X = bank_data.loc[:,columns]

#Data preprocessing
#Convert categorical features to integers
features = bank_data.copy()
categorical_columns = features.select_dtypes('object')
numeric_columns = features.select_dtypes('number')
#numeric_columns = numeric_columns[numeric_columns.columns.drop('duration')]

categoric_one_hot_columns = pd.get_dummies(categorical_columns[categorical_columns.columns.drop('y')])


frames = [numeric_columns,categoric_one_hot_columns]
X = pd.concat(frames, axis=1)
#Remove unknown columns
X = X.drop(['education_unknown','job_unknown','default_unknown'],axis=1)
print(X.head())

#print(data_frame_X)

#X.hist()
#plt.show()
y = bank_data.y

#Split data so we can use separate datase for train and testing (overfit and underfit problems)
train_X, val_X, train_y, val_y = train_test_split(X, y,train_size=0.30)

log_reg = LogisticRegression()
log_reg.fit(train_X, train_y)

predicted_subscriptions = log_reg.predict(val_X)
print(predicted_subscriptions)
print(val_y)

cnf_matrix = metrics.confusion_matrix(val_y, predicted_subscriptions)
print(cnf_matrix)

log_reg_model = LogisticRegression()

best_c_par = calculate_c_parameter(log_reg_model,False)

log_reg_model = LogisticRegression()
log_reg_model.fit(train_X,train_y)

predicted_subscriptions = log_reg_model.predict(val_X)
cnf_matrix = metrics.confusion_matrix(val_y, predicted_subscriptions)
model_accuracy = metrics.accuracy_score(val_y, predicted_subscriptions)
print("Confusion matrix: \n",cnf_matrix)
print(classification_report(val_y,predicted_subscriptions))
print("Accuracy: ",model_accuracy)

# Plot also the training points
#plt.scatter([X.iloc[:,0].tolist()], X.iloc[:,1].tolist(),c=y.iloc[:].map({'yes':'green', 'no':'red'}))
#plt.xlabel('Age')
#plt.ylabel('Balance')

#plt.show()

#The coefficient of the features in show the importance of feature in the prediction.
#The correlation coefficient ranges from –1 to 1. When it is close to 1, it means that
#there is a strong positive correlation; for example, the median house value tends to go
#up when the median income goes up. When the coefficient is close to –1, it means
#that there is a strong negative correlation; you can see a small negative correlation
#between the latitude and the median house value (i.e., prices have a slight tendency to
#go down when you go north). Finally, coefficients close to zero mean that there is no
#linear correlation. Figure 2-14 shows various plots along with the correlation coefficient
#between their horizontal and vertical axes.

#However, It does not tell us how exactly those features play a role in prediction. And the relatively low ranking of occupation seems counterintuitive.

coefs = pd.Series(log_reg.coef_[0], index=train_X.columns)
print(train_X.columns)
coefs = coefs.sort_values()
plt.subplot(1,1,1)
coefs.plot(kind="bar")
plt.show()
print(coefs.sort_values(ascending = False))


#Trying with SVM
svm_model = Pipeline((
        ("poly_features", PolynomialFeatures(degree=2)),
        ("scaler", StandardScaler()),
        ("svm_clf", LinearSVC(C=1, loss="hinge"))))
#best_c_par = calculate_c_parameter(svm_model,True)

svm_model = Pipeline((
        ("poly_features", PolynomialFeatures(degree=2)),
        ("scaler", StandardScaler()),
        ("svm_clf", LinearSVC(C=1, loss="hinge"))))
svm_model.fit(train_X, train_y)

predicted_subscriptions = svm_model.predict(val_X)
cnf_matrix = metrics.confusion_matrix(val_y, predicted_subscriptions)
model_accuracy = metrics.accuracy_score(val_y, predicted_subscriptions)
print("Confusion matrix: \n",cnf_matrix)
print(classification_report(val_y,predicted_subscriptions))
print("Accuracy: ",model_accuracy)