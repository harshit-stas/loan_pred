import pandas as pd
test=pd.read_csv(r"C:\Users\harshit Raman\Downloads\test_lAUu6dG.csv")
train=pd.read_csv(r"C:\Users\harshit Raman\Downloads\train.csv")
import numpy as np
import matplotlib as plt
a=train.info()
b=train.shape
c=train.describe()
train.columns
pd.crosstab(train['Credit_History'],train['Loan_Status'])
train.boxplot(column='ApplicantIncome') 
train['ApplicantIncome'].hist(bins=10)
train['CoapplicantIncome'].hist(bins=20)
train.boxplot(column='ApplicantIncome', by = 'Education')
train['LoanAmount'].hist()
Loan_amount_N=np.log(train['LoanAmount'])
Loan_amount_N.hist()
train.isnull().sum()
Loan_amount_N.isnull().sum()
d=Loan_amount_N.fillna(Loan_amount_N.mode(),inplace=True)
train['Gender'].fillna(train['Gender'].mode()[0], inplace=True)
train['Married'].fillna(train['Married'].mode()[0], inplace=True)
train['Married'].fillna(train['Married'].mode()[0], inplace=True)
train['Dependents'].fillna(train['Dependents'].mode()[0], inplace=True)
train['Self_Employed'].fillna(train['Self_Employed'].mode()[0], inplace=True)
train['LoanAmount'].fillna(train['LoanAmount'].mean(), inplace=True)
train['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0], inplace=True)
train['Credit_History'].fillna(train['Credit_History'].mode()[0], inplace=True)
Loan_amount_N.fillna(Loan_amount_N.mean(), inplace=True)
train['Total Income']=train['ApplicantIncome']+train['CoapplicantIncome']
train['Total Income'].hist()
Total_Income_N=np.log(train['Total Income'])
Total_Income_N.hist()
train['TotalIncomeLog']=Total_Income_N
train['LoanAmoutLog']=Loan_amount_N
train.boxplot(column='TotalIncomeLog' , by= 'Education')
x=train.iloc[:,np.r_[1:5,9:11,14:16]].values
y=train.iloc[:,np.r_[:,12]].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)
print(x_train)
from sklearn.preprocessing import LabelEncoder
LabelEncoder_x=LabelEncoder()
for i in range(0,5):
    x_train[:,i]=LabelEncoder_x.fit_transform(x_train[:,i])
x_train[:,7]=LabelEncoder_x.fit_transform(x_train[:,7])
LabelEncoder_y=LabelEncoder()
y_train=LabelEncoder_y.fit_transform(y_train)
from sklearn.preprocessing import LabelEncoder
LabelEncoder_y=LabelEncoder()
for i in range(0,5):
    x_test[:,i]=LabelEncoder_x.fit_transform(x_test[:,i])
LabelEncoder_y=LabelEncoder()
y_test=LabelEncoder_y.fit_transform(y_test)
from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
x_train=ss.fit_transform(x_train)
x_test=ss.fit_transform(x_test)
from sklearn.tree import DecisionTreeClassifier
DTCclassifier=DecisionTreeClassifier(criterion='entropy', random_state=0)
y_pred=DTCclassifier.predict(x_test)
DTCclassifier.fit(x_train,y_train)
from sklearn import metrics
print('The acccuracy', metrics.accuracy_score(y_pred,y_test))
from sklearn.naive_bayes import GaussianNB
NBclassfier=GaussianNB()
NBclassfier.fit(x_train,y_train)
y_pred=NBclassfier.predict(x_test)
y_pred
metrics.accuracy_score(y_pred,y_test)
test.info()
test.isnull().sum()
test['Gender'].fillna(test['Gender'].mode()[0], inplace= True)
test['Gender'].fillna(test['Gender'].mode()[0], inplace= True)
test['Dependents'].fillna(test['Dependents'].mode()[0], inplace= True)
test['Self_Employed'].fillna(test['Self_Employed'].mode()[0], inplace= True)
test['LoanAmount'].fillna(test['LoanAmount'].mode()[0], inplace= True)
test['Loan_Amount_Term'].fillna(test['Loan_Amount_Term'].mode()[0], inplace= True)
test['Loan_Amount_Term'].fillna(test['Loan_Amount_Term'].mode()[0], inplace= True)
test['Credit_History'].fillna(test['Credit_History'].mode()[0], inplace= True)
test.boxplot(column=('LoanAmount'))
test.LoanAmount=test.LoanAmount.fillna(test.LoanAmount.mean())
test['LoanAMountLog']=np.log(test['LoanAmount'])
test.boxplot(column=('LoanAMountLog'))
test['TotalIncome']=test['ApplicantIncome']+test['CoapplicantIncome']
test['LogIncome']=np.log(test['TotalIncome'])
test.head()
testF=test.iloc[:,np.r_[1:5,9:11,13:15]].values
print(testF)
for i in range(0,5):
    testF[:,i]=LabelEncoder_x.fit_transform(testF[:,i])
testF[:,7]=LabelEncoder_x.fit_transform(testF[:,7])
testF    
testF=ss.fit_transform(testF)
pred=NBclassfier.predict(testF)
pred
pred=LabelEncoder_y.inverse_transform(pred)
final_list=pd.DataFrame({'Loan_ID':test.Loan_ID,'pred':pred})
final_list.to_csv('Final_list.csv')
