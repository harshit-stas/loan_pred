import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler  # Added MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.feature_selection import chi2, mutual_info_classif, SelectKBest
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC  # Keep SVC for further exploration
from statsmodels.stats.multitest import multipletests  # For post-hoc tests

train=pd.read_csv(r"C:\Users\hrama\Downloads\train.csv")
train.info()
train.shape
train.describe()
train.columns
train.isnull().sum()
train['Gender'].fillna(train['Gender'].mode()[0], inplace=True)
train['Married'].fillna(train['Married'].mode()[0], inplace=True)
train['Married'].fillna(train['Married'].mode()[0], inplace=True)
train['Dependents'].fillna(train['Dependents'].mode()[0], inplace=True)
train['Self_Employed'].fillna(train['Self_Employed'].mode()[0], inplace=True)
train['LoanAmount'].fillna(train['LoanAmount'].mean(), inplace=True)
train['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0], inplace=True)
train['Credit_History'].fillna(train['Credit_History'].mode()[0], inplace=True)
train['Total Income']=train['ApplicantIncome']+train['CoapplicantIncome']

q1=train.ApplicantIncome.quantile(0.25)
q3=train.ApplicantIncome.quantile(0.75)
iqr=q3-q1
lower=q1-(1.5 * iqr)
upper=q3+(1.5 * iqr)
train['ApplicantIncome'].describe()
lower,upper
train_a=train[(train.ApplicantIncome>lower)&(train.ApplicantIncome<upper)]
q1=train_a.CoapplicantIncome.quantile(0.25)
q3=train_a.CoapplicantIncome.quantile(0.75)
iqr=q3-q1
lower=q1-1.5 * iqr
upper=q3+1.5 * iqr
train_c=train_a[(train_a.CoapplicantIncome>lower)&(train_a.CoapplicantIncome<upper)]


pd.crosstab(train_c['Credit_History'],train_c['Loan_Status'])

sns.boxplot(x=train_c['Loan_Status'],y=train_c['ApplicantIncome']) 
plt.xlabel('Loan status')
plt.ylabel('Income of applicants')
plt.title('Distribution of Applicant income over loan status')
plt.show()

sns.histplot(train_c['ApplicantIncome'],bins=20,kde=True,color='b')
plt.xlabel('Income of applicants')
plt.ylabel('Frequency of income')
plt.title('Distribution of applicant income')
plt.show()

sns.histplot(train_c['CoapplicantIncome'],bins=20,kde=True,color='b')
plt.xlabel('Income of Co-applicants')
plt.ylabel('Frequency of income')
plt.title('Distribution of Co-applicant income')
plt.show()

sns.boxplot(y=train_c['ApplicantIncome'], x = train_c['Education'])
plt.xlabel('Income of Applicants')
plt.ylabel('Frequency of income')
plt.title('Distribution of Applicant income')
plt.show()

sns.histplot(train_c['LoanAmount'],bins=20,color='red',kde=True)
plt.xlabel('Amount of Loan')
plt.ylabel('Frequency')
plt.title('Distribution of loan amount')
plt.show()

sns.boxplot(x=train_c['Loan_Status'],y=train_c['LoanAmount'])
plt.xlabel('Status of Loan')
plt.ylabel('Loan amount')
plt.title('Distribution of loan amount over loan status')
plt.show()

train_c.isnull().sum()

train_c['Total Income'] = train_c['ApplicantIncome'] + train_c['CoapplicantIncome']

sns.histplot(train_c['Total Income'],kde=True,bins=20)
plt.show()
Total_Income_N=np.log(train_c['Total Income'])
Loan_amount_N=np.log(train_c['LoanAmount'])
sns.histplot(Total_Income_N,kde=True)
plt.show()


train_c['TotalIncomeLog']=Total_Income_N
train_c['LoanAmoutLog']=Loan_amount_N
fig, ax = plt.subplots()
sns.boxplot(y='TotalIncomeLog', x='Education', data=train_c)
plt.show()


from sklearn.preprocessing import LabelEncoder

cat_col = [col for col in train_c.columns if train_c[col].dtype == 'object']

ls = LabelEncoder()

train_c[cat_col] = train_c[cat_col].apply(ls.fit_transform)
x=train_c.drop(columns=['Loan_ID','Loan_Status'],axis=1)
y=train_c['Loan_Status']

chi_selector = SelectKBest(chi2, k=10)
chi_selector.fit(x, y)
original_features = x.columns.tolist()

selected_feature_indices = chi_selector.get_support(indices=True)
selected_feature_names_chi2 = [original_features[i] for i in selected_feature_indices]

X_selected_chi2 = pd.DataFrame(chi_selector.transform(x), columns=selected_feature_names_chi2[:10])  # Assign first 10 names

x_train, x_test, y_train, y_test = train_test_split(X_selected_chi2, y, test_size=0.7, random_state=0)
x_train.info()

from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

# Define the models
models = {
    'DecisionTree': DecisionTreeClassifier(),
    'NB': GaussianNB(),
    'RandomForest': RandomForestClassifier(),
    'LogisticRegression': LogisticRegression(max_iter=1000),
    'KNeighbors': KNeighborsClassifier(n_neighbors=5),
    'svc': SVC(probability=True)  # Enable probability for ROC AUC
}

# Dictionary to store evaluation metrics for each model
model_results = {}

# Iterate over each model
for model_name, model in models.items():
    scaler = StandardScaler()
    try:
        # Data scaling (optional)
        if model_name in ['LogisticRegression', 'KNeighbors', 'svc']:
            x_train_scaled = scaler.fit_transform(x_train)
            x_test_scaled = scaler.transform(x_test)
            model.fit(x_train_scaled, y_train)
            y_train_pred = model.predict(x_train_scaled)
            y_test_pred = model.predict(x_test_scaled)
        else:
            model.fit(x_train, y_train)
            y_train_pred = model.predict(x_train)
            y_test_pred = model.predict(x_test)
        
        # Calculate evaluation metrics
        model_results[model_name] = {
            'train_accuracy': accuracy_score(y_train, y_train_pred),
            'train_f1': f1_score(y_train, y_train_pred),
            'train_precision': precision_score(y_train, y_train_pred),
            'train_recall': recall_score(y_train, y_train_pred),
            'train_roc_auc': roc_auc_score(y_train, model.predict_proba(x_train)[:, 1]) if hasattr(model, 'predict_proba') else None,
            'test_accuracy': accuracy_score(y_test, y_test_pred),
            'test_f1': f1_score(y_test, y_test_pred),
            'test_precision': precision_score(y_test, y_test_pred),
            'test_recall': recall_score(y_test, y_test_pred),
            'test_roc_auc': roc_auc_score(y_test, model.predict_proba(x_test)[:, 1]) if hasattr(model, 'predict_proba') else None
        }
        
        print(model_name)
        print('Model Evaluation Results:')
        for metric, value in model_results[model_name].items():
            if value is not None:
                print(f'\t- {metric}: {value:.4f}')
        print('---')  # Separator between models
    
    except Exception as e:
        print(f"Error training model {model_name}: {e}")  # Handle potential training errors


from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
import matplotlib.pyplot as plt

# Information Gain Feature Selection
ig_selector = SelectKBest(mutual_info_classif, k=10)
ig_selector.fit(x, y)
selected_feature_names_ig = x.columns.tolist()[:10]
# Create DataFrame with selected features and their corresponding names
X_selected_ig = pd.DataFrame(ig_selector.transform(x), columns=selected_feature_names_ig)
print("Selected Features (Chi-Square):")
print(X_selected_chi2.columns.tolist())

print("\nSelected Features (Information Gain):")
print(X_selected_ig.columns.tolist())

x_train,x_test,y_train,y_test=train_test_split(X_selected_ig,y,test_size=.7,random_state=0)
# Define the models
models = {
    'DecisionTree': DecisionTreeClassifier(),
    'NB': GaussianNB(),
    'RandomForest': RandomForestClassifier(),
    'LogisticRegression': LogisticRegression(max_iter=1000),
    'KNeighbors': KNeighborsClassifier(n_neighbors=5),
    'svc': SVC(probability=True)  # Enable probability for ROC AUC
}

# Dictionary to store evaluation metrics for each model
model_results = {}

# Iterate over each model
for model_name, model in models.items():
    scaler = StandardScaler()
    try:
        # Data scaling (optional)
        if model_name in ['LogisticRegression', 'KNeighbors', 'svc']:
            x_train_scaled = scaler.fit_transform(x_train)
            x_test_scaled = scaler.transform(x_test)
            model.fit(x_train_scaled, y_train)
            y_train_pred = model.predict(x_train_scaled)
            y_test_pred = model.predict(x_test_scaled)
        else:
            model.fit(x_train, y_train)
            y_train_pred = model.predict(x_train)
            y_test_pred = model.predict(x_test)
        
        # Calculate evaluation metrics
        model_results[model_name] = {
            'train_accuracy': accuracy_score(y_train, y_train_pred),
            'train_f1': f1_score(y_train, y_train_pred),
            'train_precision': precision_score(y_train, y_train_pred),
            'train_recall': recall_score(y_train, y_train_pred),
            'train_roc_auc': roc_auc_score(y_train, model.predict_proba(x_train)[:, 1]) if hasattr(model, 'predict_proba') else None,
            'test_accuracy': accuracy_score(y_test, y_test_pred),
            'test_f1': f1_score(y_test, y_test_pred),
            'test_precision': precision_score(y_test, y_test_pred),
            'test_recall': recall_score(y_test, y_test_pred),
            'test_roc_auc': roc_auc_score(y_test, model.predict_proba(x_test)[:, 1]) if hasattr(model, 'predict_proba') else None
        }
        
        print(model_name)
        print('Model Evaluation Results:')
        for metric, value in model_results[model_name].items():
            if value is not None:
                print(f'\t- {metric}: {value:.4f}')
        print('---')  # Separator between models
    
    except Exception as e:
        print(f"Error training model {model_name}: {e}")  # Handle potential training errors



from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
log_reg = LogisticRegression(max_iter=1000)
cv_scores = cross_val_score(log_reg, x_train, y_train, cv=5, scoring='accuracy')
print(f'Cross-validated accuracy: {np.mean(cv_scores):.4f}')
param_grid = {'C': [0.01, 0.1, 1, 10, 100]}

grid_search = GridSearchCV(log_reg, param_grid, cv=5, scoring='accuracy')
grid_search.fit(x_train,y_train)
best_log_reg = grid_search.best_estimator_
print(f'Best C parameter: {grid_search.best_params_["C"]}')
best_log_reg.fit(x_train, y_train)
y_pred = best_log_reg.predict(x_test)
y_proba = best_log_reg.predict_proba(x_test)[:, 1]
print(f'Accuracy: {accuracy_score(y_test, y_pred):.4f}')
print(f'Precision: {precision_score(y_test, y_pred):.4f}')
print(f'Recall: {recall_score(y_test, y_pred):.4f}')
print(f'F1-score: {f1_score(y_test, y_pred):.4f}')
print(f'ROC AUC: {roc_auc_score(y_test, y_proba):.4f}')

# Generate polynomial features of degree 2, considering only interactions
# (no single features raised to a power) and exclude the bias term
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_poly = poly.fit_transform(X_selected_ig)  # Create polynomial features

# Assume you have a scaler object (e.g., StandardScaler)
X_poly_scaled = scaler.fit_transform(X_poly)  # Scale the polynomial features

# Split data into training and testing sets (80% train, 20% test)
X_train_poly, X_test_poly, y_train, y_test = train_test_split(X_poly_scaled, y, test_size=0.2, random_state=42)

# Assume you have a trained logistic regression model
best_log_reg.fit(X_train_poly, y_train)  # Train the model on polynomial features

# Evaluate model performance on test data
y_pred_poly = best_log_reg.predict(X_test_poly)
y_proba_poly = best_log_reg.predict_proba(X_test_poly)[:, 1]
print("Evaluation with polynomial features:")
print(f"\tAccuracy: {accuracy_score(y_test, y_pred_poly):.4f}")
print(f"\tPrecision: {precision_score(y_test, y_pred_poly):.4f}")
print(f"\tRecall: {recall_score(y_test, y_pred_poly):.4f}")
print(f"\tF1-score: {f1_score(y_test, y_pred_poly):.4f}")
print(f"\tROC AUC: {roc_auc_score(y_test, y_proba_poly):.4f}")

# Calibrate the model using sigmoid method with 5-fold cross-validation
calibrated_log_reg = CalibratedClassifierCV(base_estimator=best_log_reg, method='sigmoid', cv=5)
calibrated_log_reg.fit(X_train_poly, y_train)

# Evaluate calibrated model performance on test data
calibrated_y_pred = calibrated_log_reg.predict(X_test_poly)
calibrated_y_proba = calibrated_log_reg.predict_proba(X_test_poly)[:, 1]
print("\nEvaluation with calibration:")
print(f"\tAccuracy: {accuracy_score(y_test, calibrated_y_pred):.4f}")
print(f"\tPrecision: {precision_score(y_test, calibrated_y_pred):.4f}")
print(f"\tRecall: {recall_score(y_test, calibrated_y_pred):.4f}")
print(f"\tF1-score: {f1_score(y_test, calibrated_y_pred):.4f}")
print(f"\tROC AUC: {roc_auc_score(y_test, calibrated_y_proba):.4f}")
