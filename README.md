**Data Analysis and Machine Learning Project:**
This repository contains a data analysis and machine learning project that focuses on predicting loan approval status based on various factors. The project involves loading and analyzing the dataset, preprocessing the data, performing feature engineering, training machine learning models, and making predictions. The README.md file provides an overview of the project and its components.

**Project Overview:**
Provide a brief description of the project and its objective. Explain what problem the project aims to solve and how the machine learning models are used for loan approval prediction.

**Exploratory analysis**
exploratory analysis of test dataset is done after filling in missing values and converting data into more suitable  form for the analysis using label encoder.After gettting data in desired format , we use the chart to explore data to better understand the data.Chart used in data are -histigram,pairplot,correlation matrix

**Data Analysis:**
   **Data loading**:
The project uses two datasets: train.csv and test.csv.
The datasets are loaded using the pandas library.
   **Data exploration**:
train.info() is used to display information about the training dataset.
train.shape provides the dimensions of the training dataset.
train.describe() summarizes the statistical properties of the dataset.
train.columns lists all the columns in the dataset.
pd.crosstab() generates a cross-tabulation table for analyzing the relationship between 'Credit_History' and 'Loan_Status'.
   **Data visualization**:
Various visualizations are created using matplotlib and pandas:
train.boxplot(column='ApplicantIncome') displays a boxplot of the 'ApplicantIncome' column.
train['ApplicantIncome'].hist(bins=10) creates a histogram of 'ApplicantIncome'.
train['CoapplicantIncome'].hist(bins=20) generates a histogram of 'CoapplicantIncome'.
train.boxplot(column='ApplicantIncome', by='Education') creates a boxplot of 'ApplicantIncome' based on 'Education'.
train['LoanAmount'].hist() generates a histogram of 'LoanAmount'.
Loan_amount_N.hist() displays a histogram of the transformed 'LoanAmount' after applying logarithmic transformation.

**Machine Learning:**
machine learning part of the project, including preprocessing, model training, and prediction:

  **Data preprocessing:**:
Missing values are handled using various strategies, such as mode and mean imputation.
Categorical variables are encoded using LabelEncoder.
Numeric features are scaled using StandardScaler.
  **Model training and evaluation**:

DecisionTreeClassifier is used as the machine learning model with 'entropy' as the criterion.
The model is trained using DTCclassifier.fit(x_train, y_train).
The accuracy of the model is evaluated using metrics.accuracy_score(y_pred, y_test).
 **Alternative model**:

GaussianNB is used as an alternative machine learning model.
The model is trained using NBclassfier.fit(x_train, y_train).
The accuracy of the model is evaluated using metrics.accuracy_score(y_pred, y_test).


**License**:
dataset was taken from kaggle.com
