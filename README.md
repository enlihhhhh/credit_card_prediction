# Credit Card Prediction Repository
<hr style="border:2px solid gray"> </hr>

## Contributors
- @enlihhhhh
- @tomokiteng
- @sivadboon 
<hr style="border:2px solid gray"> </hr>

## About
### Synopsis
We are a group of NTU students undertaking the task of predicting the Credit Card Approval for our SC1015 module Mini-Project (Introduction to Data Science and Artificial Intelligence). We have taken our dataset from kaggle, under the **'Credit Card Approval Prediction'** [here](https://www.kaggle.com/datasets/rikdifos/credit-card-approval-prediction).
### Problem Definition
Our team's objective is to distinguish the good clients from the bad clients based on the time taken for them to repay their loans. The risk is based on the time taken for them to repay their loans. We have defined the good clients and bad clients as follows:
> Good Clients
> - Impose a low risk on banks
> - Either no loans or able to repay their loans within 29 days

> Bad Clients
> - Impose a high risk on banks
> - Takes longer than 30 days to repay their loans or default on their loans
### Rationale
Our team's reason for choosing this particular dataset is as follows:
> Context
> - Usage of Credit card is extremely prevelant in today's day and age
> - However, given that credit is built on the system of trust, it is important for the banks to known which are the clients that impose a higher risk so that they can be highlighted

> Dataset
> - The dataset for the client's credit cartf approval is extremely exhaustive and delineate
> - Especially so as financial firms would collect as many information as possible to determine if the client impose a high risk
> - After all, the credit system is akin to a short term loan
<hr style="border:2px solid gray"> </hr>

## Content page
### Repository Content
1. [Main Project File](https://www.kaggle.com/datasets/rikdifos/credit-card-approval-prediction) 
   - Problem Definition (Introduction and Content Explanation)
   - Data Wranggling and Data Cleaning (Dealing with duplicates, data types and null values etc.)
   - Exploratory Data Analysis (Exploring Numerical and Categorical variables, as well as further data cleaning and aggregation)
   - Data merging
   - Data Visualisation (For all Response variables for a full picture)
   - Machine Learning (SMOTE, Decision Tree, Random Forest, Logistic Regression, XGBoost Classification)
   - Conclusion 
2. [Project Slides Brief](https://www.kaggle.com/datasets/rikdifos/credit-card-approval-prediction)
   - Introduction to Dataset (Background information and Problem Definition)
   - Data Engineering (Data wrangling, EDA, and Data insights)
   - Core Analysis (Machine Learning models and new techniques learnt)
   - Conclusion and Outcome (Data driven insights and recommendations)
3. [Project Slides Transcript](https://www.kaggle.com/datasets/rikdifos/credit-card-approval-prediction)
   - Introduction to Dataset (Background information and Problem Definition)
   - Data Engineering (Data wrangling, EDA, and Data insights)
   - Core Analysis (Machine Learning models and new techniques learnt)
   - Conclusion and Outcome (Data driven insights and recommendations)

### Models used
1. Decision Tree
2. Random Forest
3. Logistic Regression
4. XGBoost Classification
<hr style="border:2px solid gray"> </hr>

## Lesson Learnt
### Through the project
- SMOTE (Imbalanced dataset handling)
- Onehot encoding (Handling of Categorical Data)
- Random Forest from sklearn (Machine Learning model)
- Logistic Regression from sklearn (Machine Learning model)
- XGBClassifier from XGBoost (Machine Learning model)

### Extra Improvements : Using XGBoost (rfe.ranking) to determine the Best Predictors for our Response: 
For our extra improvements, we decided to use XGBoost to determine what are the best predictors for our response.
**The code we used for our Extra Improvements are as follows:**
`from sklearn.feature_selection import RFE
from sklearn.pipeline import Pipeline
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold

target_df = X_train_bal.copy()
cols = list(target_df.columns)
XGB_model = XGBClassifier() 
rfe = RFE(XGB_model)

X_rfe = rfe.fit_transform(X_train_bal,y_train_bal.GOOD_OR_BAD_CLIENT.ravel())

XGB_model.fit(X_rfe,y_train_bal)
temp_df = pd.Series(rfe.support_,index=cols)
selected_features = temp_df[temp_df==True].index
print(rfe.ranking_) # gives the ranking of all the variables, 1 being the most important
print(selected_features) # prints out the columns which are the most important`

we can see that the following variables have the highest importance affecting the accuracy of the model:
- CNT_CHILDREN
- AMT_INCOME_TOTAL
- AGE_YEARS
- YEARS_EMPLOYED
- NAME_FAMILY_STATUS

The rest of the variables are not a good estimate even though they are in the list, as not all types being included in the list 

### Areas for improvements
<hr style="border:2px solid gray"> </hr>

### References
- https://www.kaggle.com/datasets/rikdifos/credit-card-approval-prediction
- https://slidesgo.com/
