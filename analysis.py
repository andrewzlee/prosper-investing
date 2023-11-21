
"""
@author: andrewlee

# Objective: predict default rate of a given listing

# STEPS
# Look at all completed 3 year loans
# 


# When investing, we want to minimize defaults at all costs.
# This can come at the cost of missing out on some opportunities.
# Ie, we are willing to accept high false positives and a low
# precision if we also get high recall. 

# Another perspective is: the cost of false negatives is high.
# Investing in a false negative will lead to massive losses.

"""

import sqlite3
import pandas as pd
import os
import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score

from sklearn.compose import make_column_selector as selector
from sklearn.preprocessing import OrdinalEncoder

import pickle

#Set workingdir
os.chdir("/Users/andrewlee/Documents/prosper-investing/")

conn = sqlite3.connect("./db/prosper.db")



##############################
# Load and Process training data
##############################
#SELECT * FROM table WHERE id IN (SELECT id FROM table ORDER BY RANDOM() LIMIT x)

predictors = """
    list.*
"""

years = [2018,2019,2020]
dat_full = pd.DataFrame()

# Pull all data with no regard for class imbalance.
# We stop at 2020 because we are pulling 3 and 5 year loans.
# Although it has not been 5 years, we have roughly captured
# most defaults anyways.
for year in years:
    query = f"""
        SELECT CASE WHEN loan_status in (2,3) THEN 1 ELSE 0 END AS loan_status, 
            {predictors}
        FROM loans_{year} loan 
        JOIN loan_listing ll
            ON loan.loan_number = ll.loanid
        JOIN listings_{year} list
            ON ll.listingnumber = list.listing_number
        WHERE loan_status in (2,3,4) and term in (36,60)
    """
    
    print(f"Year: {year}, rows: " + str(dat.shape[0]))

    dat = pd.read_sql(query,conn)
    dat_full = pd.concat([dat_full, dat], ignore_index = True)
    

####
####### Note, below is old code that will build a balanced dataset 
####### by pulling only non-default data and concatenating with
####### default data pulled in the previous query (now modified to 
####### pull all the imbalanced data).
####

# Pull non-default data from earlier years because later years may not have 
# defaulted yet
# dat_tmp = pd.DataFrame()

# years = [2018,2019]
# for year in years:
#     query2 = f"""
#         SELECT 0 AS loan_status, 
#             {predictors}
#         FROM loans_{year} loan 
#         JOIN loan_listing ll
#             ON loan.loan_number = ll.loanid
#         JOIN listings_{year} list
#             ON ll.listingnumber = list.listing_number
#         WHERE loan_status in (4) and term in (36,60)
#     """
    
#     dat2 = pd.read_sql(query2,conn)
#     dat_tmp = pd.concat([dat_tmp, dat2], ignore_index=False)
#     print(f"Year: {year}, rows: " + str(dat2.shape[0]))
    
del predictors, query, query2, year, years

# dat_sample = dat_tmp.sample(n = dat_full.shape[0], replace = False, ignore_index = False, random_state = 123)
# dat_full = pd.concat([dat_full, dat_sample], ignore_index = True)
####
####### End of block
####



# Select only the variables we're interested in
dat_full = dat_full.iloc[:, np.r_[0, 11:12, 17:26, 29:32, 33:38, 41:43, 541:861, 862]]

del dat, dat2, dat_tmp, dat_sample



# Turn categorical variables and text into numeric values for the model
categorical_columns_selector = selector(dtype_include=object)
categorical_columns = categorical_columns_selector(dat_full)

dat_categorical = dat_full[categorical_columns]

encoder = OrdinalEncoder().set_output(transform="pandas")
dat_encoded = encoder.fit_transform(dat_categorical)
dat_encoded[:10]

dat_full = dat_full.drop(dat_categorical, axis = 1).join(dat_encoded)

del dat_encoded, encoder, dat_categorical, categorical_columns
## end

#Store processed data
dat_full.to_pickle("./modeling_data_imbalanced.pkl")

# Remove missing data
dat_full = dat_full.dropna()

#Create training data
x = dat_full.drop("loan_status",axis=1)
y = dat_full["loan_status"]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state=123)


##############################
# Test Various ML models 
##############################

# 1) KNN
from sklearn.neighbors import KNeighborsClassifier
knn_clf = KNeighborsClassifier()
knn_clf.fit(x_train,y_train)

y_pred_test_knn = knn_clf.predict(x_test)

cnf_metrics = confusion_matrix(y_test,y_pred_test_knn)
print("confusion metrics\n",cnf_metrics)
print("*"*20)
accuracy = accuracy_score(y_test,y_pred_test_knn)
print("Accuracy\n",accuracy)
print("*"*20)
clf_report = classification_report(y_test,y_pred_test_knn)
print("Classification report\n",clf_report)


# 2) Adaboost
from sklearn.ensemble import AdaBoostClassifier
adb = AdaBoostClassifier()
adb.fit(x_train,y_train)

y_pred_test_ada = adb.predict(x_test)
accuracy = accuracy_score(y_test,y_pred_test_ada)
print("accuracy",accuracy)
cnf_matrix = confusion_matrix(y_test,y_pred_test_ada)
print(cnf_matrix)
clf_report = classification_report(y_test,y_pred_test_ada)
print("classification report\n",clf_report)


# 3) Random Forest
from sklearn.ensemble import RandomForestClassifier
rf_clf = RandomForestClassifier()
rf_clf.fit(x_train,y_train)

#testing
#y_pred_test_rfc = rf_clf.predict(x_test)
y_pred_test_rfc = (rf_clf.predict_proba(x_test)[:,1] >= 0.3).astype(int)

accuracy = accuracy_score(y_test,y_pred_test_rfc)
print("accuracy",accuracy)
clf_report = classification_report(y_test,y_pred_test_rfc)
print("classification report\n",clf_report)


test_results = pd.concat([x_test[["prosper_score", "TUFicoRange"]].reset_index(drop=True),pd.DataFrame(y_pred_test_rfc, columns = ["pred"]), y_test.reset_index(drop=True)], ignore_index=False, axis = 1)
test_results.head()

test_results_defaults_only = test_results[test_results["loan_status"] == 1]
test_results_nondefault = test_results[test_results["loan_status"] == 0]
#test_results_defaults_only.head()

#Shows that the model get's worse at identifying 
test_results_defaults_only.groupby("TUFicoRange").sum()[["pred", "loan_status"]]
test_results_nondefault.groupby("TUFicoRange").sum()[["pred", "loan_status"]]



# feature_names = [f"feature {i}" for i in range(x.shape[1])]
# importances = rf_clf.feature_importances_
# std = np.std([tree.feature_importances_ for tree in rf_clf.estimators_], axis=0)
# rf_clf_importances = pd.Series(importances, index=feature_names)

# fig, ax = plt.subplots()
# rf_clf_importances.plot.bar(yerr=std, ax=ax)
# ax.set_title("Feature importances using MDI")
# ax.set_ylabel("Mean decrease in impurity")
# fig.tight_layout()
