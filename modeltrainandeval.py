#%% 
import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import KFold,GridSearchCV
from sklearn.dummy import DummyClassifier,DummyRegressor
from sklearn.neural_network import MLPRegressor,MLPClassifier
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error,accuracy_score,f1_score,r2_score
from statsmodels.formula.api import ols 
from sklearn.pipeline import Pipeline
import seaborn as sns


#%%
data=pd.read_csv("HR_data.csv").drop("Unnamed: 0",axis=1)
betterdata=data.drop(["Frustrated","Cohort","Individual","Round","Puzzler"],axis=1)
Y=data["Frustrated"].copy()
X = pd.get_dummies(data=betterdata,columns=["Phase"])
# %%
# Define models:
NNRpipe=Pipeline([("scaler", StandardScaler()),("nnr",MLPRegressor(activation="relu",solver="adam",max_iter=10000))])
NNRparam_grid = {
    "nnr__hidden_layer_sizes": [
        (10,),
        (10, 10),
        (10, 10, 10),
        (10, 10, 10, 10),
        (10, 10, 10, 10, 10),
        (100,),
        (100, 100),
        (100, 100, 100),
        (100, 100, 100, 100),
        (100, 100, 100, 100, 100),
        (200,),
        (200, 200),
        (200, 200, 200),
        (200, 200, 200, 200),
        (200, 200, 200, 200, 200),
    ]
}


Dummy_r_pipe = Pipeline([("scaler", StandardScaler()), ("dummyr",DummyRegressor(strategy="mean"))])


LogResPipe=Pipeline([("scaler", StandardScaler()),("logres",LogisticRegression(max_iter=10000))])
params = 1 / np.linspace(0.0001, 1000, 20)
lrparam_grid = {"logres__C": params}

RFRpipeline=Pipeline([("scaler", StandardScaler()),("rfr",RandomForestRegressor(criterion="squared_error"))])
RFRparamgrid={
    "rfr__n_estimators": [
        5,
         10,
         20,
         50,
         100,
         200
    ]
}

#%%
# Define cross-validation:
innersplits=5
outersplits=20
cv_outer=KFold(n_splits=outersplits,shuffle=True)


#%%
# Regressors
LR_predicts = []
LR_best_params = []
LR_mean_sq_errors = []
LR_r2_errors = []
LR_alternative_scoring=[]

RFR_predicts = []
RFR_best_params = []
RFR_mean_sq_errors = []
RFR_r2_errors = []
RFR_alternative_scoring=[]

dummyr_predicts = []
dummyr_best_params = []
dummyr_mean_sq_errors = []
dummyr_r2_errors = []
dummyr_alternative_scoring=[]

NNr_predicts = []
NNr_best_params = []
NNr_mean_sq_errors = []
NNr_r2_errors = []
NNr_alternative_scoring=[]

#%%
# Cross-validation loop:
train_indeces=[]
test_indeces=[]
for train_index,test_index in cv_outer.split(X, Y):
    # Inner split:
    CV_inner=KFold(n_splits=innersplits, shuffle=True)


    # Models:
    Dummy_r_pipe.fit(X.iloc[train_index], Y.iloc[train_index])
    dummy_r_baseline_predictions = Dummy_r_pipe.predict(X.iloc[test_index])
    dummyr_mean_sq_errors.append(mean_squared_error(Y.iloc[test_index], dummy_r_baseline_predictions))
    dummyr_r2_errors.append(r2_score(Y.iloc[test_index], dummy_r_baseline_predictions))


    LR_grid_search=GridSearchCV(estimator=LogResPipe,param_grid=lrparam_grid,cv=CV_inner,scoring="neg_mean_squared_error",verbose=3,n_jobs=-1)
    LR_grid_search.fit(X.iloc[train_index], Y.iloc[train_index])
    LR_best_params.append(LR_grid_search.best_params_["logres__C"])
    LRpredictions=LR_grid_search.predict(X.iloc[test_index])
    LR_mean_sq_errors.append(mean_squared_error(LRpredictions,Y.iloc[test_index]))
    LR_r2_errors.append(r2_score(LRpredictions,Y.iloc[test_index]))
    LR_predicts.extend(LRpredictions)

    NNr_grid_search=GridSearchCV(estimator=NNRpipe,param_grid=NNRparam_grid,cv=CV_inner,scoring="neg_mean_squared_error",verbose=3,n_jobs=-1)
    NNr_grid_search.fit(X.iloc[train_index], Y.iloc[train_index])
    NNr_best_params.append(NNr_grid_search.best_params_["nnr__hidden_layer_sizes"])
    NNrpredictions=NNr_grid_search.predict(X.iloc[test_index])
    NNr_mean_sq_errors.append(mean_squared_error(NNrpredictions,Y.iloc[test_index]))
    NNr_r2_errors.append(r2_score(NNrpredictions,Y.iloc[test_index]))
    NNr_predicts.extend(NNrpredictions)

    RFR_grid_search=GridSearchCV(estimator=RFRpipeline,param_grid=RFRparamgrid,cv=CV_inner,scoring="neg_mean_squared_error",verbose=3,n_jobs=-1)
    RFR_grid_search.fit(X.iloc[train_index], Y.iloc[train_index])
    RFR_best_params.append(RFR_grid_search.best_params_["rfr__n_estimators"])
    RFRpredictions=RFR_grid_search.predict(X.iloc[test_index])
    RFR_mean_sq_errors.append(mean_squared_error(RFRpredictions,Y.iloc[test_index]))
    RFR_r2_errors.append(r2_score(RFRpredictions,Y.iloc[test_index]))
    RFR_predicts.extend(RFRpredictions)
    
# %%
# Dataframe for scores

regression_results_dict = {
    "Dummy Regression": [],
    "Logistic Regression": [],
    "Neural Network Regression": [],
    "Random Forest Regression": [],
    
}

regression_results_dict["Dummy Regression"].extend([None, np.mean(dummyr_mean_sq_errors),dummyr_mean_sq_errors, dummyr_r2_errors])
regression_results_dict["Logistic Regression"].extend([LR_best_params, np.mean(LR_mean_sq_errors),LR_mean_sq_errors, LR_r2_errors])
regression_results_dict["Neural Network Regression"].extend([NNr_best_params,np.mean(NNr_mean_sq_errors),NNr_mean_sq_errors, NNr_r2_errors])
regression_results_dict["Random Forest Regression"].extend([RFR_best_params, np.mean(RFR_mean_sq_errors),RFR_mean_sq_errors, RFR_r2_errors])



regression_df = pd.DataFrame.from_dict(regression_results_dict, orient='index', columns=['Best Params', 'Mean, mean sq','Mean sq errors', 'R2 Errors']).transpose()

#%%
model1_perf=regression_results_dict["Dummy Regression"][2]
model2_perf=regression_results_dict["Logistic Regression"][2]
model3_perf=regression_results_dict["Neural Network Regression"][2]
model4_perf=regression_results_dict["Random Forest Regression"][2]
# %%
model1=pd.DataFrame(model1_perf)
model2=pd.DataFrame(model2_perf)
model3=pd.DataFrame(model3_perf)
model4=pd.DataFrame(model4_perf)

