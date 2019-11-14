import math
import numpy as np
import pandas as pd
import csv
import sys
import os
from datetime import datetime

from sklearn import metrics
from sklearn.model_selection import cross_validate
from sklearn.linear_model import BayesianRidge, Lasso, Ridge, LinearRegression, HuberRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score

#custom class import

try:
    data_input = sys.argv[1]
except:
    print ("Please include input filename.")
    print ("USAGE: python3 train_models.py input_data.csv")
    exit()

if not os.path.isfile(data_input):
    print ("File " + data_input + " not found.")
    print ("USAGE: python3 train_models.py input_data.csv")
    exit()

from bestModel import BestModelObject

#some global variables here
num_of_cross_val_folds = 5

data = pd.read_csv(data_input).values #create numpy array from csv sheet 
#create training and test set (80/20 split)
total_samples = data.shape[0]
train_test_index = math.floor( 0.8 * total_samples ) # index to divide 80/20 train/test data split (must be integer)
train_data = data[:train_test_index] 
test_data = data[train_test_index:]

#randomize training data (do this before before we make cross validation folds)
np.random.shuffle(train_data)

#create numpy arrays of training and test data
X = train_data[:,:3]#first 20 rows and first 3 columns(remember zero based indexing) are X
y = train_data[:,3]# first 20 rows and 4th column are y
#we'll use these to test our final model.
X_test = test_data[:,:3]
y_test = test_data[:,3]

n, d = X.shape
t = X_test.shape[0]

print ("Training inputs shape: " + str(X.shape))
print ("Training labels shape: " + str(y.shape))
print ("Test inputs shape:     " + str(X_test.shape))
print ("Test labels shape:     " + str(y_test.shape))

#this function returns a list of mean scores
def evaluate_model(model, name, ):

    #let's try doing 5 fold vs 4 fold
    #lets try leave on out? might overfit?

    scores = cross_validate(model, X, y, cv=num_of_cross_val_folds, return_train_score=True, scoring=('r2','neg_mean_squared_error','explained_variance'))

    #note
    #scores['train_neg_mean_squared_error'] will return an array of values, each element will correspond to the train MSE for a given fold. so 4 folds will get you 4 elements in this array

    mse_train = np.mean(-scores['train_neg_mean_squared_error'])
    mse_test  = np.mean(-scores['test_neg_mean_squared_error'])#this is more the validation score than test score (test on validation)
    #explantory power of our model. if 0.6 = 60% of the variation in y is explained by the model. the greater proportion that is explained means the greater explanotry power the model has
    r2_train  = np.mean(scores['train_r2'])# if model has small r2, then it has low explantory power, if r2 is high, above 70% this has really good explanatory power. 
    var_test  = np.mean(scores['test_explained_variance'])

    #add the above values to an list and return the list
    final_scores_to_return = [mse_train, mse_test, r2_train, var_test]

    return final_scores_to_return


#each model we want to test can have a set of hyperparameters we want to test as well. so we encapsulate each model with its hyperparameters in their own functions
def tune_linear_regression_hyperparameters():
    #hyperparameters --> none
    
    myModel = LinearRegression()#should we set the 'normalize" parameter to true?, default is false

    #we get back a list of the scores
    myScores = evaluate_model(myModel, 'Linear Regression') # Least squares loss with L1 reg.

    #trackers for best model and its scores
    best_model = myModel
    best_score = myScores
    best_hyperparameters =[]

    #now that we've gon through all combinations of hyperparameters store everything in a bestModelObject and return the object
    return BestModelObject(best_model, best_score, best_hyperparameters)#return best model with best hyperparameters

def tune_lasso_regression_hyperparameters():
    #hyperparameters
    #depending on what get's picked as the best alpha, maybe we should consider another pass with a smaller window around the best value.
    alpha = [1e-15, 1e-10, 1e-8, 1e-5, 1e-4, 1e-3, 1e-2, 1, 5, 10]#range from 0 to infinity, don't use 0 - use plain linear regression if this is the case
    #alpha2 = []#if make a new set of values centered around the best from above?
    tol = [0.01, .001, 0.0001, .00001]
    positive = [True, False]

    #trackers for best model and its scores
    best_model = None
    best_model_scores = None
    best_hyperparameters = []

    run_once_flag = False
    for alpha_element in alpha:
        for tol_element in tol:
            for positive_element in positive:
                myModel = Lasso(alpha=alpha_element, tol=tol_element, positive=positive_element)#should we set the 'normalize" parameter to true?, default is false
                
                #we get back a list of the scores
                myScores = evaluate_model(myModel, 'Lasso Regression') # Least squares loss with L1 reg.
                
                #if index is 0, this is teh first iteration of this loop, so just set best model and score b.c. otherwise we'd have nothing to compare against
                if run_once_flag == False:
                    best_model = myModel
                    best_score = myScores
                    best_hyperparameters.append(alpha_element)
                    best_hyperparameters.append(tol_element)
                    best_hyperparameters.append(positive_element)
                    run_once_flag = True


                #check if we have a better model based on validaiton MSE score, and update if we do
                if myScores[1] < best_score[1]: #we want to check the validation MSE
                    best_model = myModel
                    best_score = myScores
                    best_hyperparameters =[]#clear any old ones
                    best_hyperparameters.append(alpha_element)
                    best_hyperparameters.append(tol_element)
                    best_hyperparameters.append(positive_element)

    #now that we've gon through all combinations of hyperparameters store everything in a bestModelObject and return the object
    return BestModelObject(best_model, best_score, best_hyperparameters)#return best model with best hyperparameters

def tune_ridge_regression_hyperparameters():
    #hyperparameters
    alpha = [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5, 10, 20]
    tol = [0.01, .001, 0.0001, .00001]
    solver = ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']

    #trackers for best model and its scores
    best_model = None
    best_model_scores = None
    best_hyperparameters = []

    run_once_flag = False
    for alpha_element in alpha:
        for tol_element in tol:
            for solver_element in solver:
                
                myModel = Ridge(alpha=alpha_element, tol=tol_element, solver=solver_element)#should we set the 'normalize" parameter to true?, default is false
                
                #we get back a list of the scores
                myScores = evaluate_model(myModel, 'Ridge Regression') # Least squares loss with L2 reg.
                
                #if index is 0, this is teh first iteration of this loop, so just set best model and score b.c. otherwise we'd have nothing to compare against
                if run_once_flag == False:
                    best_model = myModel
                    best_score = myScores
                    best_hyperparameters.append(alpha_element)
                    best_hyperparameters.append(tol_element)
                    best_hyperparameters.append(solver_element)
                    run_once_flag = True


                #check if we have a better model based on validaiton MSE score, and update if we do
                if myScores[1] < best_score[1]: #we want the validation MSE
                    best_model = myModel
                    best_score = myScores
                    best_hyperparameters =[]#clear any old ones
                    best_hyperparameters.append(alpha_element)
                    best_hyperparameters.append(tol_element)
                    best_hyperparameters.append(solver_element)

    #now that we've gon through all combinations of hyperparameters store everything in a bestModelObject and return the object
    return BestModelObject(best_model, best_score, best_hyperparameters)#return best model with best hyperparameters

def tune_huber_regression_hyperparameters():
    #hyperparameters
    alpha = [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20]
    #eplsion: The parameter epsilon controls the number of samples that should be classified as outliers. The smaller the epsilon, the more robust it is to outliers
    #epsilon = list(range(1.0,6.0,0.5))#float, greater than 1.0, default 1.35 (do 1 to 5)
    epsilon = np.append(np.linspace(1,5,9),[1.35])
    tol = [0.01, .001, 0.0001, .00001]

    #trackers for best model and its scores
    best_model = None
    best_model_scores = None
    best_hyperparameters = []

    run_once_flag = False
    for alpha_element in alpha:
        for epsilon_element in epsilon:
            for tol_element in tol:
                
                myModel = HuberRegressor(alpha=alpha_element, epsilon=epsilon_element, tol=tol_element)#should we set the 'normalize" parameter to true?, default is false
                
                #we get back a list of the scores
                myScores = evaluate_model(myModel, 'Huber Regression') # Least squares loss with L2 reg.
                
                #if index is 0, this is teh first iteration of this loop, so just set best model and score b.c. otherwise we'd have nothing to compare against
                if run_once_flag == False:
                    best_model = myModel
                    best_score = myScores
                    best_hyperparameters.append(alpha_element)
                    best_hyperparameters.append(epsilon_element)
                    best_hyperparameters.append(tol_element)
                    run_once_flag = True


                #check if we have a better model based on validaiton MSE score, and update if we do
                if myScores[1] < best_score[1]: #we want the validation MSE
                    best_model = myModel
                    best_score = myScores
                    best_hyperparameters =[]#clear any old ones
                    best_hyperparameters.append(alpha_element)
                    best_hyperparameters.append(epsilon_element)
                    best_hyperparameters.append(tol_element)

    #now that we've gon through all combinations of hyperparameters store everything in a bestModelObject and return the object, bestModelObjec is just a custom class that acts as a container
    return BestModelObject(best_model, best_score, best_hyperparameters)#return best model with best hyperparameters

def tune_bayesian_ridge_regression_hyperparameters():
    #hyperparameters
    alpha1 = [1e-08, 1e-07,1e-06, 1e-05, 1e-04]#for to -8 to -4 in steps of 1
    lambda1 = [1e-08, 1e-07,1e-06, 1e-05, 1e-04]#for to -8 to -4 in steps of 1
    alpha2 = [1e-08, 1e-07,1e-06, 1e-05, 1e-04]#for to -8 to -4 in steps of 1
    lambda2 = [1e-08, 1e-07,1e-06, 1e-05, 1e-04]
    n_iter = list(range(200,401,50))#n_iter go from 200 to 400 in 50 size steps

    
    #trackers for best model and its scores
    best_model = None
    best_model_scores = None
    best_hyperparameters = []

    run_once_flag = False
    for alpha1_element in alpha1:
        for lambda1_element in lambda1:
            for alpha2_element in alpha2:
                for lambda2_element in lambda2:
                    for n_iter_element in n_iter:
                        
                        myModel = BayesianRidge(alpha_1=alpha1_element, lambda_1=lambda1_element, alpha_2=alpha2_element, lambda_2=lambda2_element, n_iter=n_iter_element)#should we set the 'normalize" parameter to true?, default is false
                        
                        #we get back a list of the scores
                        myScores = evaluate_model(myModel, 'Bayesian Ridge Regression') # Least squares loss with L2 reg.
                        
                        #if index is 0, this is teh first iteration of this loop, so just set best model and score b.c. otherwise we'd have nothing to compare against
                        if run_once_flag == False:
                            best_model = myModel
                            best_score = myScores
                            best_hyperparameters.append(alpha1_element)
                            best_hyperparameters.append(lambda1_element)
                            best_hyperparameters.append(alpha2_element)
                            best_hyperparameters.append(lambda2_element)
                            best_hyperparameters.append(n_iter_element)
                            run_once_flag = True


                        #check if we have a better model based on validaiton MSE score, and update if we do
                        if myScores[1] < best_score[1]: #we want the validation MSE
                            best_model = myModel
                            best_score = myScores
                            best_hyperparameters =[]#clear any old ones
                            best_hyperparameters.append(alpha1_element)
                            best_hyperparameters.append(lambda1_element)
                            best_hyperparameters.append(alpha2_element)
                            best_hyperparameters.append(lambda2_element)
                            best_hyperparameters.append(n_iter_element)

    #now that we've gon through all combinations of hyperparameters store everything in a bestModelObject and return the object
    return BestModelObject(best_model, best_score, best_hyperparameters)#return best model with best hyperparameters

def tune_neural_network_regression_hyperparameters():
    #hyperparameters
    hidden_layer_sizes = []
    for i in range(1,2,1):#we want 1 to 3 layers
        for j in range(10,101,10):#layers from size 10 to 100 in steps of 10
            myList = [j] * i
            
            # if i == 1:
            #     myList = [s + ',' for s in map(str, myList)]
        
            #myString = ','.join(map(str, myList))
            myTuple = tuple(myList)
            hidden_layer_sizes.append(myTuple)
    
    activation = ['identity', 'logistic', 'tanh', 'relu']
    sovler = ['lbfgs']
    alpha = [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5, 10, 20]
    learning_rate = ['constant', 'invscaling', 'adaptive']
    #learning_rate_init = [0.001] dont use this unless we use sdg as our solver, but i don't think were doing that for such a small dataset
    tol = [0.01, .001, 0.0001, .00001]

    #trackers for best model and its scores
    best_model = None
    best_model_scores = None
    best_hyperparameters = []

    run_once_flag = False
    for activation_element in activation:
        for solver_element in sovler:
            for alpha_element in alpha:
                for learning_rate_element in learning_rate:
                    for tol_element in tol:
                        for hidden_element in hidden_layer_sizes:
                            #default is 1 hidden layer
                            myModel = MLPRegressor(activation=activation_element, solver=solver_element, alpha=alpha_element, learning_rate=learning_rate_element, tol=tol_element, hidden_layer_sizes=hidden_element)
                            
                            #we get back a list of the scores
                            myScores = evaluate_model(myModel, 'Neural Network Regression') # Least squares loss with L2 reg.
                            
                            #if index is 0, this is teh first iteration of this loop, so just set best model and score b.c. otherwise we'd have nothing to compare against
                            if run_once_flag == False:
                                best_model = myModel
                                best_score = myScores
                                best_hyperparameters.append(activation_element)
                                best_hyperparameters.append(solver_element)
                                best_hyperparameters.append(alpha_element)
                                best_hyperparameters.append(learning_rate_element)
                                best_hyperparameters.append(tol_element)
                                best_hyperparameters.append(hidden_element)
                                run_once_flag = True


                            #check if we have a better model based on validaiton MSE score, and update if we do
                            if myScores[1] < best_score[1]: #we want the validation MSE
                                best_model = myModel
                                best_score = myScores
                                best_hyperparameters =[]#clear any old ones
                                best_hyperparameters.append(activation_element)
                                best_hyperparameters.append(solver_element)
                                best_hyperparameters.append(alpha_element)
                                best_hyperparameters.append(learning_rate_element)
                                best_hyperparameters.append(tol_element)
                                best_hyperparameters.append(hidden_element)

    #now that we've gon through all combinations of hyperparameters store everything in a bestModelObject and return the object
    return BestModelObject(best_model, best_score, best_hyperparameters)#return best model with best hyperparameters

def tune_random_forest_regression_hyperparameters():
    #hyperparameters
    #this has bootstrp sampling enabled by default, each tree gets its own sample
    n_estimators = list(range(1,26,1))#The number of trees in the forest.
    max_depth = list(range(5,26,1))#Max depth of the tree
    max_features = list(range(1,d+1,1))

    #trackers for best model and its scores
    best_model = None
    best_model_scores = None
    best_hyperparameters = []

    run_once_flag = False
    for n_estimators_element in n_estimators:
        #for max_depth_element in max_depth:
        for max_features_element in max_features:
            
            myModel = RandomForestRegressor(n_estimators=n_estimators_element, max_features=max_features_element)#should we set the 'normalize" parameter to true?, default is false
            
            #we get back a list of the scores
            myScores = evaluate_model(myModel, 'Random Forest Regression') # Least squares loss with L2 reg.
            
            #if index is 0, this is teh first iteration of this loop, so just set best model and score b.c. otherwise we'd have nothing to compare against
            if run_once_flag == False:
                best_model = myModel
                best_score = myScores
                best_hyperparameters.append(n_estimators_element)
                #best_hyperparameters.append(max_depth_element)
                best_hyperparameters.append(max_features_element)
                run_once_flag = True


            #check if we have a better model based on validaiton MSE score, and update if we do
            if myScores[1] < best_score[1]: #we want the validation MSE
                best_model = myModel
                best_score = myScores
                best_hyperparameters =[]#clear any old ones
                best_hyperparameters.append(n_estimators_element)
                #best_hyperparameters.append(max_depth_element)
                best_hyperparameters.append(max_features_element)

    #now that we've gon through all combinations of hyperparameters store everything in a bestModelObject and return the object
    return BestModelObject(best_model, best_score, best_hyperparameters)#return best model with best hyperparameters

def tune_kernel_ridge_regression_hyperparameters():
    #hyperparameters
    alpha = [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5, 10, 20]
    kernel = ['linear', 'poly', 'sigmoid', 'rbf', 'laplacian']
    degree = [2, 3, 4]#this will only be used if kernel is poly
    

    #trackers for best model and its scores
    best_model = None
    best_model_scores = None
    best_hyperparameters = []

    run_once_flag = False
    for alpha_element in alpha:
        for kernel_element in kernel:
            if kernel_element == 'poly':#use the degree
                for degree_element in degree:
                    myModel = KernelRidge(alpha=alpha_element, kernel=kernel_element, degree=degree_element)#should we set the 'normalize" parameter to true?, default is false
                    
                    #we get back a list of the scores
                    myScores = evaluate_model(myModel, 'Kernel Ridge Regression') # Least squares loss with L2 reg.
                    
                    #if index is 0, this is teh first iteration of this loop, so just set best model and score b.c. otherwise we'd have nothing to compare against
                    if run_once_flag == False:
                        best_model = myModel
                        best_score = myScores
                        best_hyperparameters.append(alpha_element)
                        best_hyperparameters.append(kernel_element)
                        best_hyperparameters.append(degree_element)
                        run_once_flag = True


                    #check if we have a better model based on validaiton MSE score, and update if we do
                    if myScores[1] < best_score[1]: #we want the validation MSE
                        best_model = myModel
                        best_score = myScores
                        best_hyperparameters =[]#clear any old ones
                        best_hyperparameters.append(alpha_element)
                        best_hyperparameters.append(kernel_element)
                        best_hyperparameters.append(degree_element)
            else:
                myModel = KernelRidge(alpha=alpha_element, kernel=kernel_element)#should we set the 'normalize" parameter to true?, default is false
                
                #we get back a list of the scores
                myScores = evaluate_model(myModel, 'Kernel Ridge Regression') # Least squares loss with L2 reg.
                
                #if index is 0, this is teh first iteration of this loop, so just set best model and score b.c. otherwise we'd have nothing to compare against
                if run_once_flag == False:
                    best_model = myModel
                    best_score = myScores
                    best_hyperparameters.append(alpha_element)
                    best_hyperparameters.append(kernel_element)
                    run_once_flag = True


                #check if we have a better model based on validaiton MSE score, and update if we do
                if myScores[1] < best_score[1]: #we want the validation MSE
                    best_model = myModel
                    best_score = myScores
                    best_hyperparameters =[]#clear any old ones
                    best_hyperparameters.append(alpha_element)
                    best_hyperparameters.append(kernel_element)


    #now that we've gon through all combinations of hyperparameters store everything in a bestModelObject and return the object
    return BestModelObject(best_model, best_score, best_hyperparameters)#return best model with best hyperparameters

def tune_SVR_hyperparameters():
    #hyperparameters
    tol = [0.01, .001, 0.0001, .00001]
    C = [0.001, 0.01, 0.1, 0.5, 1, 1.5, 2.0, 2.5, 3]#C 5 to 3, in steps of .5
    kernel = ['linear', 'poly', 'sigmoid', 'rbf']
    degree = [2, 3, 4]#this will only be used if kernel is poly
    


    #trackers for best model and its scores
    best_model = None
    best_model_scores = None
    best_hyperparameters = []

    run_once_flag = False
    for tol_element in tol:
        for c_element in C:
            for kernel_element in kernel:
                if kernel_element == 'poly':#use the degree
                    for degree_element in degree:

                        myModel = SVR(tol=tol_element, C=c_element ,kernel=kernel_element, degree=degree_element)#should we set the 'normalize" parameter to true?, default is false
                        
                        #we get back a list of the scores
                        myScores = evaluate_model(myModel, 'Support Vector Regression') # Least squares loss with L2 reg.
                        
                        #if index is 0, this is teh first iteration of this loop, so just set best model and score b.c. otherwise we'd have nothing to compare against
                        if run_once_flag == False:
                            best_model = myModel
                            best_score = myScores
                            best_hyperparameters.append(tol_element)
                            best_hyperparameters.append(c_element)
                            best_hyperparameters.append(kernel_element)
                            best_hyperparameters.append(degree_element)
                            run_once_flag = True


                        #check if we have a better model based on validaiton MSE score, and update if we do
                        if myScores[1] < best_score[1]: #we want the validation MSE
                            best_model = myModel
                            best_score = myScores
                            best_hyperparameters =[]#clear any old ones
                            best_hyperparameters.append(tol_element)
                            best_hyperparameters.append(c_element)
                            best_hyperparameters.append(kernel_element)
                            best_hyperparameters.append(degree_element)
                else:
                    myModel = SVR(tol=tol_element, C=c_element ,kernel=kernel_element)#should we set the 'normalize" parameter to true?, default is false
                        
                    #we get back a list of the scores
                    myScores = evaluate_model(myModel, 'Support Vector Regression') # Least squares loss with L2 reg.
                    
                    #if index is 0, this is teh first iteration of this loop, so just set best model and score b.c. otherwise we'd have nothing to compare against
                    if run_once_flag == False:
                        best_model = myModel
                        best_score = myScores
                        best_hyperparameters.append(tol_element)
                        best_hyperparameters.append(c_element)
                        best_hyperparameters.append(kernel_element)
                        run_once_flag = True


                    #check if we have a better model based on validaiton MSE score, and update if we do
                    if myScores[1] < best_score[1]: #we want the validation MSE
                        best_model = myModel
                        best_score = myScores
                        best_hyperparameters =[]#clear any old ones
                        best_hyperparameters.append(tol_element)
                        best_hyperparameters.append(c_element)
                        best_hyperparameters.append(kernel_element)

    #now that we've gon through all combinations of hyperparameters store everything in a bestModelObject and return the object
    return BestModelObject(best_model, best_score, best_hyperparameters)#return best model with best hyperparameters


from sklearn import datasets
from sklearn.externals import joblib

#section to call the evaluation for a given model and it's hyperparamters
print ("Training linear regression model")
best_LR_model = tune_linear_regression_hyperparameters()#get the model with the best hyperparameters
best_LR_model.model.fit(X,y)#retrain the model on the whole dataset
y_hat_train = best_LR_model.model.predict(X)
LR_MSE_train = mean_squared_error(y, y_hat_train)
LR_r2_train = r2_score(y, y_hat_train)
y_hat = best_LR_model.model.predict(X_test)#make predictions
LR_MSE_test = mean_squared_error(y_test, y_hat)#compute MSE of predictions
LR_r2_test = r2_score(y_test, y_hat)
#print("LR hyperparameters and test error") 
#print(best_LR_model.hyperparameters)
#print(LR_MSE_test)
#print(LR_r2_test)

# save model using joblib
joblib.dump(best_LR_model.model, "LR_model.joblib")

print ("Training lasso regression model")
best_lasso_model = tune_lasso_regression_hyperparameters()#get the model with the best hyperparameters
best_lasso_model.model.fit(X,y)#retrain the model on the whole dataset
y_hat_train = best_lasso_model.model.predict(X)
lasso_MSE_train = mean_squared_error(y, y_hat_train)
lasso_r2_train = r2_score(y, y_hat_train)
y_hat = best_lasso_model.model.predict(X_test)#make predictions
lasso_MSE = mean_squared_error(y_test, y_hat)#compute MSE of predictions
lasso_r2 = r2_score(y_test, y_hat)
#print("Lasso hyperparameters and test error") 
#print(best_lasso_model.hyperparameters)
#print(lasso_MSE)
#print(lasso_r2)

# save model using joblib
joblib.dump(best_lasso_model.model, "lasso_model.joblib")

print ("Training ridge regression model")
best_ridge_model = tune_ridge_regression_hyperparameters()#get the model with the best hyperparameters
best_ridge_model.model.fit(np.ascontiguousarray(X),np.ascontiguousarray(y))#retrain the model on the whole dataset
y_hat_train = best_ridge_model.model.predict(X)
ridge_MSE_train = mean_squared_error(y, y_hat_train)
ridge_r2_train = r2_score(y, y_hat_train)
y_hat = best_ridge_model.model.predict(X_test)#make predictions
ridge_MSE = mean_squared_error(y_test, y_hat)#compute MSE of predictions
ridge_r2 = r2_score(y_test, y_hat)
#print("Ridge hyperparameters and test error") 
#print(best_ridge_model.hyperparameters)
#print(ridge_MSE)
#print(ridge_r2)

# save model using joblib
joblib.dump(best_ridge_model.model, "ridge_model.joblib")

print ("Training Huber regression model")
best_huber_model = tune_huber_regression_hyperparameters()#get the model with the best hyperparameters
best_huber_model.model.fit(X,y)#retrain the model on the whole dataset
y_hat_train = best_huber_model.model.predict(X)
huber_MSE_train = mean_squared_error(y, y_hat_train)
huber_r2_train = r2_score(y, y_hat_train)
y_hat = best_huber_model.model.predict(X_test)#make predictions
huber_MSE = mean_squared_error(y_test, y_hat)#compute MSE of predictions
huber_r2 = r2_score(y_test, y_hat)
#print("Huber hyperparameters and test error") 
#print(best_huber_model.hyperparameters)
#print(huber_MSE)
#print(huber_r2)

# save model using joblib
joblib.dump(best_huber_model.model, "huber_model.joblib")

print ("Training Bayesian ridge regression model")
best_bayesian_ridge_model = tune_bayesian_ridge_regression_hyperparameters()#get the model with the best hyperparameters
best_bayesian_ridge_model.model.fit(X,y)#retrain the model on the whole dataset
y_hat_train = best_bayesian_ridge_model.model.predict(X)
bayesRidge_MSE_train = mean_squared_error(y, y_hat_train)
bayesRidge_r2_train = r2_score(y, y_hat_train)
y_hat = best_bayesian_ridge_model.model.predict(X_test)#make predictions
bayesian_ridge_MSE = mean_squared_error(y_test, y_hat)#compute MSE of predictions
bayesian_ridge_r2 = r2_score(y_test, y_hat)
#print("Bayesian Ridge hyperparameters and test error") 
#print(best_bayesian_ridge_model.hyperparameters)
#print(bayesian_ridge_MSE)
#print(bayesian_ridge_r2)

# save model using joblib
joblib.dump(best_bayesian_ridge_model.model, "bayesian_ridge_model.joblib")

print ("Training neural net model")
best_neural_network_model = tune_neural_network_regression_hyperparameters()#get the model with the best hyperparameters
best_neural_network_model.model.fit(X,y)#retrain the model on the whole dataset
y_hat_train = best_neural_network_model.model.predict(X)
NN_MSE_train = mean_squared_error(y, y_hat_train)
NN_r2_train = r2_score(y, y_hat_train)
y_hat = best_neural_network_model.model.predict(X_test)#make predictions
neural_network_MSE = mean_squared_error(y_test, y_hat)#compute MSE of predictions
neural_network_r2 = r2_score(y_test, y_hat)
#print("Neural Network hyperparameters and test error") 
#print(best_neural_network_model.hyperparameters)
#print(neural_network_MSE)
#print(neural_network_r2)

# save model using joblib
joblib.dump(best_neural_network_model.model, "NN_model.joblib")

print ("Training random forest regression model")
best_random_forest_model = tune_random_forest_regression_hyperparameters()#get the model with the best hyperparameters
best_random_forest_model.model.fit(X,y)#retrain the model on the whole dataset
y_hat_train = best_random_forest_model.model.predict(X)
randomForest_MSE_train = mean_squared_error(y, y_hat_train)
randomForest_r2_train = r2_score(y, y_hat_train)
y_hat = best_random_forest_model.model.predict(X_test)#make predictions
random_forest_MSE = mean_squared_error(y_test, y_hat)#compute MSE of predictions
random_forest_r2 = r2_score(y_test, y_hat)
#print("Random Forest Regression hyperparameters and test error") 
#print(best_random_forest_model.hyperparameters)
#print(random_forest_MSE)
#print(random_forest_r2)

# save model using joblib
joblib.dump(best_random_forest_model.model, "randomforest_model.joblib")

print ("Training kernel ridge regression model")
best_kernel_ridge_model = tune_kernel_ridge_regression_hyperparameters()#get the model with the best hyperparameters
best_kernel_ridge_model.model.fit(X,y)#retrain the model on the whole dataset
y_hat_train = best_kernel_ridge_model.model.predict(X)
kernel_MSE_train = mean_squared_error(y, y_hat_train)
kernel_r2_train = r2_score(y, y_hat_train)
y_hat = best_kernel_ridge_model.model.predict(X_test)#make predictions
kernel_ridge_MSE = mean_squared_error(y_test, y_hat)#compute MSE of predictions
kernel_ridge_r2 = r2_score(y_test, y_hat)
#print("Kernel Ridge Regression hyperparameters and test error") 
#print(best_kernel_ridge_model.hyperparameters)
#print(kernel_ridge_MSE)
#print(kernel_ridge_r2)

# save model using joblib
joblib.dump(best_kernel_ridge_model.model, "kernel_ridge_model.joblib")

print ("Training SVR model")
best_SVR_model = tune_SVR_hyperparameters()#get the model with the best hyperparameters
best_SVR_model.model.fit(X,y)#retrain the model on the whole dataset
y_hat_train = best_SVR_model.model.predict(X)
svr_MSE_train = mean_squared_error(y, y_hat_train)
svr_r2_train = r2_score(y, y_hat_train)
y_hat = best_SVR_model.model.predict(X_test)#make predictions
svr_MSE = mean_squared_error(y_test, y_hat)#compute MSE of predictions
svr_r2 = r2_score(y_test, y_hat)
#print("SVR Regression hyperparameters and test error") 
#print(best_SVR_model.hyperparameters)
#print(svr_MSE)
#print(svr_r2)

# save model using joblib
joblib.dump(best_SVR_model.model, "SVR_model.joblib")

#prep csv file to output to
filename = "results_" + str(datetime.now().strftime('%Y-%m-%d_%H:%M:%S')) + ".csv"
with open(filename, 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    # validation MSE from best cros validation model,
    # training MSE from retraining model on whole training set
    # training r^2 from retraning model on whole training set
    # test MSE
    # test r^2
    spamwriter.writerow(['Model'] + ['Validation MSE'] + ['Training MSE'] + ['Training R2'] + ['Test MSE'] + ['Test R2'] + ['Hyper-parameters'])
    spamwriter.writerow(['LR Model'] + [best_LR_model.scores[1]] + [LR_MSE_train] + [LR_r2_train] + [LR_MSE_test] + [LR_r2_test] + best_LR_model.hyperparameters)
    spamwriter.writerow(['Lasso Model'] + [best_lasso_model.scores[1]] + [lasso_MSE_train] + [lasso_r2_train] + [lasso_MSE] + [lasso_r2] + best_lasso_model.hyperparameters)
    spamwriter.writerow(['Ridge Model'] + [best_ridge_model.scores[1]] + [ridge_MSE_train] + [ridge_r2_train] + [ridge_MSE] + [ridge_r2] + best_ridge_model.hyperparameters)
    spamwriter.writerow(['Huber Model'] + [best_huber_model.scores[1]] + [huber_MSE_train] + [huber_r2_train] + [huber_MSE] + [huber_r2] + best_huber_model.hyperparameters)
    spamwriter.writerow(['Bayesian Ridge Model'] + [best_bayesian_ridge_model.scores[1]] + [bayesRidge_MSE_train] + [bayesRidge_r2_train] + [bayesian_ridge_MSE] + [bayesian_ridge_r2] + best_bayesian_ridge_model.hyperparameters)
    spamwriter.writerow(['Neural Net Model'] + [best_neural_network_model.scores[1]] + [NN_MSE_train] + [NN_r2_train] + [neural_network_MSE] + [neural_network_r2] + best_neural_network_model.hyperparameters)
    spamwriter.writerow(['Random Forest Model'] + [best_random_forest_model.scores[1]] + [randomForest_MSE_train] + [randomForest_r2_train] + [random_forest_MSE] + [random_forest_r2] + best_random_forest_model.hyperparameters)
    spamwriter.writerow(['Kernel Model'] + [best_kernel_ridge_model.scores[1]] + [kernel_MSE_train] + [kernel_r2_train] + [kernel_ridge_MSE] + [kernel_ridge_r2] + best_kernel_ridge_model.hyperparameters)
    spamwriter.writerow(['SVR Model'] + [best_SVR_model.scores[1]] + [svr_MSE_train] + [svr_r2_train] + [svr_MSE] + [svr_r2] + best_SVR_model.hyperparameters)
    
