# xgboost parameters optimization

## 1.data preparation

    from xgboost.sklearn import XGBRegressor
    from sklearn import cross_validation, metrics   #Additional scklearn functions
    from sklearn.grid_search import GridSearchCV   #Perforing grid search

    # split train data and test data
    X = np.array(train_df.drop(['target'],axis=1))
    y = np.array(train_df['target']).reshape((-1,1))
    X_test_ = np.array(test_df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2019)
    ddata = xgb.DMatrix(X,label=y)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # use xgboost.cv() training on all data
    def modelfit(alg, dtrain, useTrainCV=True, cv_folds=5, early_stopping_rounds=50, num_features=20):
        if useTrainCV:
            xgb_param = alg.get_xgb_params()
            cvresult = xgb.cv(xgb_param, dtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
                 early_stopping_rounds=early_stopping_rounds, verbose_eval=True)
            alg.set_params(n_estimators=cvresult.shape[0])

        # Fit the algorithm on the data
        alg.fit(X_train, y_train)

        dtrain_predictions = alg.predict(X_train)
        dtest_predictions = alg.predict(X_test)
        print("MSE (Train) : %f" % metrics.mean_squared_error(y_train, dtrain_predictions))
        print("MSE (Test) : %f" % metrics.mean_squared_error(y_test, dtest_predictions))

        xgb.plot_importance(alg, max_num_features=num_features)
        plt.show()
    # xgboost’s sklearn has no feature_importances，but the get_fscore() has same result

## 2.General Approach for Parameter Tuning

通常的做法如下： 
* 1.选择一个相对高一点的学习率（learning rate）：通常0.1是有用的，但是根据问题的不同，可以选择范围在0.05-0.3之间，根据选好的学习率选择最优的树的数目，xgboost有一个非常有用的cv函数可以用于交叉验证并能返回最终的最优树的数目 
* 2.调tree-specific parameters（max_depth, min_child_weight, gamma, subsample, colsample_bytree） 
* 3.调regularization parameters（lambda, alpha） 
* 4.调低学习率并决定优化的参数

### step1:Fix learning rate and number of estimators for tuning tree-based parameters

1.设置参数的初始值：
* max_depth = 5 : 调整范围3-10,4-6都是不错的初始值的选择
* min_child_weight = 1 : 如果数据是不平衡数据，初始值设置最好小于1
* gamma = 0 : 初始值通常设置在0.1-0.2范围内，并且在后续的调参中也会经常被调节
* subsample, colsample_bytree = 0.8 : 通常使用0.8作为调参的开始参数，调整范围为0.5-0.9
* scale_pos_weight = 1:因为数据为高度不平衡数据



      xgb1 = xgb.XGBRegressor(
       learning_rate =0.1,
       n_estimators=1000,
       max_depth=5,
       min_child_weight=1,
       gamma=0,
       subsample=0.8,
       colsample_bytree=0.8,
       objective= 'reg:linear',
       nthread=4,
       scale_pos_weight=1,
       seed=2019)

      modelfit(xgb1,dtrain)
      

* learning_rate=0.1
* n_estimators=303-50+1=254

### Step 2: Tune max_depth and min_child_weight
We tune these first as they will have the highest impact on model outcome. To start with, let’s set wider ranges and then we will perform another iteration for smaller ranges.

Important Note: I’ll be doing some heavy-duty grid searched in this section which can take 15-30 mins or even more time to run depending on your system. You can vary the number of values you are testing based on what your system can handle.

      param_test1 = {
       'max_depth':[3,5,7,9],
       'min_child_weight':[1,3,5]
      }
      grid1 = GridSearchCV(estimator = XGBRegressor( learning_rate=0.1, n_estimators=254, max_depth=5, min_child_weight=1, gamma=0, 
                                                        subsample=0.8, colsample_bytree=0.8, objective= 'reg:linear', nthread=4, 
                                                        scale_pos_weight=1, seed=2019),
                              param_grid = param_test1, scoring='neg_mean_squared_error',n_jobs=-1,iid=False, cv=5)
      grid1.fit(X_train,y_train)
      grid1.grid_scores_, grid1.best_params_, grid1.best_score_

- 'max_depth': 3, 'min_child_weight': 5
- fine tune the 'max_depth' and 'min_child_weight' based on the last result

      param_test2 = {
       'max_depth':[2,3,4],
       'min_child_weight':[4,5,6]
      }
      grid2 = GridSearchCV(estimator = XGBRegressor( learning_rate=0.1, n_estimators=254, max_depth=5, min_child_weight=1, gamma=0, 
                                                        subsample=0.8, colsample_bytree=0.8, objective= 'reg:linear', nthread=4, 
                                                        scale_pos_weight=1, seed=2019),
                              param_grid = param_test2, scoring='neg_mean_squared_error',n_jobs=-1,iid=False, cv=5)
      grid2.fit(X_train,y_train)
      grid2.grid_scores_, grid2.best_params_, grid2.best_score_

- 'max_depth': 3, 'min_child_weight': 5
- now we have tuned the 'max_depth' and 'min_child_weight'

### Step 3: Tune gamma
* Now lets tune gamma value using the parameters already tuned above. Gamma can take various values but I’ll check for 5 values here. You can go into more precise values as.

      param_test3 = {
          'gamma':[i/10.0 for i in range(0,5)]
      }
          grid3 = GridSearchCV(estimator = XGBRegressor( learning_rate=0.1, n_estimators=254, max_depth=3, min_child_weight=5, gamma=0, 
                                                            subsample=0.8, colsample_bytree=0.8, objective= 'reg:linear', nthread=4, 
                                                            scale_pos_weight=1, seed=2019),
                                  param_grid = param_test3, scoring='neg_mean_squared_error',n_jobs=-1,iid=False, cv=5)
          grid3.fit(X_train,y_train)
          grid3.grid_scores_, grid3.best_params_, grid3.best_score_

Here, we can see the improvement in score. So the final parameters are:
* max_depth: 3
* min_child_weight: 5
* gamma: 0.1
* 在调整超参数max_depth, min_child_weight, gamma之后，重新训练获得最优的n_estimators

      xgb2 = xgb.XGBRegressor(
       learning_rate =0.1,
       n_estimators=1000,
       max_depth=3,
       min_child_weight=5,
       gamma=0.1,
       subsample=0.8,
       colsample_bytree=0.8,
       objective= 'reg:linear',
       nthread=4,
       scale_pos_weight=1,
       seed=2019)

      modelfit(xgb2,dtrain)

### Step 4: Tune subsample and colsample_bytree
The next step would be try different subsample and colsample_bytree values. Lets do this in 2 stages as well and take values 0.6,0.7,0.8,0.9 for both to start with.

    param_test5 = {
        'subsample':[0.6,0.7,0.8,0.9],
        'colsample_bytree':[0.6,0.7,0.8,0.9]
    }
    grid5 = GridSearchCV(estimator = XGBRegressor( learning_rate=0.1, n_estimators=389, max_depth=3, min_child_weight=5, gamma=0.1, 
                                                      subsample=0.8, colsample_bytree=0.8, objective= 'reg:linear', nthread=4, 
                                                      scale_pos_weight=1, seed=2019),
                            param_grid = param_test5, scoring='neg_mean_squared_error',n_jobs=-1,iid=False, cv=5)
    grid5.fit(X_train,y_train)
    grid5.grid_scores_, grid5.best_params_, grid5.best_score_

- 'colsample_bytree': 0.8, 'subsample': 0.8
- fine tune the 'colsample_bytree' and 'subsample'

      param_test6 = {
          'subsample':[i/100.0 for i in range(75,90,5)],
          'colsample_bytree':[i/100.0 for i in range(75,90,5)]
      }
      grid6 = GridSearchCV(estimator = XGBRegressor( learning_rate=0.1, n_estimators=389, max_depth=3, min_child_weight=5, gamma=0.1, 
                                                        subsample=0.8, colsample_bytree=0.8, objective= 'reg:linear', nthread=4, 
                                                        scale_pos_weight=1, seed=2019),
                              param_grid = param_test6, scoring='neg_mean_squared_error',n_jobs=-1,iid=False, cv=5)
      grid6.fit(X_train,y_train)
      grid6.grid_scores_, grid6.best_params_, grid6.best_score_

Again we got the same values as before. Thus the optimum values are:

* subsample: 0.85
* colsample_bytree: 0.85

### Step 5: Tuning Regularization Parameters
Next step is to apply regularization to reduce overfitting. Though many people don’t use this parameters much as gamma provides a substantial way of controlling complexity. But we should always try it. I’ll tune ‘reg_alpha’ value here and leave it upto you to try different values of ‘reg_lambda’.

    param_test7 = {
        'reg_alpha':[0,0.001,0.01,0.1,1]
    }
    grid7 = GridSearchCV(estimator = XGBRegressor( learning_rate=0.1, n_estimators=389, max_depth=3, min_child_weight=5, gamma=0.1, 
                                                      subsample=0.85, colsample_bytree=0.85, objective= 'reg:linear', nthread=4, 
                                                      scale_pos_weight=1, seed=2019),
                            param_grid = param_test7, scoring='neg_mean_squared_error',n_jobs=-1,iid=False, cv=5)
    grid7.fit(X_train,y_train)
    grid7.grid_scores_, grid7.best_params_, grid7.best_score_

- 'reg_alpha': 0

      param_test8 = {
          'reg_lambda':[0.001,0.01,0.1,1],
      }
      grid8 = GridSearchCV(estimator = XGBRegressor( learning_rate=0.1, n_estimators=389, max_depth=3, min_child_weight=5, gamma=0.1, 
                                                        subsample=0.85, colsample_bytree=0.85, objective= 'reg:linear', nthread=4, 
                                                        scale_pos_weight=1, seed=2019),
                              param_grid = param_test8, scoring='neg_mean_squared_error',n_jobs=-1,iid=False, cv=5)
      grid8.fit(X_train,y_train)
      grid8.grid_scores_, grid8.best_params_, grid8.best_score_

- 'reg_lambda': 1

      param_test8 = {
          'reg_lambda':[0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
      }
      grid8 = GridSearchCV(estimator = XGBRegressor( learning_rate=0.1, n_estimators=389, max_depth=3, min_child_weight=5, gamma=0.1, 
                                                        subsample=0.85, colsample_bytree=0.85, objective= 'reg:linear', nthread=4, 
                                                        scale_pos_weight=1, seed=2019),
                              param_grid = param_test8, scoring='neg_mean_squared_error',n_jobs=-1,iid=False, cv=5)
      grid8.fit(X_train,y_train)
      grid8.grid_scores_, grid8.best_params_, grid8.best_score_

- 'reg_lambda': 0.2

You can see that we got a better CV. Now we can apply this regularization in the model and look at the impact:
    
    xgb3 = xgb.XGBRegressor(
        learning_rate =0.1,
        n_estimators=1000,
        max_depth=3,
        min_child_weight=5,
        gamma=0.1,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0,
        reg_lambda=0.2,
        objective= 'reg:linear',
        nthread=4,
        scale_pos_weight=1,
        seed=2019)

    modelfit(xgb3,dtrain)

### Step 6: Reducing Learning Rate
Lastly, we should lower the learning rate and add more trees. Lets use the cv function of XGBoost to do the job again.

    #Choose all predictors except target & IDcols
    xgb3 = xgb.XGBRegressor(
        learning_rate =0.01,
        n_estimators=10000,
        max_depth=3,
        min_child_weight=5,
        gamma=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective= 'reg:linear',
        nthread=4,
        scale_pos_weight=1,
        seed=2019)

    modelfit(xgb3,dtrain)

MSE获得巨大提升，最优参数如下：

* learing_rate=0.01
* n_estimators=2756-50+1=2707
* max_depth=3
* min_child_weight=5
* gamma=0.1
* subsample=0.8
* colsample_bytree=0.8

      xgb3.fit(X,y)# 使用最优模型在整个训练集上进行训练，再对测试数据进行预测
      submit['xgb'] = xgb3.predict(X_test_)
      submit['xgb'].to_csv(submit_file,index = False)

## 3.reference

- [Complete Guide to Parameter Tuning in XGBoost (with codes in Python)](https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/)


