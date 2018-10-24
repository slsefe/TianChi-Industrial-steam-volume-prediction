# TianChi-Industrial-steam-volume-prediction
## competition introduction
home page[https://tianchi.aliyun.com/getStart/introduction.htm?spm=5176.100066.0.0.518433af95U5St&raceId=231693]

## submit log
note: all the submit file were upload
- 10.16, lasso regression, all data&feature, online MSE:2.1002
- 10.18, linear regression, all data&feature, online MSE:3.2806
- 10.21, epsilon-SVR, all data&feature, offline MSE:0.1011, online MSE:0.9632
- 10.21, lightGBM, all data&feature, offline MSE:0.1033, online MSE:0.1496
- 10.22, xgboost, all data&feature, offline MSE:0.0914, online MSE:0.2566
- 10.23 10:21, lightGBM, 
  - drop abnormal feature 'V9','V23','V25','V30','V31','V33','V34'
  - hyperparameters optimize
  - offline result:
    - MAE:0.23018598427355388
    - MSE:0.10356576101412938
    - RMSE:0.3218163467167717
    - median_AE:0.16902948021234104
    - R2:0.8961636957276298
   - online result:
    - MSE:0.1341
- 10.23 19:04, lightGBM
  - drop abnormal feature 'V9','V23','V25','V30','V31','V33','V34'
  - hyperparameters optimize
  - drop abnormal data on train set according to sns.boxplot
  - offline result:
    - MAE:0.22386904412869685
    - MSE:0.09175053278155469
    - RMSE:0.3029035040760583
    - median_AE:0.15808274071025233
    - R2:0.8139535621816923
  - online result:
    - MSE:0.2548
- 10.23, lightGBM
  - drop abnormal feature: 'V9','V23','V25','V30','V31','V33','V34'
  - drop bilinear feature: 'V0','V6','V15','V10','V8','V27'
  - hyperparameters optimize
  - **not drop abnormal data on train set according to sns.boxplot**
  - offline result:
    - MAE:0.24648279890093366
    - MSE:0.1181034310819909
    - RMSE:0.3436617975306404
    - median_AE:0.1819716207792185
    - R2:0.8815880491259325
  - online result:
    - MSE:0.1502
