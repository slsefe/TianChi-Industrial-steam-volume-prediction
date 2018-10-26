# TianChi-Industrial-steam-volume-prediction
## competition introduction
home page[https://tianchi.aliyun.com/getStart/introduction.htm?spm=5176.100066.0.0.518433af95U5St&raceId=231693]

## submit log
note: all the submit file were upload

|time|method||offline MSE|offline R2|online MSE|状态|
|---|---|---|---|---|---|---|
|10.21|lightGBM|all features|0.1033||0.1496|little overfitting|
|10.22|xgboost|all features|0.0914||0.2566|strong overfitting|
|10.23|lightGBM|1.drop abnormal feature 'V9','V23','V25','V30','V31','V33','V34'; 2.hyperparameters optimize|0.1035|0.8961|0.1341|weak overfitting|
|10.23|lightGBM|1.drop abnormal feature 'V9','V23','V25','V30','V31','V33','V34'; 2.hyperparameters optimize; 3.drop abnormal data on train set according to sns.boxplot|0.0917|0.8139|0.2548|middle overfitting|
|10.24|lightGBM|1.drop abnormal feature: 'V9','V23','V25','V30','V31','V33','V34'; 2.drop bilinear feature: 'V0','V6','V15','V10','V8','V27'; 3.hyperparameters optimize|0.1181|0.8815|0.1502|weak overfitting|
|10.25|lightGBM|1.构造二项式特征780个; 2.标准化|0.1068|0.8928|0.1549|weak overfitting|
|10.25|lightGBM|1.构造二项式特征780个; 2.PCA降维到100个|0.1871|0.8123|0.5202|strongly overfitting|
|10.26|LightGBM|f-regression 选择20个特征|0.1040|0.8957|0.1417|weak overfitting|
|10.26|LightGBM|1.构造二项式特征780个; 2.f-regression选择100个特征; 3.PCA降维到30个|0.1365|0.8630|0.1417|nearly not overfitting|
|10.27|lightGBM|1.构造二项式特征780个; 2.互信息选择100个特征; 3.PCA降维到30个|0.1392|0.8603|||

## summary:
- do not need too many features, maybe about 25 is a accepted value. 
- must drop abnormal-distributed features, such as 'V9'
- drop abnormal samples on train set will lead to heavily overfitting, so how to deal with these data is critical
- can't simplily drop bilinear features, which will decrease the appearance while weaken overfitting, maybe PCA is a better method
- constract many features and then use PCA on all features is a terrible try, maybe use PCA only on bilinear features is better
- use f-regression to select KBest features looks like effective, maybe combine with PCA will work better
- looks like the R2 metric stands for the ability of overcoming overfitting in some way
