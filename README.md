# TianChi-Industrial-steam-volume-prediction
## competition introduction
[home page](https://tianchi.aliyun.com/getStart/introduction.htm?spm=5176.100066.0.0.518433af95U5St&raceId=231693)

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
|10.26|LightGBM|1.构造二项式特征780个; 2.f-regression选择100个特征; 3.PCA降维到30个|0.1365|0.8630|0.4164|strong overfitting|
|10.27|lightGBM|1.构造二项式特征780个; 2.互信息选择100个特征; 3.PCA降维到30个|0.1392|0.8603|0.8113|strong overfitting|

### selecting features using f-regression

|num|lgb|-|-|-|online|linear reg|-|-|-|online|
|-|-|-|-|-|-|-|-|-|-|-|
||train MSE|train R2|valid MSE|valid R2|MSE|train MSE|train R2|valid MSE|valid R2|MSE|
|38|0.0126|0.9867|0.1010|0.8986||0.1079|0.8875|0.1076|0.8920||
|37|0.0134|0.9859|0.1013|0.8984||0.1094|0.8860|0.1062|0.8934||
|36|0.0156|0.9836|0.1015|0.8981||0.1094|0.8860|0.1061|0.8935||
|35|0.0150|0.9843|0.1005|0.8991||0.1094|0.8859|0.1061|0.8935||
|34|0.0138|0.9855|0.1016|0.8980||0.1095|0.8859|0.1067|0.8930||
|33|0.0097|0.9898|0.1012|0.8984||0.1097|0.8857|0.1065|0.8931||
|32|0.0113|0.9881|0.1019|0.8977||0.1097|0.8857|0.1069|0.8927||
|31|0.0124|0.9870|0.1020|0.8976||0.1098|0.8856|0.1069|0.8927||
|30|0.0204|0.9786|0.1027|0.8969||0.1099|0.8855|0.1064|0.8932||
|29|0.0291|0.9696|0.1053|0.8943||0.1115|0.8838|0.1074|0.8923||
|28|0.0303|0.9683|0.1056|0.8941||0.1132|0.8820|0.1105|0.8891||
|27|0.0255|0.9733|0.1034|0.8962||0.1139|0.8814|0.1108|0.8888||
|26|0.0251|0.9738|0.1047|0.8949||0.1139|0.8813|0.1107|0.8889||
|25|0.0261|0.9728|0.1042|0.8954||0.1155|0.8796|0.1136|0.8860||
|24|0.0240|0.9749|0.1037|0.8959|||||||
|23|0.0335|0.9650|0.1073|0.8923|||||||
|22|0.0305|0.9681|0.1056|0.8940|||||||
|21|0.0224|0.9766|0.1051|0.8945|||||||
|20|0.0315|0.9671|0.1073|0.8923|||||||
|19|0.0368|0.9616|0.1071|0.8926|||||||
|18|0.0407|0.9575|0.1075|0.8921|||||||
|17|0.0398|0.9584|0.1072|0.8925||0.1182|0.8768|0.1177|0.8818||
|16|0.0429|0.9553|0.1082|0.8914|0.1427||||||
|15|0.0556|0.9420|0.1102|0.8895|||||||
|14|0.0629|0.9344|0.1131|0.8865|||||||
|13|0.0656|0.9316|0.1138|0.8858|||||||
|12|0.0727|0.9242|0.1176|0.8820|||||||
|11|0.0786|0.9181|0.1196|0.8800|||||||
|10|0.0911|0.9051|0.1325|0.8671|||||||
|9|0.0945|0.9015|0.1348|0.8648|||||||
|8|0.1001|0.8957|0.1380|0.8615|||||||
|7|0.1018|0.8940|0.1415|0.8580|||||||
|6|0.1017|0.8940|0.1451|0.8545|||||||
|5|0.1219|0.8730|0.1645|0.8350|||||||
|4|0.1285|0.8661|0.1669|0.8325|||||||
|3|0.1388|0.8554|0.1748|0.8247|||||||
|2|0.1638|0.8293|0.1893|0.8101|||||||
|1|0.2040|0.7875|0.2345|0.7648|||||||

### features engineering and features selection
|model|drop abnormal data|one-hot|add min|log|exp|sqrt|square|poly|drop abnormal feature|select KBest|pca|train mse|valid mse|test mse|
|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|
|lgb|-|-|-|-|-|-|-|-|-|-|-|0.0179|0.1028|-|
|lgb|True,whi=3|-|-|-|-|-|-|-|-|-|-|0.0185|0.1036|-|
|lgb|-|-|-|-|-|-|-|-|True|-|-|0.0182|0.1029|-|
|lgb|-|-|-|-|-|-|-|-|-|True,35|-|0.0189|0.1037|-|
|lgb|True,whi=3|-|-|-|-|-|-|-|True|-|-|-|-|-|
|lgb|True,whi=3|true|true|true|true|true|true|false|true|30|false|0.1082|0.1384||
|lgb|True,3|true|true|true|true|true|true|false|true|30|false||||



## summary:
- do not need too many features, maybe about 25 is a accepted value. 
- must drop abnormal-distributed features, such as 'V9'
- drop abnormal samples on train set will lead to heavily overfitting, so how to deal with these data is critical
- can't simplily drop bilinear features, which will decrease the appearance while weaken overfitting, maybe PCA is a better method
- constract many features and then use PCA on all features is a terrible try, maybe use PCA only on bilinear features is better
- use f-regression to select KBest features looks like effective, maybe combine with PCA will work better
- looks like the R2 metric stands for the ability of overcoming overfitting in some way
