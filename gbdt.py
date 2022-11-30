import pandas as pd
import numpy as np
import math
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import Lasso
from lightgbm.sklearn import LGBMRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error #均方误差
from sklearn.metrics import r2_score#R square
from sklearn.metrics import mean_absolute_error
from sklearn import tree
model2_tree = tree.DecisionTreeRegressor()
from sklearn.ensemble import GradientBoostingRegressor
model7_GBDT = GradientBoostingRegressor()


df=pd.read_excel(r'C:/Users/limitless/Desktop/data/new94/res.xlsx')
y=pd.read_excel(r'C:/Users/limitless/Desktop/data/y.xlsx')


df = np.array(df)
df = df.reshape((15709,36))
print(df[0])
df=pd.DataFrame(df)
#对数据进行标准化
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
scale.fit(df)
df = scale.transform(df)
df=np.array(df)
y = np.array(y)
y = y.reshape(15709, 1)


import numpy as np
import sklearn.model_selection
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(df,y,test_size=0.1)

'''
lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))
predict = lasso.fit(X_train, y_train).predict(X_test)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(df,y,test_size=0.1)
model2_tree = model2_tree.fit(X_train,y_train)
predict = model2_tree.predict(X_test)
'''

model7_GBDT = GradientBoostingRegressor(n_estimators=1250,learning_rate=0.1,subsample=0.95)
model7_GBDT = model7_GBDT.fit(X_train,y_train)
predict = model7_GBDT.predict(df)
predict=predict.reshape(15709,1)
result = np.concatenate((y, predict), axis=1)
np.savetxt('C:/Users/limitless/Desktop/data/图/gbdt/result.csv', result, delimiter=',')
rmse = math.sqrt(mean_squared_error(y, predict))
r2 = r2_score(y, predict)
mae=math.sqrt(mean_absolute_error(y, predict))
print("rmse: %.4f, r2: %.4f, mae: %.4f" % (rmse, r2, mae))





