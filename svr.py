import pandas as pd
import numpy as np
import math
from sklearn.metrics import mean_squared_error #均方误差
from sklearn.metrics import r2_score#R square
from sklearn.metrics import mean_absolute_error
df=pd.read_excel(r'C:/Users/limitless/Desktop/data/new94/res.xlsx')
y=pd.read_excel(r'C:/Users/limitless/Desktop/data/y.xlsx')


df = np.array(df)
#X = df.reshape((15709, 9, 4))
#df=X[:,7:8]#取我之前用到的二维矩阵中一维的数据 这里取得是五月份的
df=df.reshape((15709,36))
#取我之前用到的二维矩阵中一维的数据 这里取得是五月份的

df=pd.DataFrame(df)
#对数据进行标准化
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
scale.fit(df)
df = scale.transform(df)
df=np.array(df)
y = np.array(y)
y = y.reshape(15709, 1)
print(y[0])
import numpy as np
import sklearn.model_selection
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(df,y,test_size=0.1)
from sklearn.svm import SVR
svr = SVR(kernel='rbf',C=10000,gamma=0.5,epsilon=0.1)
svr.fit(X_train,y_train)
predict=svr.predict(df)
predict=predict.reshape(15709,1)
result = np.concatenate((y, predict), axis=1)
np.savetxt('C:/Users/limitless/Desktop/data/图/svr/result.csv', result, delimiter=',')
rmse = math.sqrt(mean_squared_error(y, predict))
r2 = r2_score(y, predict)
mae=math.sqrt(mean_absolute_error(y, predict))
print("rmse: %.4f, r2: %.4f, mae: %.4f" % (rmse, r2, mae))