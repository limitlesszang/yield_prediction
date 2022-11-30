import pydot
import numpy as np
import pandas as pd
import scipy.stats as stats
import math
from sklearn.metrics import mean_squared_error #均方误差
from sklearn.metrics import r2_score#R square
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from sklearn import metrics
from openpyxl import load_workbook
from sklearn.tree import export_graphviz
from sklearn.ensemble import RandomForestRegressor
df=pd.read_excel(r'C:/Users/limitless/Desktop/data/new94/res.xlsx')
y=pd.read_excel(r'C:/Users/limitless/Desktop/data/y.xlsx')


df = np.array(df)
#X = df.reshape((15709, 9, 4))
#df=X[:,7:8]#取我之前用到的二维矩阵中一维的数据 这里取得是五月份的
df=df.reshape((15709,36))
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
print(y[0])
import numpy as np
import sklearn.model_selection
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(df,y,test_size=0.1)

random_seed=44
random_forest_seed=200


random_forest_model=RandomForestRegressor(n_estimators=150,random_state=random_forest_seed)
random_forest_model.fit(X_train,y_train)

# Predict test set data

random_forest_predict=random_forest_model.predict(df)
random_forest_error=random_forest_predict-y
random_forest_predict=random_forest_predict.reshape(15709,1)
result = np.concatenate((y, random_forest_predict), axis=1)
np.savetxt('C:/Users/limitless/Desktop/data/图/rf/result.csv', result, delimiter=',')
rmse = math.sqrt(mean_squared_error(y, random_forest_predict))
r2 = r2_score(y, random_forest_predict)
mae=math.sqrt(mean_absolute_error(y, random_forest_predict))
print("rmse: %.4f, r2: %.4f, mae: %.4f" % (rmse, r2, mae))
'''
plt.figure(1)
plt.clf()
ax = plt.axes(aspect='equal')
plt.scatter(y_test, random_forest_predict)
plt.xlabel('True Values')
plt.ylabel('Predictions')
Lims = [0, 7]
plt.xlim(Lims)
plt.ylim(Lims)
plt.plot(Lims, Lims)
plt.grid(False)
plt.show()

plt.figure(2)
plt.clf()
plt.hist(random_forest_error, bins=30)
plt.xlabel('Prediction Error')
plt.ylabel('Count')
plt.grid(False)
plt.show()

# Calculate the importance of variables
train_X_column_name=['evi','sr','ndwi','rep']
random_forest_importance=list(random_forest_model.feature_importances_)
random_forest_feature_importance=[(feature,round(importance,8)) 
                                  for feature, importance in zip(train_X_column_name,random_forest_importance)]
random_forest_feature_importance=sorted(random_forest_feature_importance,key=lambda x:x[1],reverse=True)
plt.figure(3)
plt.clf()
importance_plot_x_values=list(range(len(random_forest_importance)))
plt.bar(importance_plot_x_values,random_forest_importance,orientation='vertical')
plt.xticks(importance_plot_x_values,train_X_column_name,rotation='vertical')
plt.xlabel('Variable')
plt.ylabel('Importance')
plt.title('Variable Importances')
plt.show()
'''