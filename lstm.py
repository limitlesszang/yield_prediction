import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
print(tf.__version__)
print(tf.keras.__version__)


import math
from sklearn.metrics import mean_squared_error #均方误差
from sklearn.metrics import r2_score#R square
from sklearn.metrics import mean_absolute_error
import scipy.stats
import time
import os
from tensorflow.keras.layers import Bidirectional,LSTM
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

os.environ["CUDA_DEVICES_ORDERS"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class TimeHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []
        self.totaltime = time.time()

    def on_train_end(self, logs={}):
        self.totaltime = time.time() - self.totaltime

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

#prepare data from gee csv
sr=pd.read_excel(r'C:/Users/limitless/Desktop/data/new.xlsx',sheet_name='Sheet1').T
lswi=pd.read_excel(r'C:/Users/limitless/Desktop/data/new.xlsx',sheet_name='Sheet2').T
ndwi=pd.read_excel(r'C:/Users/limitless/Desktop/data/new.xlsx',sheet_name='Sheet3').T
evi=pd.read_excel(r'C:/Users/limitless/Desktop/data/new.xlsx',sheet_name='Sheet4').T
df=pd.DataFrame()
for i in range(0,8875):
    data = pd.concat([sr.iloc[:,i], lswi.iloc[:,i], ndwi.iloc[:,i]], sort=False, ignore_index=True,
                     axis=1)
    df=df.append(data)
yiel=pd.read_excel(r'C:/Users/limitless/Desktop/data/1/y8875.xlsx')

for i in range(9,0,-1):
    def Yield_LSTM_Data_Precesing(df):
        df.dropna(inplace=True)
        df.sort_index(inplace=True)
        from sklearn import preprocessing
        scaler = preprocessing.MinMaxScaler()
        sca_X = scaler.fit_transform(df)
        X = np.array(sca_X)
        X = X.reshape((15709, 9, 4))
        X=X[:,0:i]
        print(X.shape)
        yie = np.array(yiel)
        print(yie.shape)
        return X,yie

    def trainModel(train_X, train_Y,val_X,val_Y):

        model = tf.keras.Sequential()
        model.add(layers.LSTM(
            100,
            input_shape=(train_X.shape[1:]),
            return_sequences=True))
        model.add(layers.Dropout(0.3))

        model.add(layers.LSTM(
            100,
            return_sequences=False))
        model.add(layers.Dropout(0.3))

        model.add(layers.Dense(
            train_Y.shape[1]))
        model.add(layers.Activation("relu"))

        model.compile(metrics=['accuracy'], loss='mean_squared_error', optimizer='adam')
        model.summary()# 可以在调用model.compile()之前初始化一个优化器对象，然后传入该函数,修改学习率等
        chechpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath="C:/Users/limitless/Desktop/data/8875/"+ str(i) +"/model_{epoch:02d}.h5",
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            period=1)
        time_callback = TimeHistory()
        history = model.fit(train_X, train_Y, epochs=epoch, batch_size=batch_size, callbacks=[chechpoint, time_callback], verbose=2, validation_data=(val_X,val_Y))
        time_arr = np.array(time_callback.times)
        model.save("C:/Users/limitless/Desktop/data/8875/"+ str(i) +"/%d_%.4f.h5" % (batch_size, lr))
        final_model = tf.keras.models.load_model(
            "C:/Users/limitless/Desktop/data/8875/"+ str(i) +"/64_0.0010.h5")  # D:\\学习\\FVC传统估算方法\\NDVI-SWA\\REF-FVC\\reslut.h51
        pre_label = final_model.predict(val_data)
        result = np.concatenate((val_label, pre_label), axis=1)
        np.savetxt('C:/Users/limitless/Desktop/data/8875/'+ str(i) +f'/{i}_result2.csv', result, delimiter=',')  # ./result/40000/result.csv
        rmse = math.sqrt(mean_squared_error(val_label, pre_label))
        r2 = r2_score(val_label, pre_label)
        mae=math.sqrt(mean_absolute_error(val_label, pre_label))
        print("rmse: %.4f, r2: %.4f, mae:%.4f" % (rmse, r2,mae))
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(val_label[:, 0], pre_label[:, 0])
        print(r_value ** 2)
        return rmse,r2,mae

    # 超参
    lr = 0.001 # lr = 0.001
    batch_size = 64 # batch_size = 128
    epoch = 700 # epoch = 450

    if __name__ == '__main__':
        X, yie = Yield_LSTM_Data_Precesing(df)
        # 切分训练集与测试集，注意所有的交叉验证等都是在训练集上做的操作，测试集只有最后的最后才会使用到
        from sklearn.model_selection import train_test_split
        train_data,val_data,train_label,val_label = train_test_split(X, yie, test_size=0.1)
        # 使用交叉验证
        from sklearn.model_selection import KFold
        kfold = KFold(n_splits=10,shuffle=True)
    score=[]

    for train_index, test_index in kfold.split(X,yie):
            # train_index 就是分类的训练集的下标，test_index 就是分配的验证集的下标
            this_train_x, this_train_y = X[train_index], yie[train_index]  # 本组训练集
            this_test_x, this_test_y = X[test_index], yie[test_index]  # 本组验证集
            x,y,z= trainModel(this_train_x, this_train_y,this_test_x,this_test_y)
            def func(a):
                score.append(a)
                return score
            x = func(x)
            y = func(y)
            z = func(z)
            print(z)
    score=np.array(score)
    score = score.reshape(10, 3)
    np.savetxt('C:/Users/limitless/Desktop/data/8875/'+ str(i) +'/score1.csv', score, delimiter=',')
