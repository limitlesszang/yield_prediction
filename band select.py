import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import xlsxwriter

#compute vis(RSI)
data = xlrd.open_workbook("C:/Users/limitless/Desktop/0414zy1e.xls")
table = data.sheet_by_index(0)
myexcel=xlsxwriter.Workbook('C:/Users/limitless/Desktop/RSI.xlsx')
tt=myexcel.add_worksheet('sheet1')
for i in range(0,162):
  for j in range(i+1,163):
    ##获取查询波段号的标准差
    s1 = table.col_values(i)
    s2 = table.col_values(j)
    #print(s1)
    data_oif = list(map(lambda x:x[0]/x[1],zip(s1,s2)))#conpute rsi
    #print(len(data_oif))
    for m in range(len(data_oif)):
        tt.write(m+i*40,j, data_oif[m])
myexcel.close()

#compute the pearson-r between RSI and yield
df = pd.read_excel(r'C:/Users/limitless/Desktop/RSI414.xlsx',sheet_name='sheet1')
df = np.array(df)
myexcel=xlsxwriter.Workbook('C:/Users/limitless/Desktop/RSI(r).xlsx')
tt=myexcel.add_worksheet('sheet1')
dd=myexcel.add_worksheet('sheet2')
for i in range(0,162):
 for j in range(i+1,163):
  a=df[0:39, 0:1].reshape(-1,1)
  b=df[(i*40):(39+i*40),j:j+1].reshape(-1,1)
  a=np.squeeze(a)
  b=np.squeeze(b)
  r,p= stats.pearsonr(a,b)
  tt.write(i, j, r)
  dd.write(i,j,p)
myexcel.close()

#plot the heatmap
df = pd.read_excel(r'C:/Users/limitless/Desktop/高光谱波段选择/DN值/166波段/RSI(r).xlsx',sheet_name='sheet1')
df.index=['396',
'404',
'413',
'421',
'430',
'438',
'447',
'456',
'464',
'473',
'482',
'490',
'499',
'508',
'516',
'525',
'533',
'542',
'550',
'559',
'568',
'576',
'585',
'594',
'602',
'611',
'619',
'628',
'637',
'645',
'654',
'662',
'671',
'679',
'688',
'696',
'705',
'714',
'722',
'731',
'739',
'748',
'757',
'765',
'774',
'782',
'791',
'800',
'808',
'817',
'825',
'834',
'842',
'851',
'860',
'868',
'877',
'885',
'894',
'903',
'911',
'920',
'928',
'937',
'946',
'954',
'963',
'971',
'980',
'988',
'997',
'1006',
'1014',
'1023',
'1031',
'1040',
'1056',
'1072',
'1089',
'1106',
'1123',
'1139',
'1156',
'1173',
'1190',
'1207',
'1224',
'1241',
'1257',
'1274',
'1291',
'1308',
'1324',
'1341',
'1358',
'1375',
'1392',
'1408',
'1425',
'1442',
'1459',
'1476',
'1493',
'1509',
'1526',
'1543',
'1560',
'1577',
'1594',
'1610',
'1627',
'1644',
'1661',
'1678',
'1695',
'1711',
'1728',
'1745',
'1762',
'1779',
'1795',
'1812',
'1829',
'1845',
'1862',
'1880',
'1896',
'1913',
'1930',
'1947',
'1964',
'1981',
'1997',
'2014',
'2031',
'2048',
'2065',
'2081',
'2098',
'2115',
'2132',
'2149',
'2166',
'2183',
'2199',
'2216',
'2233',
'2250',
'2267',
'2284',
'2300',
'2317',
'2334',
'2350',
'2367',
'2384',
'2401',
'2417',
'2432',
'2451',
'2467',
'2484'
]
print(df)
ax=sns.heatmap(df, vmax=1, vmin=-1, center=0,xticklabels=24, yticklabels=24)
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=13)
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)
plt.savefig('C:/Users/limitless/Desktop/RSI.png')


