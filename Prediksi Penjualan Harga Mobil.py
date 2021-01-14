#!/usr/bin/env python
# coding: utf-8

# In[73]:


#IMPORT LIBRARY

import numpy as np # for math. operations
import pandas as pd # for data manipulation operations
import statsmodels.api as plt # for building of models
from sklearn.linear_model import LinearRegression # for building of machine learning models
import seaborn as sns # for advanced data visualizations
sns.set() # activate


# In[74]:


#LOAD DATA

raw_data = pd.read_csv('D:\Pintaar\ds_learning\ds_project\Dataset Penjualan Mobil.csv')
raw_data.head(20)


# In[75]:


#DATA PREPROCESSING
#Exploring the descriptive statistics

raw_data.describe(include='all') #descriptive statistics for all variable, if only numerical variable can use -> raw_data.describe()


# In[76]:


#Determine unnecessary variable

data = raw_data.drop(['Model'],axis=1) #axis for coloumn
data.describe(include='all')


# In[77]:


#Dealing with missing values

data.isnull().sum() #.sum() to see the number of missing in each variable

#False (0) is the data exist, but True (1) is data missing
#There are 2 variable have missing data value, that are price (172) dan EngineV (150) from 4173 and 4345 (5%) total calculated data


# In[78]:


data_no_mv = data.dropna(axis=0) #droping baris (axis=0); coloumn (axis=1)
data_no_mv.describe(include='all')


# In[79]:


#Exploring the probability distributin functions (PDFs)

sns.distplot(data_no_mv['Price'])

#The graph showing an exponential graph, but the linear regression must normal distibution


# In[80]:


#DEALING WITH OUTLIERS
#There is high data in outside the normal range (outliers)

q = data_no_mv['Price'].quantile(0.99) #99% quantile data yang ada
q


# In[81]:


data_1 = data_no_mv[data_no_mv['Price'] < q] #Menghilangkan data 1% data teratas


# In[82]:


sns.distplot(data_1['Price'])

#the graph shows an exponential graph, but not as severe as the previous data/chart


# In[83]:


sns.distplot(data_1['Mileage'])


# In[84]:


q = data_1['Mileage'].quantile(0.99) #99% quantile data yang ada
data_2 = data_1[data_1['Mileage'] < q] #Menghilangkan data 1% data teratas


# In[85]:


sns.distplot(data_2['Mileage'])

#the graph shows an exponential graph, but not as severe as the previous data/chart


# In[86]:


sns.distplot(data_2['EngineV'])

#Grafik sangat aneh


# In[87]:


data_2['EngineV'].head() #Masuk akal


# In[88]:


data_2['EngineV'].tail() #Masuk akal


# In[89]:


data_2['EngineV'].sort_values() #Nilai di shorting dari terkecil hingga ke terbesar

#Engine Capacity (CC) volume range tidak tepat bisa hingga 99.99


# In[90]:


data_3 = data_2[data_2['EngineV'] < 6.5] #Menghilangkan data 1% data teratas
sns.distplot(data_3['EngineV'])

#the graph shows an exponential graph, but not as severe as the previous data/chart


# In[91]:


sns.distplot(data_3['Year'])


# In[92]:


q = data_3['Year'].quantile(0.01) #Quantile 1% data terbawah yang akan dihilangkan
data_4 = data_3[data_3['Year'] > q]

sns.distplot(data_4['Year'])


# In[93]:


#Data sudah bersih

data_cleaned = data_4.reset_index(drop=True)


# In[94]:


data_cleaned.describe(include='all')


# In[ ]:





# In[95]:


#EXAMMING ASSUMPTIONS


# In[96]:


import matplotlib.pyplot as plt


# In[97]:


#Check Linearity with Scatterplot

f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize =(15,3)) #sharey -> share 'Price' as y
ax1.scatter(data_cleaned['Year'],data_cleaned['Price'])
ax1.set_title('Price and Year')
ax2.scatter(data_cleaned['EngineV'],data_cleaned['Price'])
ax2.set_title('Price and EngineV')
ax3.scatter(data_cleaned['Mileage'],data_cleaned['Price'])
ax3.set_title('Price and Mileage')


plt.show()

#ax1 -> datanya masih eksponensial, ax2 -> data tidak dapat dijelaskan, ax3 -> data juga eksponensial


# In[98]:


sns.distplot(data_cleaned['Price'])

#Gambar di atas dikarenakan Price masih eksponensial


# In[99]:


#RELAXING THE ASSUMPTION (modifikasi data)

#Log Transformation
log_price = np.log(data_cleaned['Price'])
data_cleaned['log_price'] = log_price

data_cleaned.head()

#Maka terdapat variabel baru: log_price


# In[165]:


f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize =(15,3)) #sharey -> share 'Price' as y
ax1.scatter(data_cleaned['Year'],data_cleaned['log_price'])
ax1.set_title('Log Price and Year')
ax2.scatter(data_cleaned['EngineV'],data_cleaned['log_price'])
ax2.set_title('Log Price and EngineV')
ax3.scatter(data_cleaned['Mileage'],data_cleaned['log_price'])
ax3.set_title('Log Price and Mileage')


plt.show()

#data sudah lebih menunjukkan lebih linear dibandingkan sebelumnya
#Scatter tidak terlalu jauh -> homoscedasticity telah tercapai


# In[101]:


data_cleaned = data_cleaned.drop(['Price'],axis=1) #axis for coloumn
data_cleaned.head()


# In[102]:


#Normalitas tercapai dikarenakan jumlah data yang besar (>3000 data)


# In[103]:


#MULTICOLLINEARITY

from statsmodels.stats.outliers_influence import variance_inflation_factor
variables = data_cleaned[['Mileage','Year','EngineV']]
vif = pd.DataFrame()

vif["VIF"] = [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]
vif["Features"] = variables.columns


# In[104]:


vif

#baris 0 -> ada sedikit multicollinearity, baris 1 -> sangat tinggi multicollinearity & baris 2 -> sangat tinggi multicollinearity tetapi masih mendekati 6
#sehinga yang digunakan hanya baris 0 dan 2 saja, baris 1 di drop


# In[105]:


data_no_multi = data_cleaned.drop(['Year'],axis=1)


# In[106]:


data_no_multi.head()


# In[107]:


#CREATE DUMMY VARIABLES

data_with_dummies = pd.get_dummies(data_no_multi, drop_first=True)

#drop_first untuk menghilangkan 1 di setiap variabel


# In[108]:


data_with_dummies.head()


# In[109]:


#REARRANGE THE COLUMN

data_with_dummies.columns.values

#Memindah estimator log_price di depan untuk lebih mudah


# In[110]:


cols = ['log_price', 'Mileage', 'EngineV', 'Brand_BMW',
       'Brand_Mercedes-Benz', 'Brand_Mitsubishi', 'Brand_Renault',
       'Brand_Toyota', 'Brand_Volkswagen', 'Body_hatch', 'Body_other',
       'Body_sedan', 'Body_vagon', 'Body_van', 'Engine Type_Gas',
       'Engine Type_Other', 'Engine Type_Petrol', 'Registration_yes']

#log_price sudah di column depan


# In[111]:


#data preprocessed -> data yang sudah di proses dari uji asumsi hingga variabel atau column yang tidak memenuhi

data_preprocessed = data_with_dummies[cols]


# In[112]:


data_preprocessed.head()


# In[113]:


#BUILDING OF REGRESSION MODELS


# In[114]:


#declare the inputs and the targets

targets = data_preprocessed['log_price']

inputs = data_preprocessed.drop(['log_price'],axis=1) #variabel input : selain variabel log_price


# In[115]:


#scale the data : normalisasi secara otomatis dan melakukan scale up atau scale down

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(inputs) #mendapatkan nilai standar deviasi dan rata-rata dari setiap di nilai bilangan inputs


# In[116]:


inputs_scaled = scaler.transform(inputs)


# In[117]:


inputs_scaled

#nilai sudah di standarisasi


# In[118]:


#TRAIN TEST SPLIT
#Melakukan split data training dan data testing

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(inputs_scaled, targets, test_size=0.2, random_state=365) #data testing 20% (0.2), random state bebas


# In[166]:


#CREATE THE REGRESSION

reg = LinearRegression()

reg.fit(x_train, y_train)


# In[120]:


y_hat = reg.predict(x_train)


# In[121]:


plt.scatter(y_train, y_hat)
plt.xlabel('Targets (y_train)', size=18)
plt.ylabel('Predictions (y_hat)', size=18)
plt.ylim(6,13)
plt.show()

#terlihat terdapat distribusi data yang belum tepat


# In[127]:


#plotting terhadap resi 2 (perbedaan nilai antara observasi dan prediksi. Probability distribution functions

sns.distplot(y_train - y_hat)


# In[128]:


sns.distplot(y_train - y_hat)
plt.title('Residuals PDF', size=18)


# In[129]:


reg.score(x_train, y_train)

#74% menyerap variability atau valiansi data (cukup baik untuk model yang baru dijalankan)


# In[130]:


#FINDING THE WEIGHTS AND BIAS

reg.intercept_


# In[131]:


reg.coef_


# In[132]:


reg_summary = pd.DataFrame(inputs.columns.values, columns=['Features'])
reg_summary['weights'] = reg.coef_
reg_summary

#variabel metrik [nilai negatif : ketika nilai mileage naik maka nilai log_price akan turun | nilai positif : ketika nilai mileage naik maka nilai log_price akan naik juga]
#variabel kategorikal -> nilai negatif dan positif berkorelasi harga yang di benchmark (audi) terhadap setiap brand


# In[ ]:





# In[133]:


#TESTING
#Membangun, melihat, dan mengevaluasi model dengan testing


# In[134]:


y_hat_test = reg.predict(x_test)


# In[148]:


plt.scatter(y_test, y_hat_test)
plt.xlabel('Targets (y_test)', size=18)
plt.ylabel('Predictions (y_hat_test)', size=18)
plt.ylim(6,13)
plt.show()


# In[149]:


plt.scatter(y_test, y_hat_test,alpha=0.2)
plt.xlabel('Targets (y_test)', size=18)
plt.ylabel('Predictions (y_hat_test)', size=18)
plt.ylim(6,13)
plt.show()


# In[150]:


#membuat data frame baru untuk membandingkan target dan prediksi

df_perform = pd.DataFrame(y_hat_test, columns=['Prediction'])
df_perform.tail()


# In[151]:


df_perform = pd.DataFrame(np.exp(y_hat_test), columns=['Prediction'])
df_perform.head()

#mengembalikan ke log_price yang asli


# In[152]:


df_perform['Target'] = np.exp(y_test)
df_perform.head()

#Target terlihat terdapat data not available


# In[153]:


df_perform['Target'] = np.exp(y_test)
df_perform

#Target terlihat acak (ada nilai yang tidak ada), hal ini dikarenakan y_test ketika kita melakukan split training data dia mengambil testing sebesar 20% secara random, sehingga indeksnya masih mengikuti indeks yang laman


# In[154]:


df_perform['Target'] = np.exp(y_test.reset_index(drop=True))
df_perform


# In[155]:


df_perform['Residual'] = df_perform['Target'] - df_perform['Prediction']
df_perform.head()


# In[159]:


df_perform['Difference%'] = np.absolute(df_perform['Residual']/df_perform['Target']*100)
df_perform.head()


# In[160]:


df_perform


# In[162]:


df_perform.sort_values(by=['Difference%'])


# In[163]:


pd.options.display.max_rows =999
pd.set_option('display.float_format', lambda x: '%.2f' % x)
df_perform.sort_values(by=['Difference%'])


# In[164]:


df_perform.describe()


# In[ ]:




