import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import KFold
import warnings
import xgboost as xgb
from sklearn.decomposition import PCA
from sklearn import preprocessing
import math
import os

#Compute the Zero Cross
def ZeroCrossingRate(alist):
    count=0
    f_value = alist[0]
    for b_index in range(1,len(alist)):
        b_value=alist[b_index]
        if (f_value*b_value < 0):
            count += 1
        f_value = b_value
    
    return count/(len(alist)-1)    

filelist = os.listdir('F:/ingv/train/')
print('the number of data files',len(filelist))

#All features
featureName_list=['seg_id',
                'sensor_1_avg','sensor_1_sd','sensor_1_max','sensor_1_min','sensor_1_q25','sensor_1_q50',
                'sensor_1_q75','sensor_1_zc','sensor_1_85ro','sensor_1_ku','sensor_1_sk',
                'sensor_2_avg','sensor_2_sd','sensor_2_max','sensor_2_min','sensor_2_q25','sensor_2_q50',
                'sensor_2_q75','sensor_2_zc','sensor_2_85ro','sensor_2_ku','sensor_2_sk',
                'sensor_3_avg','sensor_3_sd','sensor_3_max','sensor_3_min','sensor_3_q25','sensor_3_q50',
                'sensor_3_q75','sensor_3_zc','sensor_3_85ro','sensor_3_ku','sensor_3_sk',
                'sensor_4_avg','sensor_4_sd','sensor_4_max','sensor_4_min','sensor_4_q25','sensor_4_q50',
                'sensor_4_q75','sensor_4_zc','sensor_4_85ro','sensor_4_ku','sensor_4_sk',
                'sensor_5_avg','sensor_5_sd','sensor_5_max','sensor_5_min','sensor_5_q25','sensor_5_q50',
                'sensor_5_q75','sensor_5_zc','sensor_5_85ro','sensor_5_ku','sensor_5_sk',
                'sensor_6_avg','sensor_6_sd','sensor_6_max','sensor_6_min','sensor_6_q25','sensor_6_q50',
                'sensor_6_q75','sensor_6_zc','sensor_6_85ro','sensor_6_ku','sensor_6_sk',
                'sensor_7_avg','sensor_7_sd','sensor_7_max','sensor_7_min','sensor_7_q25','sensor_7_q50',
                'sensor_7_q75','sensor_7_zc','sensor_7_85ro','sensor_7_ku','sensor_7_sk',
                'sensor_8_avg','sensor_8_sd','sensor_8_max','sensor_8_min','sensor_8_q25','sensor_8_q50',
                'sensor_8_q75','sensor_8_zc','sensor_8_85ro','sensor_8_ku','sensor_8_sk',
                'sensor_9_avg','sensor_9_sd','sensor_9_max','sensor_9_min','sensor_9_q25','sensor_9_q50',
                'sensor_9_q75','sensor_9_zc','sensor_9_85ro','sensor_9_ku','sensor_9_sk',
                'sensor_10_avg','sensor_10_sd','sensor_10_max','sensor_10_min','sensor_10_q25','sensor_10_q50',
                'sensor_10_q75','sensor_10_zc','sensor_10_85ro','sensor_10_ku','sensor_10_sk']

#features extraction
result_df=pd.DataFrame(columns=featureName_list)
i = 0

path = 'F:/ingv/ingv_mid_dataset.csv'
if os.path.exists(path):
    os.remove(path)  

midf=open(path,'a')
featurerow=''
for feature in result_df.columns:
    featurerow=featurerow+feature+','

featurerow=featurerow[:-1]+'\n'
midf.write(featurerow)

for filename in filelist:
    one_row_df=pd.DataFrame(columns=featureName_list,index=[i])
    one_row_df.iloc[0]=0
    train_df = pd.read_csv('F:/ingv/train/'+filename)
    one_row_df.loc[i,'seg_id']=filename.split('.')[0]
    
    for sensorindex in range(1,11):
        
        avg=train_df['sensor_'+str(sensorindex)].mean()
        if not math.isnan(avg):
            one_row_df.loc[i,'sensor_'+str(sensorindex)+'_avg']=avg
            
        sd=train_df['sensor_'+str(sensorindex)].std()
        if not math.isnan(sd):
            one_row_df.loc[i,'sensor_'+str(sensorindex)+'_sd']=sd
            
        smax=train_df['sensor_'+str(sensorindex)].max()
        if not math.isnan(smax):
            one_row_df.loc[i,'sensor_'+str(sensorindex)+'_max']=smax
            
        smin=train_df['sensor_'+str(sensorindex)].min()
        if not math.isnan(smin):
            one_row_df.loc[i,'sensor_'+str(sensorindex)+'_min']=smin
            
        q25=train_df['sensor_'+str(sensorindex)].quantile(0.25)
        if not math.isnan(q25):
            one_row_df.loc[i,'sensor_'+str(sensorindex)+'_q25']=q25
            
        q50=train_df['sensor_'+str(sensorindex)].quantile(0.5)
        if not math.isnan(q50):
            one_row_df.loc[i,'sensor_'+str(sensorindex)+'_q50']=q50
            
        q75=train_df['sensor_'+str(sensorindex)].quantile(0.75)
        if not math.isnan(q75):
            one_row_df.loc[i,'sensor_'+str(sensorindex)+'_q75']=q75
            
        zc=ZeroCrossingRate(train_df['sensor_'+str(sensorindex)])
        if not math.isnan(zc):
            one_row_df.loc[i,'sensor_'+str(sensorindex)+'_zc']=zc  
            
#         ro=train_df['sensor_'+str(sensorindex)].mean()
#         if not math.isnan(avg):
#             one_row_df.loc[i,'sensor_'+str(sensorindex)+'_avg']=avg
            
        ku=train_df['sensor_'+str(sensorindex)].kurt()
        if not math.isnan(ku):
            one_row_df.loc[i,'sensor_'+str(sensorindex)+'_ku']=ku
            
        sk=train_df['sensor_'+str(sensorindex)].skew()
        if not math.isnan(sk):
            one_row_df.loc[i,'sensor_'+str(sensorindex)+'_sk']=sk
                
    
    datarow=''
    for data in one_row_df.iloc[0]:
        datarow=datarow+str(data)+','

    datarow=datarow[:-1]+'\n'
    midf.write(datarow)
    
    result_df=result_df.append(one_row_df)
    
    i += 1
    if(i%100 == 0):
        print(i)

midf.close()

label_df=pd.read_csv('F:/ingv/train.csv')

for segmentid in result_df['seg_id']:
    result_id = result_df[result_df['seg_id'] == segmentid].index
    label_id =  label_df[label_df['segment_id'] == int(segmentid)].index
    result_df.loc[result_id,'label'] = float(label_df.loc[label_id,'time_to_eruption'])

df = result_df

del df['sensor_1_85ro'] 
del df['sensor_2_85ro'] 
del df['sensor_3_85ro'] 
del df['sensor_4_85ro'] 
del df['sensor_5_85ro'] 
del df['sensor_6_85ro'] 
del df['sensor_7_85ro'] 
del df['sensor_8_85ro'] 
del df['sensor_9_85ro'] 
del df['sensor_10_85ro']

#split the data set
data=df.iloc[:,1:-1]
label=df.iloc[:,-1]

train_X,test_X,train_y,test_y = train_test_split(data,label,test_size=0.3,random_state=5)

# linear regression
model1 = linear_model.LinearRegression()
model1.fit(train_X, train_y)

predict1 = model1.predict(test_X)

print('the prediction of linear regression:',predict1)

plt.figure(figsize=(20,10))
plt.plot(range(400,500),test_y[400:500], label='ture label')
plt.plot(range(400,500),predict1[400:500], label='predict label')
font1 = {'family' : 'Times New Roman',
         'weight' : 'normal',
         'size'   : 19}
plt.legend(prop=font1)
plt.tick_params(labelsize=15)
plt.xlabel('segment id',font1)
plt.ylabel('time toeruption',font1)
plt.savefig('F:/ingv/f1.jpg',dpi = 100)

print('the MAE of linear regression:',mean_absolute_error(test_y,predict1))

df.drop([random.randint(0,len(df)-1)],axis=0,inplace=True)

# linear regression with cross validation
mae_list=[]
kf = KFold(n_splits=10)

for train_index, test_index in kf.split(df):
    train_X = df.iloc[train_index, 1:-1]
    test_X = df.iloc[test_index, 1:-1]
    train_y = df.iloc[train_index, -1]
    test_y = df.iloc[test_index, -1]
    
    model2 = linear_model.LinearRegression()
    model2.fit(train_X, train_y)
    predict2 = model2.predict(test_X)
    mae_list.append(mean_absolute_error(test_y,predict2))

print('the MAE of linear regression with cross validation:',np.mean(mae_list))

# Ridge regression
mean_mae_list = []
for a_para in range(1,11):
    mae_list=[]
    kf = KFold(n_splits=10)

    for train_index, test_index in kf.split(df):
        train_X = df.iloc[train_index, 1:-1]
        test_X = df.iloc[test_index, 1:-1]
        train_y = df.iloc[train_index, -1]
        test_y = df.iloc[test_index, -1]

        model3 = linear_model.Ridge(alpha=a_para)
        model3.fit(train_X, train_y)
        predict3 = model3.predict(test_X)
        mae_list.append(mean_absolute_error(test_y,predict3))

    mean_mae_list.append(np.mean(mae_list))

plt.figure(figsize=(5,5))
plt.plot(range(1,11),mean_mae_list, label='mean absolute error')
plt.legend()
plt.savefig('F:/ingv/f2.jpg',dpi = 100)

mean_mae_list = []
for a_para in np.arange(1,3,0.1):
    mae_list=[]
    kf = KFold(n_splits=10)

    for train_index, test_index in kf.split(df):
        train_X = df.iloc[train_index, 1:-1]
        test_X = df.iloc[test_index, 1:-1]
        train_y = df.iloc[train_index, -1]
        test_y = df.iloc[test_index, -1]

        model4 = linear_model.Ridge(alpha=a_para)
        model4.fit(train_X, train_y)
        predict4 = model4.predict(test_X)
        mae_list.append(mean_absolute_error(test_y,predict4))

    mean_mae_list.append((a_para,np.mean(mae_list)))

first_alpha = sorted(mean_mae_list, key = lambda x:x[1])[0]
print('The optimal alpha:',first_alpha[0],'MAE:',first_alpha[1])

# lasso regression
warnings.filterwarnings('ignore')
mean_mae_list = []
for a_para in range(1,12001,100):
    mae_list=[]
    kf = KFold(n_splits=10)

    for train_index, test_index in kf.split(df):
        train_X = df.iloc[train_index, 1:-1]
        test_X = df.iloc[test_index, 1:-1]
        train_y = df.iloc[train_index, -1]
        test_y = df.iloc[test_index, -1]

        model5 = linear_model.Lasso(alpha=a_para)
        model5.fit(train_X, train_y)
        predict5 = model5.predict(test_X)
        mae_list.append(mean_absolute_error(test_y,predict5))

    mean_mae_list.append(np.mean(mae_list))
    
warnings.filterwarnings('default')

plt.figure(figsize=(5,5))
plt.plot(range(1,12001,100), mean_mae_list, label='mean absolute error')
plt.legend()
plt.savefig('F:/ingv/f3.jpg',dpi = 100)

warnings.filterwarnings('ignore')
mean_mae_list = []
for a_para in range(9850,9871,1):
    mae_list=[]
    kf = KFold(n_splits=10)

    for train_index, test_index in kf.split(df):
        train_X = df.iloc[train_index, 1:-1]
        test_X = df.iloc[test_index, 1:-1]
        train_y = df.iloc[train_index, -1]
        test_y = df.iloc[test_index, -1]

        model6 = linear_model.Lasso(alpha=a_para)
        model6.fit(train_X, train_y)
        predict6 = model6.predict(test_X)
        mae_list.append(mean_absolute_error(test_y,predict6))

    mean_mae_list.append((a_para,np.mean(mae_list)))
    
warnings.filterwarnings('default')

first_alpha = sorted(mean_mae_list, key = lambda x:x[1])[0]
print('The optimal alpha:',first_alpha[0],'MAE:',first_alpha[1])

data=df.iloc[:,1:-1]
label=df.iloc[:,-1]

train_X,test_X,train_y,test_y = train_test_split(data,label,test_size=0.3, random_state=714)
train_X = np.array(train_X)
test_X = np.array(test_X)

# XGBoost
model7=xgb.XGBRegressor()
model7.fit(train_X, train_y)
predict7 = model7.predict(test_X)

mean_absolute_error(test_y,predict7)

mae_list=[]
kf = KFold(n_splits=10)

for train_index, test_index in kf.split(df):
    train_X = np.array(df.iloc[train_index, 1:-1])
    test_X = np.array(df.iloc[test_index, 1:-1])
    train_y = df.iloc[train_index, -1]
    test_y = df.iloc[test_index, -1]
    
    model8 = xgb.XGBRegressor()
    model8.fit(train_X, train_y)
    predict8 = model8.predict(test_X)
    mae_list.append(mean_absolute_error(test_y,predict8))

print('the MAE of XGBoost:',np.mean(mae_list))

#PCA and scale

# pca = PCA(n_components='mle')
# dr_data = pca.fit_transform(np.array(df.iloc[:,1:-1]))

min_max_scaler = preprocessing.MinMaxScaler()
dr_data = min_max_scaler.fit_transform(np.array(df.iloc[:,1:-1]))

print('the shape of data after processing:',dr_data.shape)

# linear regression
mae_list=[]
kf = KFold(n_splits=10)

for train_index, test_index in kf.split(dr_data):
    train_X = dr_data[train_index, :]
    test_X = dr_data[test_index, :]
    train_y = df.iloc[train_index, -1]
    test_y = df.iloc[test_index, -1]
    
    model9 = linear_model.LinearRegression()
    model9.fit(train_X, train_y)
    predict9 = model9.predict(test_X)
    mae_list.append(mean_absolute_error(test_y,predict9))

print('the MAE of linear regression:',np.mean(mae_list))

# Ridge regression
mean_mae_list = []
for a_para in np.arange(1,10,0.1):
    mae_list=[]
    kf = KFold(n_splits=10)

    for train_index, test_index in kf.split(dr_data):
        train_X = dr_data[train_index, :]
        test_X = dr_data[test_index, :]
        train_y = df.iloc[train_index, -1]
        test_y = df.iloc[test_index, -1]

        model10 = linear_model.Ridge(alpha=a_para)
        model10.fit(train_X, train_y)
        predict10 = model10.predict(test_X)
        mae_list.append(mean_absolute_error(test_y,predict10))

    mean_mae_list.append((a_para,np.mean(mae_list)))

first_alpha = sorted(mean_mae_list, key = lambda x:x[1])[0]
print('The optimal alpha:',first_alpha[0],'MAE:',first_alpha[1])

# Lasso regression
warnings.filterwarnings('ignore')
mean_mae_list = []
for a_para in range(5150,5170,1):
    mae_list=[]
    kf = KFold(n_splits=10)

    for train_index, test_index in kf.split(df):
        train_X = dr_data[train_index, :]
        test_X = dr_data[test_index, :]
        train_y = df.iloc[train_index, -1]
        test_y = df.iloc[test_index, -1]

        model11 = linear_model.Lasso(alpha=a_para)
        model11.fit(train_X, train_y)
        predict11 = model11.predict(test_X)
        mae_list.append(mean_absolute_error(test_y,predict11))

    mean_mae_list.append((a_para,np.mean(mae_list)))
    
warnings.filterwarnings('default')

first_alpha = sorted(mean_mae_list, key = lambda x:x[1])[0]
print('The optimal alpha:',first_alpha[0],'MAE:',first_alpha[1])

# XGBoost
mae_list=[]
kf = KFold(n_splits=10)

for train_index, test_index in kf.split(df):
    train_X = dr_data[train_index, :]
    test_X = dr_data[test_index, :]
    train_y = df.iloc[train_index, -1]
    test_y = df.iloc[test_index, -1]
    
    model12 = xgb.XGBRegressor()
    model12.fit(train_X, train_y)
    predict12 = model12.predict(test_X)
    mae_list.append(mean_absolute_error(test_y,predict12))

print('the MAE of XGBoost:',np.mean(mae_list))