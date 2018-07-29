import pandas as pd
import weka.core.jvm as jvm
from sklearn.feature_selection import mutual_info_regression,mutual_info_classif,mutual_info_
import numpy as np

#

def encode_data():
    train_data = pd.read_csv('./data/20 Percent Training Set.csv', header=-1)
    train_data[1]=train_data[1].astype('category').cat.codes
    train_data[2]=train_data[2].astype('category').cat.codes
    train_data[3]=train_data[3].astype('category').cat.codes
    reduced_attacks={"normal": "normal","neptune":"dos", "warezclient":"r2l", "ipsweep":"probe", "portsweep":"probe", "teardrop":"dos", "nmap":"probe",
     "satan":"probe", "smurf":"dos", "pod":"dos", "back":"dos","guess_passwd":"r2l","ftp_write":"r2l","multihop":"r2l","rootkit":"u2r","buffer_overflow":"u2r","imap":"r2l","warezmaster":"r2l","phf":"r2l","land":"dos",
                     "loadmodule":"u2r","spy":"r2l"}
    train_data[41].replace(reduced_attacks,inplace=True)
    train_data.to_csv('./data/20 Percent Training Set reducedAttacks_data.csv',sep=',', encoding='utf-8',index=False,header=False)
    attck_encode={"normal":0, "dos":1 ,"r2l":2, "probe":3 ,"u2r":4}
    train_data[41].replace(attck_encode,inplace=True)
    train_data.to_csv('./data/20 Percent Training Set encoded_data.csv',sep=',', encoding='utf-8',index=False,header=False)
    two_type_encode={0:0,1:1,2:1,3:1,4:1}
    train_data[41].replace(two_type_encode,inplace=True)
    train_data.to_csv('./data/20 Percent Training Set bolean_attack.csv',sep=',', encoding='utf-8',index=False,header=False)


train_dat=pd.read_csv('./data/20 Percent Training Set bolean_attack.csv',header=-1)


train=np.asarray(train_dat)

target=train[:,41]
print(train.shape)
train=np.delete(train,41,axis=1)
print(train.shape)
mi=mutual_info_regression(train,target)
ind=np.argsort(mi,axis=0)
print("/n")
print("/n")
print(ind)
target_df=train_dat[41]
print(train_dat.shape)
ind=ind[0:ind.__len__()-8]
train_dat=train_dat.drop(ind,axis=1)
train_dat=train_dat.drop(42,axis=1)
train_dat=pd.concat([train_dat,target_df],axis=1)
print(train_dat.shape)
print(train_dat.head())
print(target_df.head())
train_dat.to_csv('./data/20 percent train set boolean feature selected.csv',sep=',', encoding='utf-8',index=False,header=False)


