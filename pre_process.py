import pandas as pd
import weka.core.jvm as jvm
import sklearn as sk


train_data=pd.read_csv('./data/20 Percent Training Set.csv',header=-1)
print(train_data.head())
train_data[1]=train_data[1].astype('category').cat.codes
train_data[2]=train_data[2].astype('category').cat.codes
train_data[3]=train_data[3].astype('category').cat.codes

print(train_data.head())
print(train_data.shape)
reduced_attacks={"normal": "normal","neptune":"dos", "warezclient":"r2l", "ipsweep":"probe", "portsweep":"probe", "teardrop":"dos", "nmap":"probe",
 "satan":"probe", "smurf":"dos", "pod":"dos", "back":"dos","guess_passwd":"r2l","ftp_write":"r2l","multihop":"r2l","rootkit":"u2r","buffer_overflow":"u2r","imap":"r2l","warezmaster":"r2l","phf":"r2l","land":"dos",
                 "loadmodule":"u2r","spy":"r2l"}
train_data[41].replace(reduced_attacks,inplace=True)

print (train_data[41].unique())
attck_encode={"normal":0, "dos":1 ,"r2l":2, "probe":3 ,"u2r":4}
train_data[41].replace(attck_encode,inplace=True)
print (train_data.dtypes)
print(train_data.head())
