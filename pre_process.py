import pandas as pd
from sklearn.feature_selection import mutual_info_regression,mutual_info_classif,mutual_info_
import numpy as np
from sklearn import preprocessing



#

def encode_data(filename):
    train_data = pd.read_csv(filename+'.csv', header=-1)
    train_data[1]=train_data[1].astype('category').cat.codes
    train_data[2]=train_data[2].astype('category').cat.codes
    train_data[3]=train_data[3].astype('category').cat.codes
    reduced_attacks={"normal": "normal","neptune":"dos", "warezclient":"r2l", "ipsweep":"probe", "portsweep":"probe", "teardrop":"dos", "nmap":"probe",
     "satan":"probe", "smurf":"dos", "pod":"dos", "back":"dos","guess_passwd":"r2l","ftp_write":"r2l","multihop":"r2l","rootkit":"u2r","buffer_overflow":"u2r","imap":"r2l","warezmaster":"r2l","phf":"r2l","land":"dos",
                     "loadmodule":"u2r","spy":"r2l"}
    train_data[41].replace(reduced_attacks,inplace=True)
    train_data.to_csv(filename+'reducedAttacks_data.csv',sep=',', encoding='utf-8',index=False,header=False)
    attck_encode={"normal":0, "dos":1 ,"r2l":2, "probe":3 ,"u2r":4}
    train_data[41].replace(attck_encode,inplace=True)
    train_data.to_csv(filename+'encoded_data.csv',sep=',', encoding='utf-8',index=False,header=False)
    two_type_encode={0:0,1:1,2:1,3:1,4:1}
    train_data[41].replace(two_type_encode,inplace=True)
    train_data.to_csv(filename+'bolean_attack.csv',sep=',', encoding='utf-8',index=False,header=False)


def slected_feature_data_sets_save(ind,filename):
    train_dat=pd.read_csv(filename+'.csv',header=-1)
    print(train_dat.shape)
    target_df=train_dat[41]
    train_dat=train_dat.drop(ind,axis=1)
    train_dat=train_dat.drop(42,axis=1)
    train_dat=pd.concat([train_dat,target_df],axis=1)
    train_dat.to_csv(filename + 'feature selected.csv',sep=',', encoding='utf-8',index=False,header=False)


def find_features():
    train_dat=pd.read_csv('./data/20 Percent Training Set bolean_attack.csv',header=-1)
    print(train_dat.shape)
    train=np.asarray(train_dat)
    target=train[:,41]
    train=np.delete(train,41,axis=1)
    mi=mutual_info_regression(train,target)
    ind=np.argsort(mi,axis=0)
    target_df=train_dat[41]
    ind=ind[0:ind.__len__()-8]
    train_dat=train_dat.drop(ind,axis=1)
    train_dat=train_dat.drop(42,axis=1)
    train_dat=pd.concat([train_dat,target_df],axis=1)
    train_dat.to_csv('./data/20 percent train set boolean feature selected.csv',sep=',', encoding='utf-8',index=False,header=False)
    slected_feature_data_sets_save(ind,'./data/20 Percent Training Set encoded_data')
    slected_feature_data_sets_save(ind,'./data/20 Percent Training Set reducedAttacks_data')
def normalized_data(filename):

    data=pd.read_csv('./data/'+filename+'.csv',header=-1)
    target=data[8]
    data=data.drop(8,axis=1)
    x=data.values
    min_max_scaler=preprocessing.MinMaxScaler()
    x_scale=min_max_scaler.fit_transform(x)
    data=pd.DataFrame(x_scale)
    data=pd.concat([data,target],axis=1)
    data.columns=["src_bytes", "service","dst_bytes","flag","diff_srv_rate","same_srv_ratedst"
        ,"host srv countdst","host same srv rate","target"]
    data.to_csv('./data/final/'+filename+' with normalized data'+'.csv',sep=',', encoding='utf-8',index=False,header=True)
    print(data.head())

#find_features()
normalized_data('20 Percent Training Set reducedAttacks_datafeature selected')
normalized_data('20 Percent Training Set encoded_datafeature selected')
normalized_data('20 percent train set boolean feature selected')
vt=1;