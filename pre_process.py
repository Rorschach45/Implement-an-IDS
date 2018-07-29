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
print (train_data[41].unique())