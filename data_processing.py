import os
#import cv2
import torch
import numpy as np
import pickle
import pybamm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
pybamm.set_logging_level("DEBUG")
from torch.utils.data import Dataset, DataLoader,TensorDataset, random_split
torch.manual_seed(42)

def get_dataset(fea_flag,batch_size):
    df=torch.load('data/all_1C_2C_merged_data.pth')
    It=np.array(df['current'].tolist()).reshape(-1,1)
    Vt = np.array(df['voltage'].tolist()).reshape(-1, 1)
    Uoc = np.array(df['OCV'].tolist()).reshape(-1, 1)
    U0 = np.array(df['U0'].tolist()).reshape(-1, 1)
    U1 = np.array(df['U1'].tolist()).reshape(-1, 1)
    U2 = np.array(df['U2'].tolist()).reshape(-1, 1)
    R0 = np.array(df['R0'].tolist()).reshape(-1, 1)
    R1 = np.array(df['R1'].tolist()).reshape(-1, 1)
    R2 = np.array(df['R2'].tolist()).reshape(-1, 1)
    soc=np.array(df['SOC'].tolist()).reshape(-1,1)
    curr = np.array(df['current'].tolist()).reshape(-1, 1)
    dqdv=np.array([df['ICA'].tolist(),df['voltage'].tolist()]).T
    dqdvocv = np.array([df['ICAocv'].tolist(), df['OCV'].tolist()]).T
    temp=np.array([df['high temperature'].tolist(),df['low temperature'].tolist(),df['aveg temperature'].tolist()]).T
    images=np.array(df['frames'].tolist())
    #images=np.gradient(images,axis=2)
    resized_images = np.array([downsample_2d_array(img/57, (2, 2)) for img in images])
    #resized_images = np.array([cv2.resize(img / 57, (32, 12),cv2.INTER_CUBIC) for img in images])
    X=torch.tensor(resized_images).unsqueeze(1)
    if fea_flag=='IV':
        #Y = np.concatenate((soc,It, Vt), axis=1)
        Y=np.concatenate((It,Vt), axis=1)
    elif fea_flag=='ICA':
        #Y = np.concatenate((soc,dqdv), axis=1)
        Y = dqdv
    elif fea_flag == 'ICAOCV':
        Y=np.concatenate((Uoc,U0,U1, U2), axis=1)
    elif fea_flag=='Temp':
        Y=np.concatenate((soc,temp),axis=1)
        #Y=temp
    else:
        Y=np.concatenate((curr,dqdv,dqdvocv[0,:]),axis=1)
    #dataset = MultiFeatureDataset(arrs)
    #scaler = StandardScaler()
    #Y = scaler.fit_transform(Y)
    Y = torch.tensor(Y)
    dataset=TensorDataset(X.to(dtype=torch.float32),Y.to(dtype=torch.float32))
    dataset_size = len(dataset)
    train_size = int(0.8 * dataset_size)  # 80% training set
    test_size = dataset_size - train_size  # 20% testing set
    generator = torch.Generator().manual_seed(42)
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size],generator=generator)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_dataloader,test_dataloader
