#import sys
import os
import time

import h5py
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from torch.utils.data import DataLoader

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

class Dataset(torch.utils.data.Dataset):
    """
    dataset = Dataset(data_type, df)

    img, y = dataset[i]
      img (np.float32): 2 x 360 x 128
      y (np.float32): label 0 or 1
    """
    def __init__(self, data_type, df):
        self.data_type = data_type
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        """
        i (int): get ith data
        """
        r = self.df.iloc[i]
        y = np.float32(r.target)
        file_id = r.id

        img = np.empty((2, 360, 128), dtype=np.float32)

        #filename = '%s/%s/%s.hdf5' % (di, self.data_type, file_id)
        filename = f"{self.data_type}/{file_id}.hdf5"
        with h5py.File(filename, 'r') as f:
            g = f[file_id]

            for ch, s in enumerate(['H1', 'L1']):
                a = g[s]['SFTs'][:, :4096] * 1e22  # Fourier coefficient complex64

                p = a.real**2 + a.imag**2  # power
                p /= np.mean(p)  # normalize
                p = np.mean(p.reshape(360, 128, 32), axis=2)  # compress 4096 -> 128

                img[ch] = p

        return img, y

class Model(nn.Module):
    def __init__(self, name, *, pretrained=False):
        """
        name (str): timm model name, e.g. tf_efficientnet_b2_ns
        """
        super().__init__()

        # Use timm
        model = timm.create_model(name, pretrained=pretrained, in_chans=2)

        clsf = model.default_cfg['classifier']
        n_features = model._modules[clsf].in_features
        model._modules[clsf] = nn.Identity()

        self.fc = nn.Linear(n_features, 1)
        self.model = model

    def forward(self, x):
        x = self.model(x)
        x = self.fc(x)
        return x

# Load model (if necessary)
model_cpu = Model('tf_efficientnet_b2_ns', pretrained=False)
device =  torch.device('cpu')
model_cpu.to(torch.device(device))
model_cpu.load_state_dict(torch.load('/code/gravitw/api/model00.pytorch', map_location=device))

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    return {"filename": file.filename}

@app.post('/predict_image')
async def predict_image(hdf5_file: UploadFile = File(...)):

    if hdf5_file is None:
        raise HTTPException(status_code=400, detail="No file was uploaded.")

    # Save the file
    with open(hdf5_file.filename, "wb") as file:
        file.write(hdf5_file.file.read())

    id_ = [f"{hdf5_file.filename.split('.')[0]}"]  # Create a label.csv   PG is id's name. the num in range is the id
    label_ = pd.DataFrame(data=id_, columns=["id"])
    label_["target"] = 0.5

    # make the prediction
    dataset_test = Dataset("./", label_)
    loader_test = DataLoader(dataset_test, batch_size=64, num_workers=2, pin_memory=True)

    tb = time.time()
    img, y = next(iter(loader_test))
    img = img.to('cpu')
    y = y.to('cpu')

    with torch.no_grad():
        y_pred = model_cpu(img.to(device))

    res = {'prediction': float(y_pred.sigmoid().squeeze().cpu().detach().numpy()), 'time':f"{time.time() - tb:.2f}"}

    return res

@app.get("/")
def root():
    return     {
    'greeting': 'Hello'
    }
