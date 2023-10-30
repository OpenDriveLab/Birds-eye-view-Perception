from petrel_client.client import Client
import pickle
import json
import cv2
import io
import torch
import numpy as np

class CephClient():
    def __init__(self, cfg, path_mapping):
        # self.client = Client('~/petreloss.conf')
        self.client = Client('~/petreloss.conf')
        self.path_mapping = path_mapping

    def get(self, data_path, dtype=np.float32):
        if self.path_mapping is not None:
            for k, v in self.path_mapping.items():
                data_path = data_path.replace(k, v)
        if(data_path.endswith('.pkl')):
            return pickle.loads(self.client.get(data_path, update_cache=True))
        elif(data_path.endswith('.json')):
            return json.loads(self.client.get(data_path, update_cache=True))
        elif(data_path.endswith('.jpg') or data_path.endswith('.png')):# image reading BGR
            value = self.client.get(data_path, update_cache=True)
            img_array = np.frombuffer(value, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img
        elif(data_path.endswith('.pth') or data_path.endswith('.pt')):
            value = self.client.get(data_path, update_cache=True)
            with io.BytesIO(value) as f:
                data = torch.load(f)
            return data
        elif(data_path.endswith('.bin')): # bin reading 
            # return np.frombuffer(self.client.get(data_path), dtype=dtype)
            # return np.load(io.BytesIO(self.client.get(data_path)), allow_pickle=True)
            return self.client.get(data_path, update_cache=True)
    
    def contains(self, data_path):
        return self.client.contains(data_path)
            