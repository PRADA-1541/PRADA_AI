import torch
import torch.utils.data as data
import numpy as np


class Dataset(data.Dataset):
    def __init__(self, df, is_training=True):
        super(Dataset, self).__init__()
        self.dataset = np.array(df)
        self.is_training = is_training

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # user의 경우 1부터 시작하므로 1을 빼준다.
        user = (self.dataset[idx][0]).astype(np.int64)
        item = (self.dataset[idx][1]).astype(np.int64)
        rating = self.dataset[idx][2].astype(np.float32)

        return user, item, rating
    
    def collate_fn(self, data):
        users, items, ratings = zip(*data)
        users = torch.LongTensor(users)
        items = torch.LongTensor(items)
        ratings = torch.FloatTensor(ratings)
        return users, items, ratings.dtype(torch.float32)
        
    
    def get_loader(self, batch_size, shuffle=True, num_workers=4):
        if self.is_training:
            return data.DataLoader(self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        else:
            return data.DataLoader(self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    
    