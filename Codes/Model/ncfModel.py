import torch
import torch.nn as nn

class MovieModel(nn.Module):
    def __init__(self, num_users, num_items, num_factors=40, dropout=0.5, num_layers=3, hidden_size=256):
        super(MovieModel, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.num_factors = num_factors
        self.dropout = dropout
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        
        self.user_embedding = nn.Embedding(self.num_users, self.num_factors)
        self.item_embedding = nn.Embedding(self.num_items, self.num_factors)
        
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(self.num_factors*2, self.hidden_size))
        for _ in range(self.num_layers-1):
            self.layers.append(nn.Linear(self.hidden_size, self.hidden_size))
        self.layers.append(nn.Linear(self.hidden_size, 1))
        
        self.dropout = nn.Dropout(self.dropout)

    def interaction2vec(self, item, user):
        user_embedding = self.user_embedding(user)
        item_embedding = self.item_embedding(item)
        return torch.cat([user_embedding, item_embedding], dim=1)
     
    def forward(self, x):
        # for layer in self.layers[:-1]:
        #     x = layer(x)
        #     x = nn.ReLU()(x)
        #     x = self.dropout(x)
        # x = self.layers[-1](x)
        #return x
        for layer in self.layers:
            x = self.dropout(x)
            x = layer(x)
        return x.squeeze()
        
    

        
    
    def predict(self, user, item):
        return self.forward(self.interaction2vec(item, user))
    
    def get_loss(self, user, item, rating):
        prediction = self.forward(self.interaction2vec(item, user))
        return nn.MSELoss()(prediction, rating)
    
    def get_topk(self, user, item, topk=10):
        prediction = self.forward(self.interaction2vec(item, user))
        return torch.topk(prediction, topk)
    
    def get_user_embedding(self, user):
        return self.user_embedding(user)
    
    def get_item_embedding(self, item):
        return self.item_embedding(item)
    
    def get_user_item_embedding(self, user, item):
        return self.user_embedding(user), self.item_embedding(item)
    