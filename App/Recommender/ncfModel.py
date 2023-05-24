import torch
import torch.nn as nn

class NCF(nn.Module):
    """
        :param num_users: number of users
        :param num_items: number of items
        :param num_factors: number of predictive factors
        :param dropout: dropout rate
        :param num_layers: number of layers in MLP model
        :param model: 'NCF'
        :param GMF: pretrained GMF model
        :param MLP: pretrained MLP model
    """
    def __init__(self, num_users, num_items, num_factors=40, dropout=0.02, num_layers=10, model=None, GMF=None, MLP=None):
        super(NCF, self).__init__()

        self.num_factors = num_factors
        self.model = model
        self.GMF = GMF
        self.MLP = MLP

        self.dropout = dropout
        self.num_layers = num_layers
        
        
        self.user_embedding_GMF = nn.Embedding(num_users, num_factors)
        self.item_embedding_GMF = nn.Embedding(num_items, num_factors)
        self.user_embedding_MLP = nn.Embedding(num_users, num_factors*(2**(num_layers-1)))
        self.item_embedding_MLP = nn.Embedding(num_items, num_factors*(2**(num_layers-1)))
        

        MLP_modules = []

        for i in range(num_layers):
            input_size = num_factors * 2**(num_layers-i)
            MLP_modules.append(nn.Dropout(dropout))
            MLP_modules.append(nn.Linear(input_size, input_size//2))
            MLP_modules.append(nn.ReLU())
            MLP_modules.append(nn.BatchNorm1d(input_size//2))
        self.MLP_layers = nn.Sequential(*MLP_modules)

        predict_size = num_factors * 2

        self.predict_layer = nn.Linear(predict_size, 1)

        self._init_weight_()
        self.ReLU = nn.ReLU()
        self.Sigmoid = nn.Sigmoid()

    def _init_weight_(self):
        nn.init.normal_(self.user_embedding_GMF.weight, std=0.01)
        nn.init.normal_(self.item_embedding_GMF.weight, std=0.01)
        nn.init.normal_(self.user_embedding_MLP.weight, std=0.01)
        nn.init.normal_(self.item_embedding_MLP.weight, std=0.01)

        for layer in self.MLP_layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
        nn.init.kaiming_uniform_(self.predict_layer.weight, a=1, nonlinearity='sigmoid')

        for layer in self.modules():
            if isinstance(layer, nn.Linear) and layer.bias is not None:
                layer.bias.data.zero_()
        

    def forward(self, user, item):
        embedding_user_GMF = self.user_embedding_GMF(user)
        embedding_item_GMF = self.item_embedding_GMF(item)

        vector_GMF = torch.mul(embedding_user_GMF, embedding_item_GMF)

        embedding_user_MLP = self.user_embedding_MLP(user)
        embedding_item_MLP = self.item_embedding_MLP(item)

        interaction = torch.cat([embedding_user_MLP, embedding_item_MLP], -1)
        vector_MLP = self.MLP_layers(interaction)
        
        # 각 모델 결과에 가중치를 곱해준다.
        vector_GMF = vector_GMF * 0.5
        vector_MLP = vector_MLP * 0.5
        
        vector_concat = torch.cat([vector_GMF, vector_MLP], -1)

        prediction = self.predict_layer(vector_concat)

        return prediction.view(-1)
        
    
    def predict(self, user, item):
        return self.forward(user, item)
    
    def get_loss(self, user, item, rating):
        prediction = self.forward(user, item)
        loss = nn.MSELoss()(prediction, rating)
        return loss
    
    def get_topk(self, user, topk):
        return torch.topk(self.forward(user, torch.arange(self.num_items).cuda()), topk)[1]
    
    def get_user_embedding(self, user):
        return self.user_embedding(user)
    
    def get_item_embedding(self, item):
        return self.item_embedding(item)
    
    def get_user_item_embedding(self, user, item):
        return self.user_embedding(user), self.item_embedding(item)
    

if __name__ == '__main__':
    import os
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from customDataset import MovieDataset


    dataPath = os.path.join(os.path.abspath(os.path.curdir),'Codes', 'Model', 'ml-latest-small')

    movie_df = pd.read_csv(os.path.join(dataPath,'movies.csv'))
    rating_df = pd.read_csv(os.path.join(dataPath,'ratings.csv'))

    movieIdMapper = {}
    for new, old in enumerate(rating_df['movieId'].unique()):
        movieIdMapper[old] = new+1

    rating_df['movieId'] = rating_df['movieId'].apply(lambda x: movieIdMapper[x])


    train_df, test_df = train_test_split(rating_df, test_size=0.1, random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42)

    train = MovieDataset(train_df)
    validation = MovieDataset(val_df, is_training=False)
    test = MovieDataset(test_df, is_training=False)

    train_loader = train.get_loader(128)
    val_loader = validation.get_loader(128)
    test_loader = test.get_loader(128)

    #########################CREATE MODEL#########################
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    GMF = None
    MLP = None

    num_users = len(rating_df["userId"].unique())+1
    num_items = max(rating_df["movieId"])+1

    print("num_users: ", num_users)
    print("num_items: ", num_items)

    model = MovieNCF(num_users, num_items, 16, 0.4, 3, "test_model", GMF, MLP)
    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    for epoch in range(100):
        model.train()
        for batch in train_loader:
            user, item, rating = batch
            user = user.to(device)
            item = item.to(device)
            rating = rating.to(device)

            optimizer.zero_grad()
            prediction = model(user, item)
            loss = criterion(prediction, rating)
            loss.backward()
            optimizer.step()
        scheduler.step()
        print("epoch: ", epoch, "loss: ", loss.item())

