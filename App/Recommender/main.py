import os
import time
import argparse
import numpy as np
import pandas as pd
from surprise import Dataset
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
import torch.utils.data as data
import torch.backends.cudnn as cudnn
from torchsummary import summary
#from tensorboardX import SummaryWriter
import wandb

import ncfModel
import config
import evaluate
import util
from customDataset import MovieDataset

#########################PARAMETERS#########################

parser = argparse.ArgumentParser()
parser.add_argument("--lr", 
	type=float, 
	default=0.001, 
	help="learning rate")
parser.add_argument("--dropout", 
	type=float,
	default=0.0,  
	help="dropout rate")
parser.add_argument("--batch_size", 
	type=int, 
	default=256, 
	help="batch size for training")
parser.add_argument("--epochs", 
	type=int,
	default=20,  
	help="training epoches")
parser.add_argument("--top_k", 
	type=int, 
	default=10, 
	help="compute metrics@top_k")
parser.add_argument("--num_factors", 
	type=int,
	default=32, 
	help="predictive factors numbers in the model")
parser.add_argument("--num_layers", 
	type=int,
	default=3, 
	help="number of layers in MLP model")
parser.add_argument("--num_ng", 
	type=int,
	default=4, 
	help="sample negative items for training")
parser.add_argument("--test_num_ng", 
	type=int,
	default=99, 
	help="sample part of negative items for testing")
parser.add_argument("--out", 
	default=True,
	help="save model or not")
parser.add_argument("--gpu", 
	type=str,
	default="0",  
	help="gpu card ID")
parser.add_argument("--step_size",
    type=int,
    default=10,
    help="step size for learning rate decay")
parser.add_argument("--gamma",
    type=float,
    default=0.5,
    help="gamma for learning rate decay")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
cudnn.benchmark = True


#########################LOADING DATA#########################
# 현재 경로에 ml-latest-small 폴더가 있어야 함
dataPath = os.path.join(os.path.abspath(os.path.curdir),'Codes', 'Model', 'ml-latest-small')

movie_df = pd.read_csv(os.path.join(dataPath,'movies.csv'))
rating_df = pd.read_csv(os.path.join(dataPath,'movieLens_ratings2.csv'))
#tag_df = pd.read_csv(os.path.join(dataPath,'tags.csv'))

movieIdMapper = {}
for new, old in enumerate(rating_df['movieId'].unique()):
    movieIdMapper[old] = new+1

rating_df['movieId'] = rating_df['movieId'].apply(lambda x: movieIdMapper[x])


train_df, test_df = train_test_split(rating_df, test_size=0.05, random_state=42)
train_df, val_df = train_test_split(train_df, test_size=0.05, random_state=42)

train = MovieDataset(train_df)
validation = MovieDataset(val_df, is_training=False)
test = MovieDataset(test_df, is_training=False)

train_loader = train.get_loader(args.batch_size)
val_loader = validation.get_loader(args.batch_size)
test_loader = test.get_loader(args.batch_size)

#########################CREATE MODEL#########################
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

GMF = None
MLP = None

num_users = len(rating_df["userId"].unique())+1
num_items = max(rating_df["movieId"])+1

print("num_users: ", num_users)
print("num_items: ", num_items)

model = ncfModel.MovieNCF(num_users, num_items, args.num_factors, args.dropout, args.num_layers, config.model, GMF, MLP)
model = model.to(device)

criterion = nn.MSELoss()
#optimizer = optim.Adam(model.parameters(), lr=args.lr)
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.7) # Best
#optimizer = optim.RMSprop(model.parameters(), lr=args.lr, momentum=0.9)
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
#scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=0.0001, last_epoch=-1)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=20, verbose=True) #Best

print(model)
model_name = config.model+"_"+str(args.num_factors)+"_"+str(args.num_layers)+"_"+str(args.dropout)+"_"+str(args.lr)+"_"+str(args.batch_size)

#########################TRAINING#########################
wandb.init(
    project="MovieLens_Adapted2",
    name=
        "ncf"+
        "-factors"+str(args.num_factors)+
        "-layers"+str(args.num_layers)+
        "-dropout"+str(args.dropout)+
        "-lr"+str(args.lr)+
        "-batch"+str(args.batch_size),
    config=args
)

wandb.watch(model)
wandb.config.update(args)

best_rmse = 1000
for epoch in range(args.epochs):
    model.train() # enable dropout if used
    start = time.time()
    #train_loader.dataset.ng_sample()
    
    train_loss = 0

    for users, items, ratings in train_loader:
        users, items, ratings = users.cuda(), items.cuda(), ratings.cuda()
        
        model.zero_grad()
        prediction = model(users, items)
        loss = criterion(prediction, ratings)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        
    train_loss /= len(train_loader)
    if (epoch+1) % 5 == 0:
        elapsed = time.time() - start
        print("The time elapse of epoch {:03d}".format(epoch) + " is: " + 
			time.strftime("%H: %M: %S", time.gmtime(elapsed)))
        print("Epoch : {}, Train Loss : {:.4f}".format(epoch+1, train_loss))
    wandb.log({"train_loss": train_loss})

    val_loss = 0
    model.eval()
    for users, items, ratings in val_loader:
        with torch.no_grad():
            users, items, ratings = users.cuda(), items.cuda(), ratings.cuda()
            prediction = model(users, items)
            val_loss += criterion(prediction, ratings).item()
            
    val_loss /= len(val_loader)
    wandb.log({"val_loss": val_loss})
    print("Epoch : {}, Val Loss : {:.4f}".format(epoch+1, val_loss))
    if val_loss < best_rmse:
        best_rmse = val_loss
        if args.out:
            torch.save(model.state_dict(), model_name+".pth")
            print("Save model in file: {}.pth".format(model_name))

    scheduler.step(metrics=val_loss)
    #scheduler.step()
print("Best RMSE: {:.4f}".format(best_rmse))

#########################TESTING#########################
model.load_state_dict(torch.load(model_name+".pth"))
model.eval()
with torch.no_grad():
    test_loss = 0
    for users, items, ratings in test_loader:
        users, items, ratings = users.cuda(), items.cuda(), ratings.cuda()
        prediction = model(users, items)
        loss = criterion(prediction, ratings)
        
        test_loss += loss.item()
    test_loss /= len(test_loader)
    print("Test Loss : {:.4f}".format(test_loss))

    wandb.log({"test_loss": test_loss})

    test_rmse = 0
    num_items = 0
    for users, items, ratings in test_loader:
        users, items, ratings = users.cuda(), items.cuda(), ratings.cuda()
        prediction = model(users, items)
        test_rmse += nn.MSELoss()(ratings.cpu(), prediction.cpu()).item()*len(ratings)
        num_items += len(ratings)
    test_rmse /= num_items
    test_rmse = np.sqrt(test_rmse)
    print("Test RMSE : {:.4f}".format(test_rmse))