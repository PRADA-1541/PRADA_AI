# app/service.py

import os, sys
from collections import OrderedDict
import pickle
import logging
from typing import List
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from Recommender.ncfModel import NCF
from Recommender.customDataset import Dataset
from model import User, Cocktail, Rating

model_name = 'model'

class UserService:
    def __init__(self):
        self.path = os.path.join(os.getcwd(), 'model', 'user_dict.pkl')
        self.users = []
        self.not_assigned_indices = []

        logging.info("User Service 초기화")
        user_dict = {}
        # pickle 파일이 존재하지 않으면 생성
        if not os.path.exists(self.path):
            self.create_user_dict()
        else:
            print("pickle 파일이 존재합니다.")
            with open(self.path, 'rb') as f:
                user_dict = pickle.load(f)
                # pickle 파일이 빈 dict라면
        self.last_mapping_id = 0 if user_dict == {} else max(user_dict.values())
        
        for user_id, mapping_id in user_dict.items():
            self.users.append(User(userId= user_id, mappingId = mapping_id))

        # 할당되지 않은 mapping id를 저장
        self.not_assigned_indices = [id for id in range(self.last_mapping_id) if id not in user_dict.values()]

    def get_all_users(self):
        """
        현재 메모리에 업로드된 모든 유저 정보를 반환
        """
        return self.users
        
    def create_user_dict(self):
        """
        pickle 파일이 존재하지 않을 때, 빈 dict를 pickle 파일로 저장
        """
        # pickle 파일로 저장
        with open(self.path, 'wb') as f:
            pickle.dump({}, f)


    def get_user_id_list(self):
        """
        현재 메모리에 업로드된 모든 유저의 user id만 리스트로 반환
        """
        return [user.userId for user in self.users]
    
    def get_user_dict(self):
        """
        현재 메모리에 업로드된 모든 유저의 user id와 mapping id를 dict로 반환
        """
        user_dict = {}
        for user in self.users:
            user_dict[user.userId] = user.mappingId
        return user_dict
    
    def get_num_users(self):
        """
        현재 메모리에 업로드된 유저의 수를 반환
        """
        return len(self.users)

    def add_users(self, users: set):
        """
        새로운 유저를 추가
        """
        logging.info("서비스 유저 추가 : " + str(users) + "")
        user_id_list = [user.userId for user in self.users]
        new_users = []

        for user in users:
            if user not in user_id_list:
                new_mapping_id = self.assign_new_mapping_id()
                new_users.append(User(userId = user, mappingId = new_mapping_id))

        self.save_new_users(new_users)

    def delete_users(self, users: set):
        """
        기존 유저를 삭제
        """
        logging.info("서비스 유저 삭제 : " + str(users) + "")
        user_id_list = [user.userId for user in self.users]
        deleted_users = []

        for user in users:
            if user.userId in user_id_list:
                deleted_users.append(user)
                self.not_assigned_indices.append(user.mappingId)

    def save_deleted_users(self, deleted_users: List[User]):
        """
        삭제된 유저를 pickle에 반영
        """
        # pickle 파일 로드 후 deleted_users 제거
        with open(self.path, 'rb') as f:
            user_dict = pickle.load(f)
            for user in deleted_users:
                del user_dict[user.userId]

        # pickle 파일로 저장
        with open(self.path, 'wb') as f:
            pickle.dump(user_dict, f)

        self.last_mapping_id = max(user_dict.values())
        self.users.remove(user)


    def save_new_users(self, new_users: List[User]):
        """
        새로운 유저를 pickle에 반영
        """
        # pickle 파일 로드 후 new_users 추가
        with open(self.path, 'rb') as f:
            user_dict = pickle.load(f)
            for user in new_users:
                user_dict[user.userId] = user.mappingId

        # pickle 파일로 저장
        with open(self.path, 'wb') as f:
            pickle.dump(user_dict, f)

        self.last_mapping_id = max(user_dict.values())
        self.users.extend(new_users)


    def assign_new_mapping_id(self):
        """
        새로운 유저에게 mapping id를 할당
        """
        if len(self.not_assigned_indices) > 0:
            return self.not_assigned_indices.pop(0)
        else:
            self.last_mapping_id += 1
            return self.last_mapping_id


    def clear_all(self):
        """
        모든 유저 정보를 삭제
        """
        self.users = []
        self.not_assigned_indices = []
        self.create_user_dict()
        self.last_mapping_id = 0


class CocktailService:
    def __init__(self):
        self.path = os.path.join(os.getcwd(), 'model', 'cocktail_dict.pkl')
        self.cocktails = []
        self.not_assigned_indices = []

        logging.info("Cocktail Service 초기화")
        cocktail_dict = {}
        # pickle 파일이 존재하지 않으면 생성
        if not os.path.exists(self.path):
            self.create_cocktail_dict()
        else:
            print("pickle 파일이 존재합니다.")
            with open(self.path, 'rb') as f:
                cocktail_dict = pickle.load(f)
                # pickle 파일이 빈 dict라면
        self.last_mapping_id = 0 if cocktail_dict == {} else max(cocktail_dict.values())
        
        for cocktail_id, mapping_id in cocktail_dict.items():
            self.cocktails.append(Cocktail(cocktailName= cocktail_id, mappingId = mapping_id))

        # 할당되지 않은 mapping id를 저장
        self.not_assigned_indices = [id for id in range(self.last_mapping_id) if id not in cocktail_dict.values()]

    def get_all_cocktails(self):
        """
        현재 메모리에 업로드된 모든 칵테일 정보를 반환
        """
        return self.cocktails
        
    def create_cocktail_dict(self):
        """
        pickle 파일이 존재하지 않을 때, 빈 dict를 pickle 파일로 저장
        """
        # pickle 파일로 저장
        with open(self.path, 'wb') as f:
            pickle.dump({}, f)


    def get_cocktail_id_list(self):
        """
        현재 메모리에 업로드된 모든 칵테일의 cocktail id만 리스트로 반환
        """
        return [cocktail.cocktailName for cocktail in self.cocktails]
    
    def get_cocktail_dict(self):
        """
        현재 메모리에 업로드된 모든 칵테일의 cocktail id와 mapping id를 dict로 반환
        """
        cocktail_dict = {}
        for cocktail in self.cocktails:
            cocktail_dict[cocktail.cocktailName] = cocktail.mappingId
        return cocktail_dict
    
    def get_num_cocktails(self):
        """
        현재 메모리에 업로드된 칵테일의 수를 반환
        """
        return len(self.cocktails)

    def add_cocktails(self, cocktails: set):
        """
        새로운 칵테일을 추가
        """
        logging.info("서비스 유저 추가 : " + str(cocktails) + "")
        cocktail_id_list = [cocktail.cocktailName for cocktail in self.cocktails]
        new_cocktails = []

        for cocktail in cocktails:
            if cocktail not in cocktail_id_list:
                new_mapping_id = self.assign_new_mapping_id()
                new_cocktails.append(Cocktail(cocktailName = cocktail, mappingId = new_mapping_id))

        self.save_new_cocktails(new_cocktails)

    def delete_cocktails(self, cocktails: set):
        """
        기존 칵테일을 삭제
        """
        logging.info("서비스 칵테일 삭제 : " + str(cocktails) + "")
        cocktail_id_list = [cocktail.cocktailId for cocktail in self.cocktails]
        deleted_cocktails = []

        for cocktail in cocktails:
            if cocktail.cocktailId in cocktail_id_list:
                deleted_cocktails.append(cocktail)
                self.not_assigned_indices.append(cocktail.mappingId)

    def save_deleted_cocktails(self, deleted_cocktails: List[Cocktail]):
        """
        삭제된 칵테일을 pickle에 반영
        """
        # pickle 파일 로드 후 deleted_users 제거
        with open(self.path, 'rb') as f:
            cocktail_dict = pickle.load(f)
            for cocktail in deleted_cocktails:
                del cocktail_dict[cocktail.cocktailId]

        # pickle 파일로 저장
        with open(self.path, 'wb') as f:
            pickle.dump(cocktail_dict, f)

        self.last_mapping_id = max(cocktail_dict.values())
        self.cocktails.remove(deleted_cocktails)


    def save_new_cocktails(self, new_cocktails: List[Cocktail]):
        """
        새로운 칵테일을 pickle에 반영
        """
        # pickle 파일 로드 후 new_users 추가
        with open(self.path, 'rb') as f:
            cocktail_dict = pickle.load(f)
            for cocktail in new_cocktails:
                cocktail_dict[cocktail.cocktailName] = cocktail.mappingId

        # pickle 파일로 저장
        with open(self.path, 'wb') as f:
            pickle.dump(cocktail_dict, f)

        self.last_mapping_id = max(cocktail_dict.values())
        self.cocktails.extend(new_cocktails)


    def assign_new_mapping_id(self):
        """
        새로운 칵테일에게 mapping id를 할당
        """
        if len(self.not_assigned_indices) > 0:
            return self.not_assigned_indices.pop(0)
        else:
            self.last_mapping_id += 1
            return self.last_mapping_id


    def clear_all(self):
        """
        모든 칵테일 정보를 삭제
        """
        self.cocktails = []
        self.not_assigned_indices = []
        self.create_cocktail_dict()
        self.last_mapping_id = 0

    
class RatingService:
    def __init__(self):
        self.ratings = []
        self.data_path = os.path.join(os.getcwd(), 'data', 'ratings')
        self.train_path = os.path.join(self.data_path, 'train.csv')
        self.test_path = os.path.join(self.data_path, 'test.csv')

        print("Rating Service 초기화")

        # data_path에 파일이 존재하지 않으면 빈 파일 생성
        if not os.path.exists(self.train_path):
            self.create_rating_csv(self.train_path)

        if not os.path.exists(self.test_path):
            self.create_rating_csv(self.test_path)

        self.train = pd.read_csv(self.train_path)
        self.test = pd.read_csv(self.test_path)

        

    def create_rating_csv(self, path):
        # csv 파일로 생성
        pd.DataFrame(columns=['userId', 'cocktailId', 'rating']).to_csv(path, index=False)

    def add_ratings(self, ratings: List):
        """
        새로운 평점을 추가
        """
        logging.info("서비스 평점 추가 : " + str(ratings) + "")
        rating_df = pd.DataFrame(ratings).drop(columns=['timestamp'])

        # train, test split
        new_train, new_test = train_test_split(rating_df, test_size=0.1)

        # train, test csv 파일에 추가
        train = pd.read_csv(self.train_path)
        test = pd.read_csv(self.test_path)

        train = pd.concat([train, new_train])
        test = pd.concat([test, new_test])

        train.to_csv(self.train_path, index=False)
        test.to_csv(self.test_path, index=False)

    def get_train_test_ratings_df(self):
        return pd.read_csv(self.train_path), pd.read_csv(self.test_path)
        # 추후 데이터가 쌓이면 window를 이용하여 train, test의 비율을 조절할 수 있도록 수정

    def get_data_loaders(self, user_id_mapper, cocktail_id_mapper):
        """
        train, test 데이터를 DataLoader로 반환
        """
        train, test = self.get_train_test_ratings_df()
        print(":::", train, test)
        # 각 user, cocktail에 mapping id 부여
        train['userId'] = train['userId'].apply(lambda x: user_id_mapper[str(x)])
        train['cocktailId'] = train['cocktailId'].apply(lambda x: cocktail_id_mapper[str(x)])
        test['userId'] = test['userId'].apply(lambda x: user_id_mapper[str(x)])
        test['cocktailId'] = test['cocktailId'].apply(lambda x: cocktail_id_mapper[str(x)])
        print(":::", train, test)


        train, val = train_test_split(train, test_size=0.1)

        train_dataset = Dataset(train, is_training=True)
        val_dataset = Dataset(val, is_training=False)
        test_dataset = Dataset(test, is_training=False)
        
        train_loader = train_dataset.get_loader(128)
        val_loader = val_dataset.get_loader(128)
        test_loader = test_dataset.get_loader(128)

        return train_loader, val_loader, test_loader


class RecommenderService:
    def __init__ (self, num_users, num_items):
        self.root = os.path.join(os.getcwd(), 'Recommender')
        self.weight_path = os.path.join(self.root, 'weights')
        self.whole_weight_path = os.path.join(self.weight_path, model_name+'.pth')
        #self.sub_weight_path = os.path.join(self.weight_path, 'sub_weights.pth')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.load_model(NCF(num_users, num_items, 5, 0.4, 3), self.whole_weight_path)
        
    def load_model(self, model, weights):
        print("모델 로드")
        self.model = model
        checkpoint = torch.load(weights, map_location=self.device)
        sub_checkpoint = OrderedDict([(layer, checkpoint[layer]) for layer in checkpoint if '_embedding_' not in layer])
        self.model.load_state_dict(sub_checkpoint, strict=False)
        self.model.to(self.device)
            

    def fit(self, train_loader, val_loader, test_loader):
        print("학습 시작")
        self.model.eval()

        criterion = nn.MSELoss()
        optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.7) # Best
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=20, verbose=True)

        best_rmse = 1000
        for epoch in range(200):
            if epoch % 10 == 0:
                print("epoch : ", epoch)
            self.model.train() # enable dropout if used
            
            train_loss = 0

            for users, items, ratings in train_loader:
                users, items, ratings = users.to(self.device), items.to(self.device), ratings.to(self.device)
                
                self.model.zero_grad()
                prediction = self.model(users, items)
                loss = criterion(prediction, ratings)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                
            train_loss /= len(train_loader)

            val_loss = 0
            self.model.eval()
            for users, items, ratings in val_loader:
                with torch.no_grad():
                    users, items, ratings = users.to(self.device), items.to(self.device), ratings.to(self.device)
                    prediction = self.model(users, items)
                    val_loss += criterion(prediction, ratings).item()
                    
            val_loss /= len(val_loader)
            
            if val_loss < best_rmse:
                best_rmse = val_loss
                torch.save(self.model.state_dict(), model_name+".pth")

            scheduler.step(metrics=val_loss)

        print("Best RMSE: {:.4f}".format(best_rmse))

        #########################TESTING#########################
        self.model.load_state_dict(torch.load(model_name+".pth"))
        self.model.eval()
        with torch.no_grad():
            test_rmse = 0
            num_items = 0
            for users, items, ratings in test_loader:
                users, items, ratings = users.to(self.device), items.to(self.device), ratings.to(self.device)
                prediction = self.model(users, items)
                test_rmse += nn.MSELoss()(ratings.cpu(), prediction.cpu()).item()*len(ratings)
                num_items += len(ratings)
            test_rmse /= num_items
            test_rmse = np.sqrt(test_rmse)

        return self.recommends.extend(ratings)
    
    def predict(self, user_id, top_k):
        self.model.eval()
        with torch.no_grad():
            prediction = self.model(torch.tensor([user_id]).to(self.device), torch.arange(0, self.num_items).to(self.device))
            top_k = torch.topk(prediction, top_k).indices.cpu().numpy()
            print(top_k)
            return prediction.cpu().numpy()
