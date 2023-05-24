# app/controller.py
from typing import Dict, List
from service import UserService, CocktailService, RatingService, RecommenderService
from model import User, Cocktail, Rating

class UserController:
    def __init__(self):
        self.service = UserService()

    def create_user(self, user: User):
        existing_user = self.service.get_user(user.userId)
        if existing_user:
            return existing_user
        new_user = User(userId=user.userId, userMappingId=user.userMappingId)
        return self.service.create_user(new_user)

    def get_users(self):
        return self.service.get_users()

    def get_user(self, user_id: int):
        return self.service.get_user(user_id)

class CocktailController:
    def __init__(self):
        self.service = CocktailService()

    def create_cocktail(self, cocktail: Cocktail):
        existing_cocktail = self.service.get_cocktail(cocktail.cocktailId)
        if existing_cocktail:
            return existing_cocktail
        new_cocktail = Cocktail(cocktailId=cocktail.cocktailId, cocktailName=cocktail.cocktailName)
        return self.service.create_cocktail(new_cocktail)
    
    def get_cocktails(self):
        return self.service.get_cocktails()
    
    def get_cocktail(self, cocktail_id: int):
        return self.service.get_cocktail(cocktail_id)
    
class RatingController:
    def __init__(self):
        self.service = RatingService()

    def create_rating(self, rating: Rating):
        return self.service.create_rating(rating)

    def get_ratings(self):
        return self.service.get_ratings()
    
    def save_ratings(self, rating: List[Rating]):
        return self.service.save_ratings(rating)
    
class RecommenderController:
    def __init__(self):
        self.recommender_service = RecommenderService()
        self.user_service = UserService()
        self.cocktail_service = CocktailService()
        self.rating_service = RatingService()

    def check_dataset(self, ratings: Dict[str, Rating]):
        # 데이터 저장
        self.rating_service.save_ratings(ratings)

        # 데이터셋 학습
        self.recommender_service.fit(ratings)


        for rating in ratings.ratings:
            user = self.user_service.get_user(rating.userId)
            cocktail = self.cocktail_service.get_cocktail(rating.cocktailId)
            if not user:
                return {"message": "존재하지 않는 유저입니다."}
            if not cocktail:
                return {"message": "존재하지 않는 칵테일입니다."}
        return self.service.check_dataset(ratings)

