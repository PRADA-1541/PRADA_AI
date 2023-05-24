from typing import List
from pydantic import BaseModel

class User(BaseModel):
    userId: str
    mappingId: int

class Cocktail(BaseModel):
    cocktailName: str
    mappingId: int

class Rating(BaseModel):
    userId: int
    cocktailId: int
    rating: float
    timestamp: int