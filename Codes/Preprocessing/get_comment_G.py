import os
import json

cur_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(cur_path, "..\..\Dataset")

#dataset_name = 'gutekueche_cocktail_reviews_4.json'
dataset_name = 'all_recipe_cocktail_reviews.json'
dataset_path = os.path.join(data_path, dataset_name)

with open(dataset_path, 'r', encoding='utf-8') as f:
    dataset = json.load(f)

cocktail_reviews = dataset['cocktail_reviews']

max_len = 0
for content in cocktail_reviews:
    #print(content['Review'])
    max_len = max(max_len, len(content['Review']))
    #print(content['Timestamp'])

print(max_len)