import os
import sys
import json
from translate_api import send_papago_api

def translate_Reviews_and_Timestamp(review):
    origin_review = review['Review']
    origin_timestamp = review['Timestamp']

    # Translate Review
    translated_review = send_papago_api(origin_review)
    translated_timestamp = send_papago_api(origin_timestamp)

    processed_review = review
    processed_review['Review'] = translated_review
    processed_review['Timestamp'] = translated_timestamp

    return processed_review


cur_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(cur_path, "..\..\Dataset")
#print(cur_path)
#print(os.path.dirname(os.path.abspath(__file__)))

dataset_name = 'gutekueche_cocktail_reviews_0.json'
dataset_path = os.path.join(data_path, dataset_name)

with open(dataset_path, 'r', encoding='utf-8') as f:
    dataset = json.load(f)

cocktail_reviews = dataset['cocktail_reviews']

cocktail_reviews = [translate_Reviews_and_Timestamp(review) 
                    for review in cocktail_reviews]

dataset['cocktail_reviews'] = cocktail_reviews

print(dataset)


#dataset = json.dumps(dataset, ensure_ascii=False, indent=4)
#print(dataset)

