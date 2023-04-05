import os
import pandas as pd
import json

def do_mapping(profiles, reviews, save=False):
    # 파일 로드
    cur_path = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(cur_path, "..\..\Dataset")
    profiles_data_path = os.path.join(data_dir, "{}".format(profiles))
    reviews_data_path = os.path.join(data_dir, "{}".format(reviews))
    
    with open(profiles_data_path, 'r', encoding='utf-8') as f:
        profiles_dataset = json.load(f)

    with open(reviews_data_path, 'r', encoding='utf-8') as f:
        reviews_dataset = json.load(f)

    cocktail_dataset = profiles_dataset['cocktail_profiles']

    # Dataframe 변환
    df_profiles = pd.DataFrame(cocktail_dataset)
    df_reviews = pd.DataFrame(reviews_dataset)

    # Cocktail ID Mapping
    cocktail_profiles = df_profiles[['ID', 'Name']]
    result = pd.concat([df_reviews['Cocktail'].map(cocktail_profiles.set_index('ID')['Name']), df_reviews], axis=1)
    result.columns = ['Cocktail', 'ID', 'Review', 'Rating', 'Timestamp']

    # Json 파일로 저장
    if save:
        result.to_json(os.path.join(data_dir, 'result.json'), orient='records', force_ascii=False, indent=4)
        

    return result







