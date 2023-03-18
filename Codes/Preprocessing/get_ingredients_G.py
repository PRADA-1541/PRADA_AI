import os
import json

cur_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(cur_path, "..\..\Dataset")
dataset_name = 'gutekueche_cocktail_profiles_filtered_by_2.json'
dataset_path = os.path.join(data_path, dataset_name)

with open(dataset_path, 'r', encoding='utf-8') as f:
    dataset = json.load(f)

cocktail_ingredients = []
#num_ing_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, "½", "⅓", "⅔", "¼", "¾", "⅕", "⅖", "⅗", "⅘", "⅙", "⅚", "⅐", "⅛", "⅜", "⅝", "⅞"]

for cocktail in dataset:
    ingredients_list = cocktail['Ingredients']
    for inst in ingredients_list:
        #대문자로 바꾸기
        pure_ingredient = inst.upper()
        # 앞뒤 공백 제거
        pure_ingredient = pure_ingredient.strip(" ")

        pure_ingredient = pure_ingredient.split(" ")[2:]
        pure_ingredient = " ".join(pure_ingredient)
        cocktail_ingredients.append(pure_ingredient)

cocktail_ingredients = list(set(cocktail_ingredients))
cocktail_ingredients.sort()

print(cocktail_ingredients)
print(len(cocktail_ingredients))

# 
with open('gutekueche_cocktail_ingredients.json', 'w', encoding='utf-8') as f:
    json.dump(cocktail_ingredients, f, ensure_ascii=False, indent=4)

#cocktail_ingredients = list(set(cocktail_ingredients))

