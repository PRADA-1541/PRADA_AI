import os
import json

cur_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(cur_path, "..\..\Dataset")
dataset_name = 'gutekueche_cocktail_ingredients_eng.json'
dataset_path = os.path.join(data_path, dataset_name)

with open(dataset_path, 'r', encoding='utf-8') as f:
    
    g_dataset = json.load(f)

cocktail_ingredients = []

cocktail_ingredients.extend(g_dataset)

dataset_name = 'all_recipe_cocktail_ingredients.json'
dataset_path = os.path.join(data_path, dataset_name)

with open(dataset_path, 'r', encoding='utf-8') as f:
    ar_dataset = json.load(f)

cocktail_ingredients.extend(ar_dataset)

cocktail_ingredients = list(set(cocktail_ingredients))
cocktail_ingredients.sort()

print(cocktail_ingredients)
print(len(cocktail_ingredients))

# 
with open('cocktail_ingredients.json', 'w', encoding='utf-8') as f:
    json.dump(cocktail_ingredients, f, ensure_ascii=False, indent=4)

#cocktail_ingredients = list(set(cocktail_ingredients))

