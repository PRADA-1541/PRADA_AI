import pandas as pd
import os

path = os.path.join(os.path.curdir, 'Dataset')

cock_df = pd.read_csv(os.path.join(path, "all_drinks.csv"))
cock_df = cock_df.loc[:, ['strDrink', 'idDrink', 'strAlcoholic', 'strCategory', 'strGlass', 'strIBA']]
recipe_df = pd.read_csv(os.path.join(path, "cocktail_recipes.csv"))

print(cock_df.columns)
print(recipe_df.columns)

df = pd.merge(left=cock_df, right=recipe_df, how="inner", left_on="strDrink", right_on="name")

print(df)