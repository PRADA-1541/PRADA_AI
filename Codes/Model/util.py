import pandas as pd
import numpy as np
import os
from similarity import cosim, pearson, jacsim

def getRating(fname):
    df = pd.read_csv(fname)
    data = df.to_numpy()
    X = data[:, :-1]
    Y = data[:, -1]
    # center the X
    X -= X.mean(axis=0)
    # normalize the X
    X /= X.std(axis=0)
    return X, Y

def getDataframe(fname):
    """
    fname: file name

    fname에 해당하는 파일(.csv, .json)을 dataframe으로 변환
    """
    fname = str(fname)
    if '.csv' in fname[-4:]:
        df = pd.read_csv(fname)
    elif '.json' in fname[-5:]:
        df = pd.read_json(fname)
    else:
        df = None
        raise Exception('잘못된 파일 형식입니다.')
        
    return df

def getCocktailRatio(dataframe, colName):
    """
    dataframe: cocktail list dataframe
    colName: column으로 변환할 attribute의 이름
    
    dataframe의 list attribute를 column으로 변환
    """

    # colName에 해당하는 attribute를 list로 변환
    list_column = dataframe[colName].to_list()

    # column 정의
    columns = []
    
    # 각 ingredient를 column으로 변환
    for item in list_column:
        for ingredient in item:
            name = ingredient.split(' ')[2:]
            name = ' '.join(name)
            name.strip()

            columns.append(name)
    
    # 중복 제거
    columns = list(set(columns))
    columns.sort()

    # dataframe에 column 추가
    for column in columns:
        dataframe[column] = 0

    # 각 ingredient의 개수를 column에 추가
    for i in range(len(dataframe)):
        volume = dataframe['Volume'][i]
        for ingredient in dataframe[colName][i]:
            quantity, unit,name = ingredient.split(' ')[0], ingredient.split(' ')[1], ingredient.split(' ')[2:]
            quantity = quantity.strip()
            unit = unit.strip()
            name = ' '.join(name)
            name.strip()

            if unit == 'ml':
                dataframe[name][i] = float(quantity) / volume  # 액체류일경우 volume으로 나눠줌
            else:
                dataframe[name][i] = 1

    dataframe = dataframe[['ID', 'Name']+columns]
    return dataframe

def findByName(dataframe, name):
    """
    dataframe: cocktail list dataframe
    name: cocktail의 이름

    name에 해당하는 cocktail의 ID를 반환
    """
    return dataframe[dataframe['Name'] == name]

def findById(dataframe, ID):
    """
    dataframe: cocktail list dataframe
    ID: cocktail의 ID

    ID에 해당하는 cocktail의 Name을 반환
    """
    return dataframe[dataframe['ID'] == ID]

def getSimilarity(dataframe, cocktail1, cocktail2, byName=False, method='cosine'):
    """
    dataframe: cocktail list dataframe
    cocktail1: cocktail1의 ID 또는 Name
    cocktail2: cocktail2의 ID 또는 Name

    cocktail1과 cocktail2의 유사도를 계산
    """

    if byName:
        cocktail1 = findByName(dataframe, cocktail1)
        cocktail2 = findByName(dataframe, cocktail2)
        
    else:
        cocktail1 = findById(dataframe, cocktail1)
        cocktail2 = findById(dataframe, cocktail2)

    cocktail1 = cocktail1.drop(['ID', 'Name'], axis=1).to_numpy()[0]
    cocktail2 = cocktail2.drop(['ID', 'Name'], axis=1).to_numpy()[0]
    
    if method == 'cosine':
        similarity = cosim(cocktail1, cocktail2)
    elif method == 'pearson':
        similarity = pearson(cocktail1, cocktail2)
    elif method == 'jaccard':
        similarity = jacsim(cocktail1, cocktail2)
    else:
        raise Exception('다음 Method만 지원: cosine, pearson, jaccard')


    return similarity

def getSimilarities(dataframe, history, target, byName=False, method='cosine', topk=5):
    """
    dataframe: cocktail list dataframe
    history: 사용자의 history  / [1, 3, 4, 5]
    target: 추천된 cocktail의 ID 또는 Name  / 1
    topk: similarity가 높은 topk개의 cocktail을 반환

    history의 칵테일들과 cocktail의 유사도를 계산
    """
    if type(history) != list:
        raise Exception('history는 list 또는 tuple이어야 합니다.')

    # 사용자 history가 topk보다 작을 경우 topk를 history의 길이로 설정
    if len(history) < topk:
        topk = len(history)

    if method == 'cosine':
        sim = cosim
    elif method == 'pearson':
        sim = pearson
    elif method == 'jaccard':
        sim = jacsim
    else:
        raise Exception('다음 Method만 지원: cosine, pearson, jaccard')
    

    # history cocktail의 ID를 추출]
    if byName:
        history = [findByName(dataframe, int(cocktail)) for cocktail in history]
        target = findByName(dataframe, target)
    else:
        history = [findById(dataframe, int(cocktail)) for cocktail in history]
        target = findById(dataframe, target)

    target = target.drop(['ID', 'Name'], axis=1).to_numpy()[0]

    
    # history cocktail의 유사도를 계산
    sim_dict = {}

    for cocktail in history:
        c = cocktail.drop(['ID', 'Name'], axis=1).to_numpy()[0]
        similarity = sim(c, target)
        sim_dict[cocktail['ID'].values[0]] = similarity

    # 유사도가 높은 topk개의 cocktail을 반환
    sim_dict = sorted(sim_dict.items(), key=lambda x: x[1], reverse=True)
    sim_dict = sim_dict[:topk]

    return sim_dict

if __name__ == '__main__':
    # fname 정의
    cur_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(cur_path, '..', '..', 'Dataset')
    fname = os.path.join(data_path, 'PRADA_cocktail_list.json')

    # DataFrame 생성
    df = getDataframe(fname)

    # Cocktail Content Profile 생성
    cocktail_content_profile = getCocktailRatio(df, 'Ingredients')
    print(cocktail_content_profile.head())

    # Cocktail Similarity 계산
    cosine_sim = getSimilarity(cocktail_content_profile, 'Angel Blue-d', 'White Russian', byName=True, method='cosine')
    pearson_sim = getSimilarity(cocktail_content_profile, 'Angel Blue-d', 'White Russian', byName=True, method='pearson')
    jaccard_sim = getSimilarity(cocktail_content_profile, 'Angel Blue-d', 'White Russian', byName=True, method='jaccard')

    print(cosine_sim, pearson_sim, jaccard_sim)

    cosine_sim = getSimilarity(cocktail_content_profile, 0, 91, method='cosine')
    pearson_sim = getSimilarity(cocktail_content_profile, 0, 91, method='pearson')
    jaccard_sim = getSimilarity(cocktail_content_profile, 0, 91, method='jaccard')

    print(cosine_sim, pearson_sim, jaccard_sim)

    # 출력 포맷
    cocktail1, cocktail2 = 91, 0
    sim = getSimilarity(cocktail_content_profile, cocktail1, cocktail2, method='cosine')

    print(f"similarity of two items: \
          {cocktail_content_profile[cocktail_content_profile['ID'] == cocktail1]['Name'].to_list()[0]}\
           & \
          {cocktail_content_profile[cocktail_content_profile['ID'] == cocktail2]['Name'].to_list()[0]}")
    
    print(f"Similarity = {sim}")
    print("=====================================")

    # history와 target을 입력받아 유사도가 높은 topk개의 cocktail을 반환
    history = [0, 87, 85, 82, 3]
    target = 91

    similarities = getSimilarities(cocktail_content_profile, history, target, method='cosine', topk=4)
    print(f"{cocktail_content_profile[cocktail_content_profile['ID'] == target]['Name'].to_list()[0]} Coctail이 추천된 이유 \n")

    for cocktail, similarity in similarities:
        print(f"ID: {cocktail}, \
              Name: {cocktail_content_profile[cocktail_content_profile['ID'] == cocktail]['Name'].to_list()[0]}, \
              Similarity: {similarity}")


    


