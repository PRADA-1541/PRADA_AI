from flask import Flask, request, jsonify
import logging
import threading
import queue
import json
from service import UserService, CocktailService, RatingService, RecommenderService

# feature 정의
user_id_feature = "userId"
cocktail_id_feature = "cocktailId"
rating_feature = "rating"
timestamp_feature = "timestamp"

# 서비스 객체 생성
user_service = UserService()
cocktail_service = CocktailService()
rating_service = RatingService()
recommender_service = RecommenderService(user_service.get_num_users(), cocktail_service.get_num_cocktails())

app = Flask(__name__)


# 모델 객체 생성
#model = RecommenderModel()

# 모델 학습 함수
def train_model(args, result_queue):
    # 모델 학습 로직 구현
    user_id_list = set()
    cocktail_id_list = set()

    for rating in args:
        user_id_list.add(str(rating[user_id_feature])) # 학습 데이터에서 유저 아이디 추출
        cocktail_id_list.add(str(rating[cocktail_id_feature])) # 학습 데이터에서 칵테일 이름 추출


    # 새로운 유저와 칵테일이 추가되었을 경우 서비스에 추가
    user_service.add_users(list(user_id_list))
    cocktail_service.add_cocktails(list(cocktail_id_list))

    # 학습 데이터를 서비스에 추가
    ratings = [rating for rating in args if user_in_service(rating) and cocktail_in_service(rating)]
    
    rating_service.add_ratings(ratings)

    # 데이터 로더 생성
    user_id_mapper = user_service.get_user_dict()
    cocktail_id_mapper = cocktail_service.get_cocktail_dict()

    train_loader, val_loader, test_loader = rating_service.get_data_loaders(user_id_mapper, cocktail_id_mapper)

    # 모델 로드
    recommender_service.load_model(max(user_id_mapper.values())+1, max(cocktail_id_mapper.values())+1, None)

    #print("Max", max(user_id_mapper.values())+1, max(cocktail_id_mapper.values())+1)

    #print(cocktail_id_mapper)

    # 모델 학습
    recommender_service.fit(train_loader, val_loader, test_loader)

    predictions = recommender_service.predict_all(user_id_mapper, cocktail_id_mapper)

    #print("결과: ", predictions)

    #result_queue.put(result)
    result_queue.extend(predictions)
    #print("결과 큐에 넣음", result_queue )

def cocktail_in_service(rating):
    return str(rating[cocktail_id_feature]) in cocktail_service.get_cocktail_id_list()

def user_in_service(rating):
    return str(rating[user_id_feature]) in user_service.get_user_id_list()
    

# 모델 추론 함수
def predict(result_queue):
    # id_mapper 생성
    user_id_mapper = user_service.get_user_dict()
    cocktail_id_mapper = cocktail_service.get_cocktail_dict()

    # 예측
    predictions = recommender_service.predict_all(user_id_mapper, cocktail_id_mapper)

    #result_queue.put(result)
    result_queue.extend(predictions)

def create_success_message():
    response = create_respond()
    response["code"] = 0,
    response["message"] = "success"
    return response

def create_respond():
    return { "code": None, "message": None }

# /model/train 엔드포인트
@app.route('/model/train', methods=['POST'])
def train_endpoint():
    print("학습 요청 수신")

    args = request.get_json()
    result_queue = list()


    # 학습을 다른 스레드에서 실행
    train_model(args, result_queue)

    response = create_success_message()
    response["result"] = result_queue
    #train_thread = threading.Thread(target=train_model, args=(args, result_queue))
    #train_thread.start()

    #train_thread.join()

    # if train_thread.is_alive():
    #     print("학습 요청 처리 중...")
    #     response = create_success_message()
    # else:
    #     print("학습 요청 처리 완료")
    #     response = create_success_message()
    #     response["result"] = result_queue

    return jsonify(response)
    

# /model/predict 엔드포인트
@app.route('/model/predict', methods=['POST'])
def predict_endpoint():
    result_queue = list()


    # 학습을 다른 스레드에서 실행
    predict(result_queue)

    response = create_success_message()
    response["result"] = result_queue

    return jsonify(response)

if __name__ == '__main__':
    app.run(port=5000)


