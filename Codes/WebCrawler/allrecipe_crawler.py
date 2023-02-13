import time

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common import exceptions
import os
import json

IMPLICIT_WAIT = 8
COLLECT_PROFILES = 1
COLLECT_REVIEWS = 2
file_name = 'All_Recipes_Links.txt'
file_path = os.path.join(os.path.curdir, file_name)
need_crawl = False

recipe_links = list()
item_id = 1
item_list = []
review_list = []

"""
mode: 칵테일 정보 = 1, 리뷰 및 평점 수집 = 2
"""
mode = COLLECT_REVIEWS
###
# 웹드라이버 옵션 설정 후 생성
def create_driver(headless=False):
    options = Options()
    options.headless = headless
    driver = webdriver.Chrome(ChromeDriverManager().install(), options=options)
    driver.implicitly_wait(IMPLICIT_WAIT)
    return driver


def is_valid_link(link):
    cond1 = link is not None
    cond2 = '#' not in str(link)
    return cond1 and cond2


def is_category_link(link):
    cond1 = 'cocktails/' in str(link)
    cond2 = is_valid_link(link)
    return cond1 and cond2


def is_recipe_link(link):
    cond1 = 'recipe/' in str(link)
    cond2 = is_valid_link(link)
    return cond1 and cond2


# create a new browser instance
driver = create_driver()

# 레시피 링크 리스트 파일이 없으면 링크 탐색
if os.path.isfile(file_path):
    with open(file_path, 'r', encoding='utf8') as file:
        for line in file:
            recipe_links.append(line[:-1])
    print(recipe_links)

else:
    # navigate to a website
    main_url = "https://www.allrecipes.com/recipes/133/drinks/cocktails/"
    driver.get(main_url)

    # find and interact with elements on the page
    a_links = driver.find_elements(By.TAG_NAME, 'a')

    a_links = list(map(lambda li: li.get_attribute('href'), a_links))
    a_links = list(filter(is_valid_link, a_links))

    category_links = [link for link in a_links if is_category_link(link)]
    category_links.append(main_url)

    # 카테고리별로 레시피 링크 수집
    for category in category_links:
        driver.get(category)
        recipe_links.extend([link for link in a_links if is_recipe_link(link)])

    # 중복 제거
    recipe_links = list(set(recipe_links))
    with open(file_path, 'w', encoding='utf8') as fline:
        for link in recipe_links:
            fline.write(link + '\n')

# test용
cnt = 0


# 레시피 링크별 데이터 스크래핑
for recipe in recipe_links:
    #if cnt == 5: break
    #cnt += 1

    driver.get(recipe)
    driver.implicitly_wait(IMPLICIT_WAIT) # 컨텐츠 로딩까지 잠시 대기

    container = driver.find_element(By.TAG_NAME, 'article')
    header = container.find_element(By.CLASS_NAME, 'loc.article-post-header')

    if mode == COLLECT_PROFILES:
        item_profile = dict()
        item_profile['ID'] = item_id
        item_profile['Name'] = header.find_element(By.TAG_NAME, 'h1').text
        rating_avg = "NA"
        try:
            rating_avg = header.find_element(By.ID, 'mntl-recipe-review-bar__rating_2-0').text

        except exceptions.NoSuchElementException:
            pass

        finally:
            item_profile['AverageRating'] = rating_avg

        content = container.find_element(By.CLASS_NAME, 'loc.article-content')
        content_data = content.find_element(By.ID, 'article-content_1-0')
        recipe_details = content_data.find_element(By.ID, 'recipe-details_1-0')
        recipe_details_items = recipe_details.find_elements(By.CLASS_NAME, 'mntl-recipe-details__item')

        for item in recipe_details_items:
            recipe_details_label = item.find_element(By.CLASS_NAME, 'mntl-recipe-details__label').text[:-1]
            recipe_details_value = item.find_element(By.CLASS_NAME, 'mntl-recipe-details__value').text
            item_profile[recipe_details_label] = recipe_details_value

        ingredients_list = content_data.find_element(By.CLASS_NAME, 'mntl-structured-ingredients__list')
        ingredients_items = ingredients_list.find_elements(By.CLASS_NAME, 'mntl-structured-ingredients__list-item')
        ingredients = []

        for item in ingredients_items:
            ingredient = item.find_elements(By.TAG_NAME, 'span')
            ingredient_quantity = ingredient[0].text + ' '
            ingredient_unit = ingredient[1].text + ' '
            ingredient_name = ingredient[2].text
            #ingredients.append({"name": ingredient_name,
            #                    "quantity": ingredient_quantity,
            #                    "unit": ingredient_unit})
            ingredients.append(ingredient_quantity + ingredient_name + ingredient_unit)
        item_profile["Ingredients"] = ingredients

        directions_list = content_data.find_element(By.ID, 'mntl-sc-block_2-0')
        directions_steps = directions_list.find_elements(By.TAG_NAME, 'p')
        directions = [step.text for step in directions_steps]

        item_profile["Directions"] = directions

        item_list.append(item_profile)

    elif mode == COLLECT_REVIEWS:
        content = container.find_element(By.CLASS_NAME, 'loc.article-content')
        content_data = content.find_element(By.ID, 'article-content_1-0')
        review_container = content_data.find_element(By.CLASS_NAME, 'feedback-list')

        # 리뷰가 있으면 리뷰 수집
        try:
            reviews_list = review_container.find_element(By.CLASS_NAME, 'feedback-list__items')
        # 리뷰가 없으면 다음 레시피로
        except exceptions.NoSuchElementException:
            continue

        while True:
            # 로드 버튼이 있으면 클릭
            try:
                load_review_button = reviews_list.find_element(By.CLASS_NAME, 'feedback-list__load-more')
                load_review_button = load_review_button.find_element(By.TAG_NAME, 'button')


                load_review_button.click()
                time.sleep(1)
                driver.implicitly_wait(IMPLICIT_WAIT)  # 컨텐츠 로딩까지 잠시 대기

            except exceptions.NoSuchElementException:
                break

        review_items = reviews_list.find_elements(By.CLASS_NAME, 'feedback-list__item')

        for item in review_items:
            review_profile = dict()

            review_profile['Cocktail'] = item_id

            review_profile['User'] = item.find_element(By.CLASS_NAME, 'feedback__display-name').text

            review_display = item.find_element(By.CLASS_NAME, 'feedback__text')
            review_paras = review_display.find_elements(By.TAG_NAME, 'p')
            review_feedback = ""
            for para in review_paras:
                review_feedback += para.text + ' '
            review_profile['Review'] = review_feedback

            review_ratings = item.find_element(By.CLASS_NAME, 'feedback__stars')
            full_stars = review_ratings.find_elements(By.CLASS_NAME, 'icon.ugc-icon-star.ugc-icon-avatar-null')
            review_profile['Rating'] = len(full_stars)

            review_list.append(review_profile)
    item_id += 1

# get the page source
page_source = driver.page_source

# close the browser
driver.close()

if mode == COLLECT_PROFILES:
    # 프로필 저장
    output_json = {"cocktail_profiles": item_list}
    with open("all_recipe_cocktail_profiles.json", "w", encoding='utf8') as outfile:
        json.dump(output_json, outfile, ensure_ascii=False, indent=4)

elif mode == COLLECT_REVIEWS:
    output_json = {"cocktail_reviews": review_list}
    with open("all_recipe_cocktail_reviews.json", "w", encoding='utf8') as outfile:
        json.dump(output_json, outfile, ensure_ascii=False, indent=4)


print('category')

print('recipe')
print(len(recipe_links))
for l in recipe_links:
    print(l)

for item in item_list:
    print(item)



