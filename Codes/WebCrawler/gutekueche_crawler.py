import time

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common import exceptions
import os
import json

IMPLICIT_WAIT = 10
COLLECT_PROFILES = 1
COLLECT_REVIEWS = 2
END_OF_PAGE = 83
TEST_END_OF_PAGE = 3
scrape_range = 4 # scrape_range~scrape_range+200

file_name = 'gutekueche.txt'
file_path = os.path.join(os.path.curdir, file_name)
need_crawl = False

recipe_links = list()
item_id = 1 + 200 * scrape_range
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
    link = str(link)
    tail_of_link = link[-5:]

    is_digit = [c.isdigit() for c in tail_of_link]
    cond1 = all(is_digit)
    cond2 = is_valid_link(link)
    return cond1 and cond2


# create a new browser instance
driver = create_driver(headless=False)

# 레시피 링크 리스트 파일이 없으면 링크 탐색
if os.path.isfile(file_path):
    with open(file_path, 'r', encoding='utf8') as file:
        for line in file:
            recipe_links.append(line[:-1])

else:

    # navigate to a website
    start_url = "https://www.gutekueche.at/cocktail-alle-rezepte?p="

    # 83 page까지
    for page_num in range(1, END_OF_PAGE + 1):

        driver.get(start_url + str(page_num))
        driver.implicitly_wait(IMPLICIT_WAIT)

        # get all the links on the page
        links = driver.find_elements(By.TAG_NAME, "a")
        for link in links:
            href = link.get_attribute("href")
            if is_recipe_link(href):
                recipe_links.append(href)
                print(href)

    # 중복 제거
    recipe_links = list(set(recipe_links))
    with open(file_path, 'w', encoding='utf8') as fline:
        for link in recipe_links:
            fline.write(link + '\n')

print(recipe_links)
from_ = scrape_range*200
end_ = min([from_ + 200, len(recipe_links)])
recipe_links = recipe_links[from_:end_]

# test용
cnt = 0

# 레시피 링크별 데이터 스크래핑
for recipe in recipe_links:
    # if cnt == 5: break
    if cnt == 529:
        item_id += 1
        continue
    cnt += 1
    driver.get(recipe+'?comments=all#comment_sort')
    if cnt == 1:
        time.sleep(10)
    driver.implicitly_wait(IMPLICIT_WAIT) # 컨텐츠 로딩까지 잠시 대기
    time.sleep(2)

    container = driver.find_element(By.ID, 'middle')

    if mode == COLLECT_PROFILES:
        article = container.find_element(By.TAG_NAME, 'article')
        item_profile = dict()
        item_profile['ID'] = item_id
        item_profile['Name'] = article.find_element(By.TAG_NAME, 'h1').text
        rating_avg = "NA"
        try:
            rating_avg = article.find_element(By.CLASS_NAME, 'rateit_value').text

        except exceptions.NoSuchElementException:
            pass

        finally:
            item_profile['AverageRating'] = rating_avg

        content = article.find_element(By.CLASS_NAME, 'teaser-detail.sec')
        item_profile["description"] = content.find_element(By.TAG_NAME, 'p').text
        ingredients_list = container.find_element(By.CLASS_NAME, 'ingredients-table')
        ingredients_items = ingredients_list.find_elements(By.TAG_NAME, 'tr')
        ingredients = []

        for item in ingredients_items:
            ingredient_quantity = item.find_element(By.TAG_NAME, 'td').text + ' '
            ingredient_unit_and_name = item.find_elements(By.TAG_NAME, 'th')
            ingredient_unit = ingredient_unit_and_name[0].text + ' '
            ingredient_name = ingredient_unit_and_name[1].text + ' '
            #ingredients.append({"name": ingredient_name,
            #                    "quantity": ingredient_quantity,
            #                    "unit": ingredient_unit})
            ingredients.append(ingredient_quantity + ingredient_unit + ingredient_name)

        item_profile["Ingredients"] = ingredients

        cocktail_keywords_div = article.find_element(By.CLASS_NAME, 'recipe-categories')
        cocktail_keywords = cocktail_keywords_div.find_elements(By.TAG_NAME, 'a')
        keywords = [keyword.text for keyword in cocktail_keywords]

        item_profile["Keywords"] = keywords[:-1]

        directions_section = article.find_element(By.CLASS_NAME, 'sec.rezept-preperation')
        directions_list = directions_section.find_element(By.TAG_NAME, 'ol')
        directions_steps = directions_list.find_elements(By.TAG_NAME, 'li')
        directions = [step.text for step in directions_steps]
        item_profile["Directions"] = directions

        item_list.append(item_profile)

    elif mode == COLLECT_REVIEWS:
        comment_section = container.find_element(By.ID, 'plugin-user-comments')

        try:
            comments = comment_section.find_elements(By.CLASS_NAME, 'comment.comment-l1.grid')
            for comment in comments:
                user_id = comment.find_element(By.TAG_NAME, 'strong').text
                comment_text = comment.find_element(By.TAG_NAME, 'p').text
                time_stamp = comment.find_element(By.TAG_NAME, 'time').text
                review_profile = dict()
                review_profile['Cocktail'] = item_id
                review_profile['User'] = user_id
                review_profile['Review'] = comment_text
                review_profile['Timestamp'] = time_stamp
                review_list.append(review_profile)

        except exceptions.NoSuchElementException:
            pass

    item_id += 1

# get the page source
page_source = driver.page_source

# close the browser
driver.close()

if mode == COLLECT_PROFILES:
    # 프로필 저장
    output_json = {"cocktail_profiles": item_list}
    with open("../../Dataset/gutekueche_cocktail_profiles.json", "w", encoding='utf8') as outfile:
        json.dump(output_json, outfile, ensure_ascii=False, indent=4)

elif mode == COLLECT_REVIEWS:
    output_json = {"cocktail_reviews": review_list}
    with open("gutekueche_cocktail_reviews_"+str(scrape_range)+".json", "w", encoding='utf8') as outfile:
        json.dump(output_json, outfile, ensure_ascii=False, indent=4)


print(len(recipe_links))
for l in recipe_links:
    print(l)

for item in item_list:
    print(item)

