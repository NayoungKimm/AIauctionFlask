from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import Select
from bs4 import BeautifulSoup
from selenium.common.exceptions import TimeoutException
import re
from selenium.common.exceptions import NoSuchElementException
import time
from selenium.common.exceptions import StaleElementReferenceException
import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
from sklearn.impute import SimpleImputer
from flask import Flask, request, jsonify,redirect,url_for,render_template
import logging
from flask_cors import CORS
import os
from flask import Flask, render_template, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from urllib.parse import quote_plus
import jwt

app = Flask(__name__)
CORS(app, resources={r"/submit": {"origins": "http://127.0.0.1:5000"}})

# 데이터베이스 설정
password = "Kknnyy0819@@!"
url_encoded_password = quote_plus(password)
app.config[
    "SQLALCHEMY_DATABASE_URI"
] = f"mysql+pymysql://root:{url_encoded_password}@localhost/Estate_db"
db = SQLAlchemy(app)


# User 모델 정의
class Users(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    gender = db.Column(db.String(10), nullable=False)
    name = db.Column(db.String(20), nullable=False)


# 사용자 정보를 데이터베이스에 저장
def create_user(email, gender, nickname):
    user = Users.query.filter_by(email=email).first()
    if user:
        return "existing_user"

    new_user = Users(email=email, gender=gender, name=nickname)
    db.session.add(new_user)
    db.session.commit()
    return "new_user"


@app.route('/submit', methods=['POST'])
def submit_data():
    global input_year, keyword,weighted_final_price
    data = request.json
    courtname_input = data['courtname_input']
    input_year = float(data['input_year'])
    keyword = int(data['keyword'])
    역과의거리 = float(data['역과의거리'])

    # 로그 출력
    app.logger.info(f"Received data - courtname_input: {courtname_input}, input_year: {input_year}, keyword: {keyword}, 입력_역과의거리: {역과의거리}")
    # TODO: 위에서 받은 데이터로 원하는 작업 수행
    data = pd.read_csv("final_database.csv")

    chrome_options = Options()
    chrome_options.add_argument("--headless")  # 이 부분을 추가합니다.
    chrome_service = Service(executable_path='/Users/ny/Downloads/chromedriver-mac-x64/chromedriver')
    driver = webdriver.Chrome(service=chrome_service, options=chrome_options)
    driver.get('https://www.speedauction.co.kr/v3/')

    # 아이디와 비밀번호 입력
    username = "rudfks1212"
    password = "rudfks1212"
    number="12"

    # 아이디와 비밀번호 입력 부분의 요소를 찾고 값을 설정합니다.
    username_input = driver.find_element(By.NAME, "id")
    password_input = driver.find_element(By.NAME, "pw")

    # 아이디 및 비밀번호 입력
    username_input.clear()
    username_input.send_keys(username)
    password_input.clear()
    password_input.send_keys(password)

    # 로그인 버튼 클릭 대기 설정
    wait = WebDriverWait(driver, 10)
    login_button = wait.until(EC.element_to_be_clickable((By.XPATH, '/html/body/table/tbody/tr[2]/td/table/tbody/tr/td/table/tbody/tr/td[2]/table/tbody/tr[3]/td/table/tbody/tr[1]/td/table/tbody/tr[2]/td/table/tbody/tr/td[1]/table/tbody/tr[1]/td/table/tbody/tr[3]/td/table/tbody/tr/td[3]/input')))

    # 로그인 버튼 클릭
    login_button.click()

    # 이미지 클릭
    image_element = driver.find_element(By.ID, "navi_img1")
    image_element.click()

    # 지역 선택 (서울)
    region_select = driver.find_element(By.NAME, "region_code1")
    region_select.find_element(By.XPATH, "./option[@value='1100000000']").click()

    # 다세대(빌라) 체크박스 선택
    villa_checkbox = driver.find_element(By.XPATH, "//input[@title='다세대(빌라)']")
    villa_checkbox.click()

    # 날짜 설정
    start_date_input = driver.find_element(By.ID, "sell_yyyymmdd_ss")
    start_date_input.clear()
    start_date_input.send_keys("2016-01-01")

    end_date_input = driver.find_element(By.ID, "sell_yyyymmdd_ee")
    end_date_input.clear()
    end_date_input.send_keys("2022-12-31")

    # 이름 입력 받기
    name_input = courtname_input
        
    # 셀렉트 박스 요소 찾기
    court_select = Select(driver.find_element(By.NAME, "courtNo_main"))
        
    # 이름에 해당하는 값을 선택
    for option in court_select.options:
        if name_input in option.text:
            option.click()
            break

    # '년도를 입력해주세요'에 사용자 입력 값을 입력합니다.
    year_input = driver.find_element(By.XPATH, '/html/body/table/tbody/tr[2]/td/table/tbody/tr/td/table/tbody/tr[2]/td/table/tbody/tr/td[3]/table/tbody/tr[2]/td/table/tbody/tr[3]/td/table/tbody/tr[2]/td/table/tbody/tr[2]/td/table/tbody/tr[2]/td[4]/select')
    year_input.send_keys(input_year)

    # 사용자에게 텍스트를 입력받아 <input name="eventNo2"> 요소에 값을 설정
    text_input = driver.find_element(By.XPATH,'/html/body/table/tbody/tr[2]/td/table/tbody/tr/td/table/tbody/tr[2]/td/table/tbody/tr/td[3]/table/tbody/tr[2]/td/table/tbody/tr[3]/td/table/tbody/tr[2]/td/table/tbody/tr[2]/td/table/tbody/tr[2]/td[4]/input')
    text_input.send_keys(keyword)

    입력_역과의거리=역과의거리

    # 검색 클릭
    search_image = driver.find_element(By.XPATH, "/html/body/table/tbody/tr[2]/td/table/tbody/tr/td/table/tbody/tr[2]/td/table/tbody/tr/td[3]/table/tbody/tr[2]/td/table/tbody/tr[3]/td/table/tbody/tr[2]/td/table/tbody/tr[3]/td/img")
    search_image.click()

    # 주어진 CSS selector에 해당하는 요소의 텍스트를 가져옵니다.
    element_text = driver.find_element(By.CSS_SELECTOR, "body > table > tbody > tr:nth-child(2) > td > table > tbody > tr > td > table > tbody > tr:nth-child(2) > td > table > tbody > tr > td:nth-child(3) > table:nth-child(9) > tbody > tr:nth-child(1) > td > table > tbody > tr:nth-child(13) > td > table > tbody > tr:nth-child(3) > td:nth-child(4) > table > tbody > tr > td:nth-child(2)").text
    해당매물_임차인대항력=0.0
    # "대항력있는임차인"이라는 단어가 있는지 확인합니다.
    if "대항력있는임차인" in element_text:
        해당매물_임차인대항력 = 1
    else:
        해당매물_임차인대항력 = 0

    print(f"해당매물_임차인_대항력: {해당매물_임차인대항력}")

    # 주어진 CSS selector에 해당하는 요소의 텍스트를 가져옵니다.
    element_text = driver.find_element(By.CSS_SELECTOR, "body > table > tbody > tr:nth-child(2) > td > table > tbody > tr > td > table > tbody > tr:nth-child(2) > td > table > tbody > tr > td:nth-child(3) > table:nth-child(9) > tbody > tr:nth-child(1) > td > table > tbody > tr:nth-child(13) > td > table > tbody > tr:nth-child(3) > td:nth-child(7)").text

    # "[입찰" 다음에 나오는 숫자를 추출합니다.
    match = re.search(r'\[입찰(\d+)명\]', element_text)
    해당매물_입찰인수=0.0
    if match:
        해당매물_입찰인수 = float(match.group(1))
    else:
        해당매물_입찰인수 = 0

    print(f"해당매물_입찰인수: {해당매물_입찰인수}")  # 예: 2 또는 0

    # 주어진 CSS selector에 해당하는 요소의 텍스트를 가져옵니다.
    element_text = driver.find_element(By.CSS_SELECTOR, "body > table > tbody > tr:nth-child(2) > td > table > tbody > tr > td > table > tbody > tr:nth-child(2) > td > table > tbody > tr > td:nth-child(3) > table:nth-child(9) > tbody > tr:nth-child(1) > td > table > tbody > tr:nth-child(13) > td > table > tbody > tr:nth-child(3) > td:nth-child(8)").text
    해당매물_유찰수=0.0
    # "X회" 형태의 숫자 X를 추출합니다.
    match = re.search(r'(\d+)회\.', element_text)

    if match:
        number_of_times = float(match.group(1))
    else:
        number_of_times = 0

    해당매물_유찰수 = number_of_times
    print(f"해당매물_유찰수: {해당매물_유찰수}")  # 예: 1 또는 0

    # 검색된 항목중 첫번째 연결 사이트를 nclick 이벤트를 실행하여 검색 수행
    search_image2 = driver.find_element(By.XPATH, "/html/body/table/tbody/tr[2]/td/table/tbody/tr/td/table/tbody/tr[2]/td/table/tbody/tr/td[3]/table[1]/tbody/tr[1]/td/table/tbody/tr[12]/td/table/tbody/tr[3]")
    search_image2.click()

    # search_image2 클릭
    search_image2.click()
    wait = WebDriverWait(driver, 20)

    # 새로 열린 탭으로 전환
    driver.switch_to.window(driver.window_handles[-1])

    print(type(driver))

    try:
        print_area_element = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "#printArea > tbody > tr > td > table > tbody > tr:nth-child(8) > td > table > tbody > tr:nth-child(4) > td:nth-child(2)"))
        )
        # 이후 코드 ...
    except TimeoutException:
        print("Element not found within the given time frame.")

    # BeautifulSoup를 사용하여 나머지 구조를 파싱
    area_value = print_area_element.text
    해당매물_전용=0.0
    match = re.search(r'\((\d+\.\d+)평\)', area_value)
    if match:
        해당매물_전용 = float(match.group(1))
        print(f"해당매물_전용: {해당매물_전용}")


    # 금액 데이터 가져오기
    price_element = driver.find_element(By.ID, 'tdbx01')
    price_value = price_element.text
    match = re.search(r'(\d{1,3}(?:,\d{3})*)(?=원)',price_value)
    해당매물_감정가=0.0
    if match:
        cleaned_value = match.group(1).replace(',', '')
        해당매물_감정가 = float(cleaned_value)
        print(f"해당매물_감정가: {해당매물_감정가}")
    else:
        print("No match found")
        
    # 주어진 CSS selector에 해당하는 요소의 텍스트를 가져옵니다.
    해당매물_대지권=0.0
    element_text = driver.find_element(By.CSS_SELECTOR, "#printArea > tbody > tr > td > table > tbody > tr:nth-child(8) > td > table > tbody > tr:nth-child(3) > td:nth-child(2)").text
    # 괄호 안의 숫자를 추출합니다.
    match = re.search(r'\((\d+\.\d+)평\)', element_text)
    if match:
        해당매물_대지권 = float(match.group(1))
    else:
        해당매물_대지권 = 0

    print(f"해당매물_대지권: {해당매물_대지권}")  # 예: 5.23

        
    # 기존 탭에서 자바스크립트 함수를 호출
    driver.execute_script("goStaticList(2);")
    # 잠시 대기. 새 창이나 탭이 완전히 로드될 때까지 기다립니다.
    time.sleep(2)

    # 새로 열린 창이나 탭으로 포커스 전환
    driver.switch_to.window(driver.window_handles[-1])

    # WebDriverWait를 사용하여 페이지 로드를 기다립니다.
    wait = WebDriverWait(driver, 10)

    # 새 창의 URL 가져오기
    new_tab_url = driver.current_url


    # URL로 요청 보내기
    headers = {"User-Agent": "Your User Agent"}  # 필요에 따라 User-Agent 설정
    response = requests.get(new_tab_url, headers=headers)

    # 응답 상태 코드 확인
    status_code = response.status_code

    extracted_values = []
    size_values = []


    # BeautifulSoup를 사용하여 새로 열린 탭의 내용을 분석
    if status_code == 200:
        response.encoding = 'euc-kr'
        soup_new_tab = BeautifulSoup(response.text, 'html.parser')
        # <td> 태그 내용 추출 및 저장 (align="right", <font> 태그 color="red")
        td_tags = soup_new_tab.find_all('td', align='right')
        for td_tag in td_tags:
            font_tag = td_tag.find('font', color='red')
            if font_tag:
                text = font_tag.get_text()
                cleaned_text = re.sub('매각', '', text)  # ¸Å°¢ 제거
                extracted_values.append(cleaned_text)
                    
        td_tags = soup_new_tab.find_all('td', align='right')  # 모든 해당 td 태그 가져오기

        expect_price = []

        for td_tag in td_tags:
            first_line_font = td_tag.find('font', color="#000000")  # 첫 번째 줄의 font 태그 찾기
            if first_line_font:
                font_lines = first_line_font.stripped_strings  # 모든 문자열을 가져옵니다.
                first_line = next(font_lines, None)  # 첫 번째 문자열만 가져옵니다.
                if first_line:
                    numbers = re.findall(r'[0-9,]+', first_line)
                    if numbers:
                        clean_number = numbers[0].replace(',', '').replace('\'', '')
                        expect_price.append(clean_number)  # 첫 번째로 추출된 숫자 

        print(expect_price)
                
        font_tags = soup_new_tab.find_all('font', color='00479F')
        print(f"Found {len(font_tags)} font tags with color 00479F")  # 몇 개의 font 태그를 찾았는지 출력
        for font_tag in font_tags:
            text = font_tag.get_text().strip()

            # cleaned_text 초기화
            cleaned_text = text
            cleaned_text = re.sub(r'\[\s+(\d)', r'[\1', cleaned_text)

            match = re.search(r'\[전용 ([0-9.]+)평\]', cleaned_text)
            if match:
                size_value = float(match.group(1))
                size_values.append(size_value)

    else:
        print("Failed to fetch the webpage. Status code:", status_code)

    # 가설1
    data_1 = {'매각가격': extracted_values,'전용':size_values,'감정가':expect_price}
    df = pd.DataFrame(data_1)

    # 콤마(,) 제거 및 숫자로 변환
    df['매각가격'] = df['매각가격'].str.replace(',', '').astype(float)
    if df['전용'].dtype != 'object':
        # 이미 숫자 타입인 경우
        df['전용'] = df['전용'].astype(float)
    else:
        # 문자열일 경우 콤마 제거 및 숫자로 변환
        df['전용'] = df['전용'].str.replace(',', '').astype(float)
        
    df['감정가'] = df['감정가'].astype(float)

    # 매각가격을 전용으로 나눈 값 저장
    df['평당시세'] = df['매각가격'] / df['전용']
    df['낙찰가율'] = df['매각가격'] / df['감정가']
    # 평균 계산
    averageroom_ratio = df['평당시세'].mean()
    averageprice_ratio=df['낙찰가율'].mean()
    print("평균평당시세: " + str(averageroom_ratio))
    print("평균낙찰가율: " + str(averageprice_ratio))
    #가설1 최종계산
    hypothe1=averageroom_ratio*해당매물_전용
    #가설2 최종계산
    hypothe2=averageprice_ratio*해당매물_감정가
    print("가설1(매각가/전용*해당매물전용): " + str(hypothe1))
    print("가설2(주변낙찰가율*감정가): " + str(hypothe2))

    #-------------------------------------------------------

    # 데이터 전처리
    cols = ['대지권', '전용', '감정가', '입찰인수', '유찰수','역과의거리','임차인대항력']
    for col in cols:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        data = data[~((data[col] < (Q1 - 1.5 * IQR)) | (data[col] > (Q3 + 1.5 * IQR)))]
        
    x = data[cols]
    y = data[['낙찰가']]
    data.dropna(subset=cols, inplace=True)
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, test_size=0.3)
    imputer = SimpleImputer(strategy='mean')
    x_train = imputer.fit_transform(x_train)
    y_train = imputer.fit_transform(y_train)
    x_test = imputer.fit_transform(x_test)
    y_test = imputer.fit_transform(y_test)
    mlr = LinearRegression()
    mlr.fit(x_train, y_train)

    predict_data = [[해당매물_대지권, 해당매물_전용, 해당매물_감정가, 해당매물_입찰인수, 해당매물_유찰수, 입력_역과의거리, 해당매물_임차인대항력]]
    # 모델을 사용하여 예측 수행
    predicted_price = mlr.predict(predict_data)
    # 가중치 정의
    weight_hypothe1 = 0.7
    weight_hypothe2 = 0.1
    weight_predicted = 0.9

    # 가중치 적용하여 가격 예측
    weighted_price = weight_hypothe1 * hypothe1 + weight_hypothe2 * hypothe2
    weighted_final_price = 0.5 * weighted_price + weight_predicted * predicted_price[0][0]
    # 예측된 낙찰가 출력
    #print(f"예측된 낙찰가: {weighted_final_price:,.0f} 원")
    #return jsonify(status="success", message="Data 계산완료!")
    logging.debug("In the submit endpoint")
    logging.debug(f"input_year: {input_year}")
    logging.debug(f"keyword: {keyword}")
    # 다른 값을 로그로 찍고 싶으면 여기에 추가
    driver.quit()
    
    return jsonify({"status": "success", "redirect": url_for('result')})

@app.route("/login", methods=['GET',"POST"])
def login():
    kakao_token = request.json.get("access_token")
    if kakao_token is None:
        return jsonify({"error": "No access token"}), 400

    headers = {"Authorization": f"Bearer {kakao_token}"}
    response = requests.get("https://kapi.kakao.com/v2/user/me", headers=headers)
    if response.status_code != 200:
        return jsonify({"error": "Invalid access token"}), 400
    kakao_user = response.json()

    # kakao_user 정보를 출력합니다.
    print(f"Received token: {kakao_token}")
    print(f"User Info: {kakao_user}")

    email = kakao_user["kakao_account"]["email"]
    gender = kakao_user["kakao_account"]["gender"]
    try:
        name = kakao_user["kakao_account"]["profile"]["nickname"]
    except KeyError:
        name = "default_name"  # Set default name if nickname does not exist

    message = create_user(email, gender, name)

    token = jwt.encode(
        {"email": email, "gender": gender, "name": name},
        "Kknnyy0819@@!",
        algorithm="HS256",
    )

    print(f"Generated JWT Token: {token}")

    return jsonify({"message": message, "token": token})


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/details')
def details():
    return render_template('details.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/login2')
def login2():
    return render_template('login2.html')


@app.route('/result')
def result():
    # 로직 처리 (예: 데이터 가져오기)
    info = {
        'input_year': int(input_year),
        'keyword': keyword,
        'weighted_final_price': int(weighted_final_price)
    }
    return render_template('result.html', user=info)

input_year = 0.0
keyword = 0.0
weighted_final_price=0.0

# 로깅 설정
logging.basicConfig(level=logging.DEBUG)

if __name__ == '__main__':
    app.run(debug=True)
