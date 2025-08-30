# 라이브러리 불러오기
import smtplib
import requests
import json

# 개인 정보 입력(email, 앱 비밀번호)
my_email = "wj.jang.wooriib@gmail.com"
password = "vvsh cvxa xjuw dqqd"

def send_mail(mail_addr: str, msg: str):
    # 방법 2(with 사용)
    with smtplib.SMTP("smtp.gmail.com") as connection:
        connection.starttls() #Transport Layer Security : 메시지 암호화
        connection.login(user=my_email, password=password)
        connection.sendmail(
            from_addr=my_email,
            to_addrs=mail_addr,
            msg=msg
        )

def send_kakao():

    params = {
        'REST_API_key': '1d34c857df2280b64048ec88b9689a84',
        'Redirect_URI': 'https://example.com/oauth&response_type=code',
        'code': '3b2h5Q7EeTcjSpeklN_qSmLPG-0pe_JlYJFMtRRXCQi4zfZJbKf9QAAAAQKFyIgAAABmNWPyAf_A_o_BVb6-Q'
    }

    url = 'https://kauth.kakao.com/oauth/token'
    client_id = params['REST_API_key']
    redirect_uri = params['Redirect_URI']
    code = params['code']

    data = {
        'grant_type': 'authorization_code',
        'client_id': client_id,
        'redirect_uri': redirect_uri,
        'code': code,
    }

    response = requests.post(url, data=data)
    tokens = response.json()

    access_token = tokens['access_token']

    url = "https://kapi.kakao.com/v2/api/talk/memo/default/send"

    headers = {
        "Authorization": "Bearer " + tokens["access_token"]
    }

    data = {
        'object_type': 'text',
        'text': '테스트입니다',
        'link': {
            'web_url': 'https://developers.kakao.com',
            'mobile_web_url': 'https://developers.kakao.com'
        },
        'button_title': '바로 확인'
    }

    data = {'template_object': json.dumps(data)}
    response = requests.post(url, headers=headers, data=data)
    response.status_code

if __name__ == "__main__":
    mail_addr = "am.woojin@gmail.com"
    mail_msg = "Subject:Hello\n\nThis is the body of my email."
    send_mail(mail_addr, mail_msg)
    print("finish to send email!")

    #send_kakao()
    #print("finish to send kakao")
