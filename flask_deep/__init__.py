'''

flask app 실행을 위한 파일
flask 파라미터로 전달되는 __name__ 파라미터 : flask app을 구분하기 위한 구분자로 사용
app.debug = True : 코드 수정 시 바로바로 디버깅 가능

@app.route: 페이지 URL과 함수를 연결 (그 페이지에서 들어온 입력을 중개)
render_template: 해당 경로를 웹 브라우저로 전달한다

'''

import os, sys
from re import A
from flask import Flask, escape, request,  Response, g, make_response
from flask import jsonify
from flask.templating import render_template
from werkzeug import secure_filename # WSGI 
from . import electra_infer
import json 
 
app = Flask(__name__)
app.debug = True # 코드 수정 시 바로바로 디버깅 
 
# Main page
@app.route('/')
def index():
    return render_template('index.html')
 
@app.route('/nst_get')
def nst_get():
    
    
    
    return render_template('nst_get.html')


'''
input: jsonified input text
output: result json 
'''

# @app.route('/nst_post', methods=['GET','POST'])
# def nst_post():
    
#     if request.method == 'POST':
        
#         data = request.form['input_txt']

#         '''
#         json으로 들어온 입력 처리 
#         '''
#         # print(request.is_json)
#         # parmas = request.get_json()

#     # input, predict 결과 전송 
#     return render_template('nst_post.html', result_txt=data)
 
@app.route('/nst_post', methods=['GET','POST'])
def nst_post():
    if request.method == 'POST':
        # User Image (target image)
        # form태그에서 input type="file"로 넘어오면 .save, .filename 등의 멤버 변수(?)가 자동으로 생기는 듯
        # model_dir, input_txt_dir : json
         
        user_txt = request.files['input_txt']
        user_txt.save('./flask_deep/static/images/'+str(user_txt.filename))
        user_txt_path = './flask_deep/static/images/'+str(user_txt.filename)
 
        # predict code 호출 
        result_txt = electra_infer.main(user_txt_path) #return 하게 해야겠네 
 
    # input, predict 결과 전송 
    return render_template('nst_post.html', result_txt=result_txt)