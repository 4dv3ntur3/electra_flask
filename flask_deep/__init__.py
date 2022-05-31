import os, sys
from flask import Flask, request,  Response
# from werkzeug import secure_filename # WSGI 
from werkzeug.utils import secure_filename
import electra_infer # from . import electra_infer 이런 거 할 필요 없음 (같은 디렉토리 내)
import torch
import configparser

# for load model
from transformers import AutoModelForTokenClassification, ElectraTokenizer

app = Flask(__name__)
app.debug = True

model_dict = None


def config_read():

    config = configparser.ConfigParser()
    config.read('config.ini', encoding='utf-8')

    return config['model']['model_dir']



# 최초 request 직전에만 수행
@app.before_first_request
def before_first_request():

    cwd = os.getcwd() # 이 부분 config로 수정할 수 있도록 (현재 working directory 기준으로 잡힘 )

    model_dir = config_read()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    training_params = torch.load(os.path.join(model_dir, 'training_parameters.bin'))

    tokenizer = ElectraTokenizer.from_pretrained(training_params['training_args'].model_name_or_path)

    # Check whether model exists
    if not os.path.exists(model_dir):
        raise Exception("Model doesn't exists! Train first!")

    try:
        model = AutoModelForTokenClassification.from_pretrained(model_dir)  # Config will be automatically loaded from model_dir
        model.to(device)
        model.eval()
        print("***** Model Loaded *****")

    except:
        raise Exception("Some model files might be missing...")

    
    model_dict_temp = {
        'model': model,
        'model_dir': model_dir,
        'device': device,
        'training_params': training_params,
        'tokenizer': tokenizer
    }

    global model_dict
    model_dict = model_dict_temp

@app.route('/predict', methods=['GET','POST'])
def predict():

    global model_dict
    if model_dict is None:
        print("something is wrong... check model... ")
        

    if request.method == 'POST':

        input_txt = request.json['text']

        result_txt = electra_infer.main(input_txt, model_dict)

    elif request.method == 'GET':
        # 이 부분은 request parameter로 받아오는 부분... 수정 필요?
        pass
 
    return Response(result_txt, mimetype="application/json", status=200)


if __name__ == "__main__":
    app.run(host='127.0.0.1')

