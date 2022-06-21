from flask import Flask, request, jsonify
from transformers import ElectraTokenizerFast, ElectraForTokenClassification, ElectraTokenizer
from transformers import TokenClassificationPipeline
import torch

app = Flask(__name__)
pipeline = None

def init_pipeline():
    global pipeline
    # init config
    tokenizer = ElectraTokenizerFast.from_pretrained('monologg/koelectra-base-v3-discriminator')
    model = ElectraForTokenClassification.from_pretrained("model")
    pipeline = TokenClassificationPipeline(model=model, tokenizer=tokenizer, framework='pt')

@app.route('/pii_demo', methods=['POST'])
def pii_demo():
    if pipeline == None:
        return "Server not ready"
    lines = request.get_json()["lines"]

    if not lines:
        return "Empty sentences requested"
        
    with torch.no_grad():
        sentence_metas = pipeline(lines, batch_size=32)
        respo = [
            [
                {'label' : pii_metas['entity'],
                'start': pii_metas['start'], 
                'end': pii_metas['end']} 
                for pii_metas in sentence_meta] 
                for sentence_meta in sentence_metas
            ]
        json_data = jsonify({"result" : respo})
        del sentence_metas, respo
        return json_data

if __name__ == '__main__':
    init_pipeline()
    app.config['JSON_AS_ASCII'] = False
    app.run(debug=True)