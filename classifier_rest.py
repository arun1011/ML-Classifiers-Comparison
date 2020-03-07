from flask import Flask, request, jsonify
import ml_model
import json
import os 
import shutil


app = Flask(__name__)

class SyntBotsAIException(Exception):
    status_code = 400
    def __init__(self, message, status_code=None, payload=None):
        Exception.__init__(self)
        self.message = message
        if status_code is not None:
            self.status_code = status_code
        self.payload = payload

    def to_dict(self):
        rv = dict(self.payload or ())
        rv['message'] = self.message
        return rv


@app.errorhandler(SyntBotsAIException)
def handle_invalid_usage(error):
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response

@app.route('/ai/classifierstatus', methods = ['GET'])

def getHealthstatus():
    return "classifier is executing"

def fileProcess(uploaded_file):
    uploaded_file = uploaded_file.replace('\\','/')
    uploaded_file_name = uploaded_file.split('/')[-1]
    doc_processed_folder = approot+'Input/'
    doc_processed_file = doc_processed_folder+"/"+uploaded_file_name
    print("doc_processed_file>>>>>>>>"+doc_processed_file)
    print("uploaded_file>>>>>>>>"+uploaded_file)
    file_exist = os.path.exists(doc_processed_file)
    if not file_exist:
        shutil.move(uploaded_file, doc_processed_file)
    return doc_processed_file


  
@app.route('/ai/summarize', methods = ['POST'])
def summarize():
    req = request.json; print(req)
    response = {}
    status = 1
    summarizeResp = {}
    try:
        uploaded_file = req['inputFile']
        features = req['features']
        summarizeResp = ml_model.summarize(uploaded_file,features)
        print('api response--->', summarizeResp)
    except Exception as ae:
           print(ae)
           status=500
           err_msg = ""
           if hasattr(ae, 'message'):
               err_msg = ae.message
           else:
               err_msg = str(ae)
           raise SyntBotsAIException(err_msg, status_code=status)
    response['ai_response'] = summarizeResp
    response['ai_status'] = status
    response=json.dumps(response); print(response)
    return response    

@app.route('/ai/compare', methods = ['POST'])
def compare():
    req = request.json; print(req)
    response = {}
    status = 1
    compareResp = []
    try:
        uploaded_file = req['inputFile']
        features = req['features']
        vectorize = req['vectorize']
        compareResp = ml_model.compare(uploaded_file,features,vectorize)
        print(compareResp)
    except Exception as ae:
           print(ae)
           status=500
           err_msg = ""
           if hasattr(ae, 'message'):
               err_msg = ae.message
           else:
               err_msg = str(ae)
           raise SyntBotsAIException(err_msg, status_code=status)
    response['ai_response'] = compareResp
    response['ai_status'] = status
    response=json.dumps(response); print(response)
    return response    

@app.route('/ai/train', methods = ['POST'])
def train():
    req = request.json; print(req)
    response = {}
    status = 1
    compareResp = []
    try:
        uploaded_file = req['inputFile']
        features = req['features']
        vectorize = req['vectorize']
        modelKey = req['modelKey']
        modelName = req['modelName']
        compareResp = ml_model.train(uploaded_file,features,modelKey,vectorize,modelName)
        print(compareResp)
    except Exception as ae:
           print(ae)
           status=500
           err_msg = ""
           if hasattr(ae, 'message'):
               err_msg = ae.message
           else:
               err_msg = str(ae)
           raise SyntBotsAIException(err_msg, status_code=status)
    response['ai_response'] = compareResp
    response['ai_status'] = status
    response=json.dumps(response); print(response)
    return response    

	
@app.route('/ai/predict', methods = ['POST'])
def predict():
    req = request.json; print(req)
    response = {}
    status = 1
    compareResp = []
    try:
        text = req['text']
        vectorize = req['vectorize']
        model_name = req['modelName']
        compareResp = ml_model.predict(text,vectorize,model_name)
        print(compareResp)
    except Exception as ae:
           print(ae)
           status=500
           err_msg = ""
           if hasattr(ae, 'message'):
               err_msg = ae.message
           else:
               err_msg = str(ae)
           raise SyntBotsAIException(err_msg, status_code=status)
    response['ai_response'] = compareResp
    response['ai_status'] = status
    response=json.dumps(response); print(response)
    return response   
	
@app.route('/ai/getAlgorithms', methods = ['POST'])
def getAlgorithms():
    response = {}
    algoKeys = []
    status = 1
    try:
        algorithms = ml_model.get_models()
        for key,val in algorithms.items():
            algoKeys.append(key)
    except Exception as ae:
           print(ae)
           status=500
           err_msg = ""
           if hasattr(ae, 'message'):
               err_msg = ae.message
           else:
               err_msg = str(ae)
           raise SyntBotsAIException(err_msg, status_code=status)
    response['ai_response'] = algoKeys
    response['ai_status'] = status
    response=json.dumps(response); print(response)
    return response  
      
if __name__ == "__main__":
    app.run(host='0.0.0.0',port=5004,debug=True)
