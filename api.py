from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin

from prediction import *

app = Flask(__name__)
CORS(app, support_credentials=True)

@app.route('/api/amazon', methods=['POST','GET'])
@cross_origin(supports_credentials=True)
def amazon_ml():
    text = request.args.get('news_text')
    model = request.args.get('model')
    #condintional checks
    result = prediction(text,model,'Amazon')
    return result

@app.route('/api/apple', methods=['POST','GET'])
@cross_origin(supports_credentials=True)
def apple_ml():
    text = request.args.get('news_text')
    model = request.args.get('model')

    # condintional checks
    result = prediction(text,model,'Apple')
    return result


@app.route('/')
@cross_origin(supports_credentials=True)
def hello_world():
    return 'Hello, World!'


if __name__ == "__main__":
  app.run(debug=True)