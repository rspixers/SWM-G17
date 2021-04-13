from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin

import "../prediction.py"


app = Flask(__name__)
# CORS(app, support_credentials=True)

@app.route('/api/amazon', methods=['POST','GET'])
@cross_origin(supports_credentials=True)
def amazon_ml():
    text = request.args.get('news_text')
    model = request.args.get('model')

    #condintional checks
    result = predictStockPrices(text,model,'Amazon')

    return jsonify({"svm": True, "LR": False})

@app.route('/api/apple', methods=['POST','GET'])
@cross_origin(supports_credentials=True)
def apple_ml():
    text = request.args.get('news_text')
    model = request.args.get('model')

    # condintional checks
    result = predictStockPrices(text,model,'Apple')
    return jsonify({"Random Forest": True, "LR": False})


@app.route('/')
@cross_origin(supports_credentials=True)
def hello_world():
    return 'Hello, World!'


if __name__ == "__main__":
  app.run(debug=True)