from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin


app = Flask(__name__)
CORS(app, support_credentials=True)

@app.route('/api/apple', methods=['POST','GET'])
@cross_origin(supports_credentials=True)
def amazon_ml():
    content = request.json
    print(content['mytext'])
    print("apple")
    return jsonify({"label":1})

@app.route('/api/amazon', methods=['POST','GET'])
@cross_origin(supports_credentials=True)
def apple_ml():
    content = request.json
    print(content['mytext'])
    return jsonify({"label":1})


if __name__ == "__main__":
  app.run(host='0.0.0.0', port=8000, debug=True)