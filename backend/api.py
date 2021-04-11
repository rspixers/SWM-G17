from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin


app = Flask(__name__)
CORS(app, support_credentials=True)

@app.route('/api/add_message/<uuid>', methods=['GET', 'POST'])
@cross_origin(supports_credentials=True)
def add_message(uuid):
    content = request.json
    print(content['mytext'])
    return jsonify({"uuid":uuid})

if __name__ == "__main__":
  app.run(host='0.0.0.0', port=8000, debug=True)