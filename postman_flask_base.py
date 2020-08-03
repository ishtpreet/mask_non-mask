from flask import Flask, request
import json
import numpy as np
import cv2

app = Flask(__name__)
@app.route('/')
@app.route('/home', methods=['POST'])
def home():
    requested_data = request.get_json()
    name = requested_data['name']
    return '''{}'''.format(name)

if __name__=="__main__":
    app.run(debug=True)