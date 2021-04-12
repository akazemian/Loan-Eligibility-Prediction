from flask import render_template, request, jsonify, Flask
import flask
import numpy as np
import traceback
import pickle
import pandas as pd
from flask_restful import Resource, Api, reqparse

# App definition
app = Flask(__name__)
api = Api(app)

# importing models
model = pickle.load(open('tuned_model','rb'))

class Predict(Resource):
    def post(self):
        json_data = request.get_json()
        df = pd.DataFrame(json_data,index=[0])
        res = model.predict(df)

        return res.tolist()

# assign endpoint
api.add_resource(Predict, '/predict')      
        


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
