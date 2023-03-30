from flask import Flask, make_response, request
import json
import numpy as np
import pandas as pd
import pickle
from modulo_hernandez_gonzalez_ricardo_paramont import CategorizeSalary

#importando pipeline de preprocesamiento de dataframe
with open('pipeline.pkl','rb') as f:
    pipeline = pickle.load(f)

#importando modelo de clasificacion
with open('best_gb.pkl','rb') as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    json_data = request.json
    register_df = pd.json_normalize(json_data)
    print(register_df.columns)
    register_df = register_df.replace([-1,'-1','Unknown / Non-Applicable'], np.nan)
    register_df = pipeline.transform(register_df)
    categorize = CategorizeSalary()
    prediction = categorize.inverse_transform( pd.Series( model.predict(register_df) ) )[0]
    return json.dumps({"prediction":f'{prediction}'})
    #return json_data

if __name__ == '__main__':
     app.run(port=8080)