from flask import Flask
from flask import request
import h2o
import pandas as pd

app = Flask(__name__)


@app.route("/predict", methods=['POST'])
def do_prediction():
    model_path = 'output/gradient_boosting_model.hex'
    model = h2o.load_model(model_path)

    json = request.get_json()
    data = pd.DataFrame(json, index=[0])

    testing = pd.DataFrame(data)
    test = h2o.H2OFrame(testing)
    pred_ans = model.predict(test).as_data_frame()
    result = {"Prediction": pred_ans.predict.values[0]}
    return result


if __name__ == "__main__":
    h2o.init()
    app.run(host='0.0.0.0', port=5000)
