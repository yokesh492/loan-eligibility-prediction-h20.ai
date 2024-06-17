from flask import Flask
from flask_restful import Resource, Api
from flask import request
import h2o
import pandas as pd


class LoanPred(Resource):
    def __init__(self, model):
        self.model = model

    def get(self):
        data = dict()
        data['Current Loan Amount'] = [request.args.get('Current Loan Amount')]
        data['Term'] = [request.args.get('Term')]
        data['Credit Score'] = [request.args.get('Credit Score')]
        data['Years in current job'] = [request.args.get('Years in current job')]
        data['Home Ownership'] = [request.args.get('Home Ownership')]
        data['Annual Income'] = [request.args.get('Annual Income')]
        data['Purpose'] = [request.args.get('Purpose')]
        data['Monthly Debt'] = [request.args.get('Monthly Debt')]
        data['Years of Credit History'] = [request.args.get('Years of Credit History')]
        data['Months since last delinquent'] = [request.args.get('Months since last delinquent')]
        data['Number of Open Accounts'] = [request.args.get('Number of Open Accounts')]
        data['Number of Credit Problems'] = [request.args.get('Number of Credit Problems')]
        data['Current Credit Balance'] = [request.args.get('Current Credit Balance')]
        data['Maximum Open Credit'] = [request.args.get('Maximum Open Credit')]
        data['Bankruptcies'] = [request.args.get('Bankruptcies')]
        data['Tax Liens'] = [request.args.get('Tax Liens')]

        testing = pd.DataFrame(data)
        test = h2o.H2OFrame(testing)
        pred_ans = self.model.predict(test).as_data_frame()

        return{'ans': pred_ans.predict.values[0]}


def init():
    app = Flask(__name__)
    api = Api(app)

    h2o.init()
    model_path = 'output/gradient_boosting_model.hex'
    uploaded_model = h2o.load_model(model_path)

    api.add_resource(LoanPred, '/', resource_class_kwargs={'model': uploaded_model})
    app.run(port=1234)
