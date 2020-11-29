import json
import os
import time
import joblib
import pandas
from azureml.core.model import Model

# Called when the service is loaded
def init():
    global model
    # Get the path to the deployed model file and load it
    model_path = Model.get_model_path('TitanicAutoML')
    model = joblib.load(model_path)
    
    #Print statement for appinsights custom traces:
    print ("model initialized" + time.strftime("%H:%M:%S"))

# Called when a request is received
def run(mini_batch):
    # This runs for each batch
    resultList = []
        
    # process each file in the batch
    for f in mini_batch:
        # Read the comma-delimited data
        data = pandas.read_csv(f)
        predictions = model.predict(data)

        # Append prediction to results
        resultList.append("{}: {}".format(os.path.basename(f), predictions))
    return resultList
