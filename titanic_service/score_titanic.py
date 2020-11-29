import json
import joblib
import pandas
import time
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
def run(raw_data):
    try:
        # Get the input data as a pandas df
        data = json.loads(raw_data)['data']
        data = pandas.read_json(data, orient='records')

        # Get a prediction from the model
        predictions = model.predict(data)
        # Get the corresponding classname for each prediction (0 or 1)
        classnames = ['not-survived', 'survived']
        predicted_classes = []
        for prediction in predictions:
            predicted_classes.append(classnames[prediction])
        
        # Log the input and output data to appinsights:
        info = {
            "input": raw_data,
            "output": predictions.tolist()
            }
        # Return the predictions as JSON
        return json.dumps(predicted_classes)
    except Exception as e:
        error = str(e)
        print (error + time.strftime("%H:%M:%S"))
        return error
