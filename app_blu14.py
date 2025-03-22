import os
import json
import pickle
import joblib
import pandas as pd
from flask import Flask, jsonify, request
from peewee import (
    Model, IntegerField, FloatField,
    TextField, IntegrityError
)
from playhouse.shortcuts import model_to_dict
from playhouse.db_url import connect


########################################
# Begin database stuff

# The connect function checks if there is a DATABASE_URL env var.
# If it exists, it uses it to connect to a remote postgres db.
# Otherwise, it connects to a local sqlite db stored in predictions.db.
DB = connect(os.environ.get('DATABASE_URL') or 'sqlite:///predictions.db')

class Prediction(Model):
    observation_id = IntegerField(unique=True)
    observation = TextField()
    proba = FloatField()
    true_class = IntegerField(null=True)

    class Meta:
        database = DB


DB.create_tables([Prediction], safe=True)

# End database stuff
########################################

########################################
# Unpickle the previously-trained model


with open('columns.json') as fh:
    columns = json.load(fh)

pipeline = joblib.load('pipeline.pickle')

with open('dtypes.pickle', 'rb') as fh:
    dtypes = pickle.load(fh)


# End model un-pickling
########################################


########################################
# Begin webserver stuff

app = Flask(__name__)


from flask import jsonify, request
import pandas as pd

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint to handle prediction requests.
    Receives a JSON payload with an 'id' and 'observation', processes the observation,
    and returns the predicted probability.
    """
    # Deserialize the JSON payload from the request
    obs_dict = request.get_json()

    # Validate that the JSON contains the required fields: 'id' and 'observation'
    if not obs_dict or 'id' not in obs_dict or 'observation' not in obs_dict:
        return jsonify({"error": "Invalid request: 'id' and 'observation' are required"}), 400

    # Extract the 'id' and 'observation' from the JSON payload
    _id = obs_dict['id']
    observation = obs_dict['observation']

    # Validate that the observation contains all required fields
    required_fields = ['age', 'education', 'hours-per-week', 'native-country']
    if not all(field in observation for field in required_fields):
        return jsonify({"error": "Observation is invalid! Missing required fields."}), 400

    try:
        # Convert the observation into a DataFrame using the predefined columns
        # and apply the specified data types (dtypes)
        obs = pd.DataFrame([observation], columns=columns).astype(dtypes)
    except Exception as e:
        # Handle errors during DataFrame creation or type casting
        return jsonify({"error": f"Observation is invalid! {str(e)}"}), 400

    try:
        # Use the pipeline to predict the probability of the positive class
        proba = pipeline.predict_proba(obs)[0, 1]
        response = {'proba': proba}

        # Create a new Prediction record to save in the database
        p = Prediction(
            observation_id=_id,
            proba=proba,
            observation=request.data
        )
        try:
            # Save the prediction to the database
            p.save()
        except IntegrityError:
            # Handle the case where the observation ID already exists in the database
            error_msg = f"Observation ID {_id} already exists"
            response['error'] = error_msg
            print(error_msg)
            DB.rollback()  # Rollback the transaction to avoid partial updates
    except Exception as e:
        # Handle errors during prediction or database operations
        return jsonify({"error": f"Failed to make prediction: {str(e)}"}), 500

    # Return the response as JSON
    return jsonify(response)


@app.route('/update', methods=['POST'])
def update():
    """
    Endpoint to update the true class of an existing observation.
    Receives a JSON payload with 'id' and 'true_class', updates the corresponding record in the database,
    and returns the updated observation.
    If the observation ID does not exist, returns an appropriate error message.
    """
    # Deserialize the JSON payload from the request
    obs = request.get_json()

    # Validate that the JSON contains the required fields: 'id' and 'true_class'
    if not obs or 'id' not in obs or 'true_class' not in obs:
        return jsonify({"error": "Invalid request: 'id' and 'true_class' are required"}), 400

    try:
        # Retrieve the prediction record from the database using the provided 'id'
        p = Prediction.get(Prediction.observation_id == obs['id'])

        # Update the 'true_class' field with the value from the request
        p.true_class = obs['true_class']

        # Save the updated record to the database
        p.save()

        # Return the updated observation as a JSON response
        return jsonify(model_to_dict(p))
    except Prediction.DoesNotExist:
        # Handle the case where the observation ID does not exist in the database
        error_msg = f"Observation ID {obs['id']} does not exist"
        return jsonify({'error': error_msg}), 404
    except Exception as e:
        # Handle any other unexpected errors
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


@app.route('/list-db-contents')
def list_db_contents():
    return jsonify([
        model_to_dict(obs) for obs in Prediction.select()
    ])


# End webserver stuff
########################################

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, port=5000)
