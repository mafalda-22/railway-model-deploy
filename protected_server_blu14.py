import os
import json
import pickle
import joblib
import pandas as pd
from flask import Flask, jsonify, request
from peewee import (
    SqliteDatabase, PostgresqlDatabase, Model, IntegerField,
    FloatField, TextField, IntegrityError
)
from playhouse.shortcuts import model_to_dict
from playhouse.db_url import connect

########################################
# Begin database stuff

DB = connect(os.environ.get('DATABASE_URL') or 'sqlite:///predictions.db')


class Prediction(Model):
    observation_id = TextField()
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


with open(os.path.join('columns.json')) as fh:
    columns = json.load(fh)


with open(os.path.join('pipeline.pickle'), 'rb') as fh:
    pipeline = joblib.load(fh)


with open(os.path.join('dtypes.pickle'), 'rb') as fh:
    dtypes = pickle.load(fh)


# End model un-pickling
########################################

########################################
# Input validation functions

def attempt_predict(request, columns, dtypes, pipeline):
    # Validate observation_id
    if "observation_id" not in request:
        return {
            "observation_id": observation_id,
            "error": "Missing observation_id"
        }, 200
    
    observation_id = request["observation_id"]

    # Check data exists
    if "data" not in request:
        return {
            "observation_id": observation_id,
            "error": "Field 'data' is required"
        }, 200

    data = request["data"]
    
    # Validate age
    if "age" not in data:
        return {
            "observation_id": observation_id,
            "error": "Field 'age' is required"
        }
        
    if not isinstance(data["age"], int):
        return {
            "observation_id": observation_id,
            "error": f"Field 'age' must be an integer (got {data['age']})"
        }, 200
        
    if data["age"] < 0 or data["age"] > 120:
        return {
            "observation_id": observation_id,
            "error": f"Invalid value for 'age': {data['age']}. Must be between 0 and 120"
        }, 200

    CATEGORICAL_RULES = {
        "workclass": ["State-gov", "Self-emp-not-inc", "Private", "Federal-gov", 
                     "Local-gov", "Self-emp-inc", "Without-pay", "Never-worked", "?"],
        "education": ["Bachelors", "HS-grad", "11th", "Masters", "9th", 
                     "Some-college", "Assoc-acdm", "Assoc-voc", "7th-8th", 
                     "Doctorate", "Prof-school", "5th-6th", "10th", "1st-4th", 
                     "Preschool", "12th"],
        "marital-status": ["Never-married", "Married-civ-spouse", "Divorced", 
                          "Married-spouse-absent", "Separated", "Married-AF-spouse", 
                          "Widowed"],
        "race": ["White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"],
        "sex": ["Male", "Female"]
    }

    for field, allowed_values in CATEGORICAL_RULES.items():
        if field in request["data"]:
            value = request["data"][field]
            if value not in allowed_values:
                return {
                    "observation_id": observation_id,
                    "error": f"Invalid value for '{field}': '{value}'. Allowed: {allowed_values}"
                }



    try:
        # Check columns
        request_columns = set(request["data"].keys())
        expected_columns = set(columns)
        missing_columns = expected_columns - request_columns
        extra_columns = request_columns - expected_columns


        if missing_columns or extra_columns:
            error_message = []
            if missing_columns:
                error_message.append(f"Missing columns: {', '.join(missing_columns)}")
            if extra_columns:
                error_message.append(f"Extra columns: {', '.join(extra_columns)}")
            
            return {
                "observation_id": observation_id,
                "error": " | ".join(error_message)
            }, 200
        
        # Convert input data into a DataFrame with the correct columns and types
        obs = pd.DataFrame([request["data"]], columns=columns)
        obs = obs.astype(dtypes)
        
        # Generate the prediction probabilities and the prediction itself
        probabilities = pipeline.predict_proba(obs)[0]
        prediction = pipeline.predict(obs)[0]
        
        # Format the response
        response = {
            "observation_id": observation_id,
            "prediction": bool(prediction),  # Convert prediction to a boolean (True/False)
            "probability": max(probabilities)  # Extract the highest probability (for the predicted class)
        }
        return response
    
    except Exception as e:
        # Handle errors by returning an error message
        return {
            "observation_id": observation_id,
            "error": f"Prediction error: {str(e)}"
        }



def check_valid_column(observation):
    
    valid_columns = {
      "age",
      "workclass",
      "education", 
      "marital-status", 
      "race",
      "sex",
      "capital-gain",
      "capital-loss",
      "hours-per-week"
    }
    
    keys = set(observation.keys())
    
    if len(valid_columns - keys) > 0: 
        missing = valid_columns - keys
        error = "Missing columns: {}".format(missing)
        return False, error
    
    if len(keys - valid_columns) > 0: 
        extra = keys - valid_columns
        error = "Unrecognized columns provided: {}".format(extra)
        return False, error    

    return True, ""



def check_categorical_values(observation):
    
    valid_category_map = {
        "workclass": ["?", "State-gov", "Self-emp-not-inc", "Private", "Federal-gov", "Local-gov", "Self-emp-inc", "Without-pay", "Never-worked"],
        "education": ["Bachelors", "HS-grad", "11th", "Masters", "9th", "Some-college", "Assoc-acdm", "Assoc-voc", "7th-8th", "Doctorate", "Prof-school", "5th-6th", "10th", "1st-4th", "Preschool", "12th"],
        "marital-status": ["Never-married", "Married-civ-spouse", "Divorced", "Married-spouse-absent", "Separated", "Married-AF-spouse", "Widowed"],
        "race": ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo','Other'],
        "sex": ["Male", "Female"],
    }
    
    for key, valid_categories in valid_category_map.items():
        if key in observation:
            value = observation[key]
            if value not in valid_categories:
                error = "Invalid value provided for {}: {}. Allowed values are: {}".format(
                    key, value, ",".join(["'{}'".format(v) for v in valid_categories]))
                return False, error
        else:
            error = "Categorical field {} missing"
            return False, error

    return True, ""



def validate_age(observation):
    age = observation.get("age")
    if age is None: 
        error = "Field 'age' is required"
        return False, error

    if not isinstance(age, int):
        error = f"Field 'age' must be an integer (got {age})"
        return False, error
    
    if age < 0 or age > 120:
        error = f"Invalid value for 'age': {age}. Must be between 0 and 120"
        return False, error

    return True, ""

def validate_hours_per_week(observation):
    
    hours_per_week = observation.get("hours-per-week")
        
    if not hours_per_week: 
        error = "Field `hours_per_week` missing"
        return False, error

    if not isinstance(hours_per_week, int):
        error = "Field `hours_per_week` is not an integer"
        return False, error
    
    if hours_per_week < 0 or hours_per_week > 168:
        error = "Field `hours_per_week` is not between 0 and 168"
        return False, error

    return True, ""


def validate_capital_gain(observation):
    
    capital_gain = observation.get("capital-gain")
        
    if not capital_gain: 
        error = "Field `capital_gain` missing"
        return False, error

    if not isinstance(capital_gain, int):
        error = "Field `capital_gain` is not an integer"
        return False, error
    
    if capital_gain < 0:
        error = "Field `capital_gain` must be greater than 0"
        return False, error

    return True, ""

def validate_capital_loss(observation):
    
    capital_loss = observation.get("capital-loss")
        
    if not capital_loss: 
        error = "Field `capital_loss` missing"
        return False, error

    if not isinstance(capital_loss, int):
        error = "Field `capital_loss` is not an integer"
        return False, error
    
    if capital_loss < 0:
        error = "Field `capital_loss` must be greater than 0"
        return False, error

    return True, ""


def validate_sex(observation):
    
    sex = observation.get("sex")
        
    if not sex: 
        error = "Field `sex` missing"
        return False, error

    if not isinstance(sex, str):
        error = "Field `sex` is not an string"
        return False, error
    
    if sex not in ['Female', 'Male']:
        error = "Field `sex` must be male or female"
        return False, error

    return True, ""


def validate_race(observation):
    
    race = observation.get("race")
        
    if not race: 
        error = "Field `race` missing"
        return False, error

    if not isinstance(race, str):
        error = "Field `race` is not an string"
        return False, error
    
    if race not in ["White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"]:
        error = "Field `race` is not correct"
        return False, error

    return True, ""

def validate_workclass(observation):
    
    workclass = observation.get("workclass")
        
    if not workclass: 
        error = "Field `workclass` missing"
        return False, error

    if not isinstance(workclass, str):
        error = "Field `workclass` is not an string"
        return False, error
    
    if workclass not in ['State-gov', 'Self-emp-not-inc', 'Private', 'Federal-gov', 'Local-gov', '?','Self-emp-inc', 'Without-pay', 'Never-worked']:
        error = "Field `workclass` is not correct"
        return False, error

    return True, ""

def validate_education(observation):
    
    education = observation.get("education")
        
    if not education: 
        error = "Field `education` missing"
        return False, error

    if not isinstance(education, str):
        error = "Field `education` is not an string"
        return False, error
    
    if education not in ['Bachelors', 'HS-grad', '11th', 'Masters', '9th', 'Some-college', 'Assoc-acdm','Assoc-voc', '7th-8th', 'Doctorate', 'Prof-school', '5th-6th', '10th', '1st-4th','Preschool', '12th']:
        error = "Field `education` is not correct"
        return False, error

    return True, ""

def validate_marital_status(observation):
    
    marital_status = observation.get("marital_status")
        
    if not marital_status: 
        error = "Field `marital_status` missing"
        return False, error

    if not isinstance(marital_status, str):
        error = "Field `marital_status` is not an string"
        return False, error
    
    if marital_status not in ['Never-married', 'Married-civ-spouse', 'Divorced', 'Married-spouse-absent','Separated', 'Married-AF-spouse', 'Widowed']:
        error = "Field `marital_status` is not correct"
        return False, error

    return True, ""



# End input validation functions
########################################

########################################
# Begin webserver stuff

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    if 'data' not in data:
        return jsonify({"error": "Missing 'data' in request"}), 200
    
    observation = data["data"]
    
    # Ensure 'observation_id' exists
    if "observation_id" not in data:
        return jsonify({"error": "Missing 'observation_id' in request"}), 200
    
    # Validation checks
    columns_ok, error = check_valid_column(observation)
    if not columns_ok:
        return jsonify({'error': error}), 200


    age_ok, error = validate_age(observation)
    if not age_ok:
        return jsonify({'error': error}), 200


    # Prediction
    result = attempt_predict(data, columns, dtypes, pipeline)
    print(result)

    if "error" in result:
        return jsonify(result), 200

    # Store prediction in the database
    try:
        with DB.atomic():
            Prediction.create(
                observation_id=result['observation_id'],
                proba=float(result['probability']),
                observation=json.dumps(observation)  # Store data as JSON
            )
    except IntegrityError:
        result['error'] = f"Observation ID {result['observation_id']} already exists"
        return jsonify(result), 200

    return jsonify(result), 200
    

@app.route('/update', methods=['POST'])
def update():
    obs = request.get_json()
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
    
if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, port=5000)
