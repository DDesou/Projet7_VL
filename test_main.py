from fastapi.testclient import TestClient
from main import app
import json
import requests
import warnings
from sklearn.exceptions import ConvergenceWarning #, InconsistentVersionWarning
import shap
shap.initjs()

# Ignore inconsistent version warnings during testing
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="sklearn")
#warnings.filterwarnings("ignore", category=InconsistentVersionWarning, module="sklearn")


client = TestClient(app)

def read_json_file(file_path):
    with open(file_path, 'r') as json_file:
        return json.load(json_file)
    
def raise_custom_warning():
    warning_message = "This is a custom warning message for testing."
    warnings.warn(warning_message, category=DeprecationWarning)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"Hello my API": "It works!!!"}

def test_get_prediction():
    # Assuming your prediction endpoint is '/prediction/'
    data1 = read_json_file("./tests/input1.json")
    data0 = read_json_file("./tests/input0.json")

    # Raise a custom warning
    raise_custom_warning()

    # Replace with actual data for prediction
    base_url = "http://127.0.0.1:8000"
    endpoint = "/prediction/"
    url = f"{base_url}{endpoint}"

    try:
        # Send request for usecase1
        response = requests.get(url, data=data1, timeout=80)
        response.raise_for_status()  # Raise an HTTPError for bad responses
        proba_default = eval(response.content)["probability"]
        result = round(proba_default * 100, 1)

        # Send request for usecase0
        response0 = requests.get(url, data=data0, timeout=80)
        response0.raise_for_status()
        proba_default0 = eval(response0.content)["probability"]
        result0 = round(proba_default0 * 100, 1)

        # Assertions
        assert response.status_code == 200
        assert "probability" in response.json()
        assert result >= 40  # threshold 40% for usecase1
        assert result0 < 40  # threshold 40% for usecase0

    except requests.exceptions.RequestException as e:
        # Handle exceptions, print an error message, or fail the test
        print(f"Request failed: {e}")
        assert False  # Fail the test