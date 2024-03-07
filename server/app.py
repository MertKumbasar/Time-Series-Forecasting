# we need to install some python packages
# pip install numpy pandas matplotlib statsmodels pmdarima flask flask-cors openpyxl


from flask import Flask, request, jsonify
from flask_cors import CORS
from data import perform_analysis


app = Flask(__name__)
CORS(app, origins="frontend url")  



@app.route("/results", methods=["POST"])
def results():
    
    # catch the request parameters
    request_data = request.get_json()
    device_id = request_data.get("deviceId")
    dataset_choice = request_data.get("dataFormat")

    # run the analysis with those parameters
    results = perform_analysis(device_id, dataset_choice)

    
    return jsonify(results)

if __name__ == "__main__":
    app.run(debug=True, port=3000)