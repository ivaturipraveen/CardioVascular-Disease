from flask import Flask, request, jsonify, render_template
import joblib
from flask_cors import CORS



app = Flask(__name__)
CORS(app)

# Load the trained model
model = joblib.load('best_model_pelican.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    data = request.get_json()
    print(data)
    
    map = (float(data['ap_lo']) + (float(data['ap_hi']) - float(data['ap_lo'])) / 3)
    X = [[float(data['age']), float(data['gender']), float(data['height']), float(data['weight']), float(data['ap_hi']), 
          float(data['ap_lo']), float(data['cholestrol']), float(data['gluc']), float(data['smoke']), float(data['alco']), 
          float(data['active']), float(data['bmi']), float(data['bmi_category']), float(data['pulse_pressure']), map]]
    y_pred = model.predict(X)
    print(y_pred)
    
    if y_pred[0] == 0:
        return jsonify({"message": "Predicted class: No Cardiovascular disease"})
    else:
        return jsonify({"message": "Predicted class: Cardiovascular disease"})

if __name__ == '__main__':
    app.run(debug=True)
