from flask import Flask, jsonify, request
from flask_cors import CORS
from model_files.ml_predict import predict_defect
from tensorlow.keras.models import load_model
import base64
from decouple import config


app = Flask("quality_control_api")
CORS(app)


@app.route('/', methods=['POST'])
def predict():
    key_dict = request.get_json()
    image = key_dict["image"]
    imgdata = base64.b64decode(image)
    model = tf.keras.models.load_model('/content/Metal_surface_defect_detector.h5')
    defect = predict_defect(model, imgdata)
    response = {
        "result": defect,
    }
    response = jsonify(response)
    return response


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)