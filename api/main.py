from flask import Flask, jsonify, request
from flask_cors import CORS
from model_files.ml_predict import predict_defect
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
import base64
import os


app = Flask("quality_control_api")
CORS(app)


@app.route('/', methods = ['GET' , 'POST'])
def predict():
    # key_dict = request.get_json()
    # image = key_dict["image"]
    # imgdata = base64.b64decode(image)
    # model_path=os.path.join('model_files','Metal_surface_defect_detector (1).h5')
    # model = load_model(model_path)
    # defect = predict_defect(model, imgdata)
    # response = {
    #     "result": defect,
    # }
    file = request.files['file']
    basepath = os.path.dirname(__file__)
    file_path = os.path.join(
        basepath, 'uploads', secure_filename(file.filename))
    file.save(file_path)
    imgdata = base64.b64decode(image)
    model_path=os.path.join('model_files','Metal_surface_defect_detector (1).h5')
    model = load_model(model_path)
    defect = predict_defect(model, imgdata)
    response = {
        "result": defect,
    }
    response = jsonify(response)
    return response


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)