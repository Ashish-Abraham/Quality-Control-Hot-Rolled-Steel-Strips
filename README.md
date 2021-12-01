# Quality-Control-Hot-Rolled-Steel-Strips
Automated quality inspection has received increased attention across industries in the wake of Industry 4.0. This project aims to identify metal surface defects such as rolled-in scale, patches, crazing, pitted surface, inclusion and scratches (as depicted in the image below) in Hot-Rolled Steel Strips. The defects are classified into their specific classes via a convolutional neural network (CNN).<br /><br />
![Image](https://github.com/Ashish-Abraham/Quality-Control-Hot-Rolled-Steel-Strips/blob/main/Surface%20Defects.png)
## Neural Network
* Framework : TensorFlow( Version: 2.6.0 )
* Architecture: Convolutional Neural Network 2D<br/><br/>
![Image](https://github.com/Ashish-Abraham/Quality-Control-Hot-Rolled-Steel-Strips/blob/main/cnn_architecture.png)
* Validation Accuracy : 97.22%<br/><br/>
![Image](https://github.com/Ashish-Abraham/Quality-Control-Hot-Rolled-Steel-Strips/blob/main/model_accuracy.png)
![Image](https://github.com/Ashish-Abraham/Quality-Control-Hot-Rolled-Steel-Strips/blob/main/loss_curve.png)
* Dataset used:<br />NEU Surface Defect Database<br />https://www.kaggle.com/fantacher/neu-metal-surface-defects-data
### How to train
Upload the python notebook in the folder **'model'** to Google Colab and run each cell for training the model.
## How to use:
* Download ML model .h5 file from folder **'model'**
* model = load_model('model.h5')<br/><br/>
It can be deployed in a continuous production line using computer vision libraries. Any opinion or help on this is welcome.

