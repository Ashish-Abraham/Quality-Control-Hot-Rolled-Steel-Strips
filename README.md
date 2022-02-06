# Quality-Control-Hot-Rolled-Steel-Strips
Automated quality inspection has received increased attention across industries in the wake of Industry 4.0. This project aims to identify metal surface defects such as rolled-in scale, patches, crazing, pitted surface, inclusion and scratches (as depicted in the image below) in Hot-Rolled Steel Strips. The defects are classified into their specific classes via a convolutional neural network (CNN).<br /><br />
![Image](https://github.com/Ashish-Abraham/Quality-Control-Hot-Rolled-Steel-Strips/blob/main/Surface%20Defects.png)
## Neural Network
* Multi-Class Classification Model
* Framework : TensorFlow( Version: 2.6.0 )
* Architecture: Convolutional Neural Network 2D<br/><br/>
![Image](https://github.com/Ashish-Abraham/Quality-Control-Hot-Rolled-Steel-Strips/blob/main/cnn_architecture.png)
* Validation Accuracy : 97.22%<br/><br/>
![Image](https://github.com/Ashish-Abraham/Quality-Control-Hot-Rolled-Steel-Strips/blob/main/model_accuracy.png)
![Image](https://github.com/Ashish-Abraham/Quality-Control-Hot-Rolled-Steel-Strips/blob/main/loss_curve.png)
* Dataset used:<br />NEU Surface Defect Database<br />https://www.kaggle.com/fantacher/neu-metal-surface-defects-data
### Model Result<br/><br/>
![Image](https://github.com/Ashish-Abraham/Quality-Control-Hot-Rolled-Steel-Strips/blob/main/results.png)
## How to use:
* Model has been deployed as api using Flask.
* Host in required web service and use.
* Take a demo at https://huggingface.co/spaces/ashishabraham22/Quality-Ctrl-Steel <br>
It can be deployed in a continuous production line using computer vision libraries. Any help regarding this is welcome.

