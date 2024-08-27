### DownLoad Project
Link: https://drive.google.com/file/d/1THdOTKhYIw6llJvYN1L4VqKc7oFCjqq5/view?usp=drive_link
## penTrackingModel [TensorFlow 2.x]
This is a TensorFlow based custom model [developed by Pankaj Kumar], and is on development phase.

Note: There is a bug in this model, that is it predicts 10 objects whether there are 10 objects or less. This bug can be fixed by re-training model.

# MAIN KEYPOINTS PEN DETECTION WEB APPLICATION

### 1. Phase_1 [Data Collections & Annotations]
#### >> Data Collections, such as collecting training datasets
#### >> Downloading Annotation Tools, for e.g., labelImg 
#### >> Making Annotations, using labelImg tool and saving all training data in training folder
![Screenshot 2024-08-25 075350](https://github.com/user-attachments/assets/ce1c6937-2050-4732-a3ff-33e50c9e1d47)
![image](https://github.com/user-attachments/assets/a7557600-549d-4b70-a848-ceb6ca063a32)


### 2. Phase_2 [Training, Building & Saving penDetectionModel]
#### >> Training model using traing scripts
```# Load data

images, labels = load_data(image_dir)

# Preprocess data
images = images/255.0 #Normalize images

# Flatten the labels
labels = labels.reshape((labels.shape[0], -1))

# Define a Sequential Model
model = tf.keras.Sequential([tf.keras.layers.Input(shape=(224,224,3)),
                             tf.keras.layers.Conv2D(32,(3,3), activation='relu'),
                             tf.keras.layers.MaxPooling2D((2,2)),
                             tf.keras.layers.Flatten(),
                             tf.keras.layers.Dense(128,activation='relu'),
                             tf.keras.layers.Dense(max_boxes*4)]) #Assuming 4 coordinates for bounding box
...
...
...
# Train the model with status updates

model.fit(images, labels, epochs=500, verbose=1, callbacks=[MetricsCallback()])
    

print("Model training is completed!")
```
![image](https://github.com/user-attachments/assets/493b765a-78ea-4446-ada8-b8495c84af16)


## 3. Phase_3 [TESTING MODEL]
### >>> After Saving model, Testing & Model Evaluation is done
### >>> Testing using particular image file

```
# Required libraries to test an image file
import tensorflow as tf
import cv2
import numpy as np
from matplotlib import pyplot as plt
```

```
# Prediction function

def predict_image(image_path):
    image = cv2.imread(image_path)
    image_resized = (cv2.resize(image, (224,224)))/255.0
    image_expanded = np.expand_dims(image_resized, axis=0)

    predictions = model.predict(image_expanded)
    # predictions = predictions.reshape((max_boxes, 4))
    predictions = predictions.reshape((10, 4))

    return predictions
```

```
image_path = 'E:/AI_PROJECTS/ObjectDetection/PenDetection/test/41dvxYtTN7L._AC_UF1000,1000_QL80_FMwebp_.webp'
predictions = predict_image(image_path)
print(predictions)

OUTPUT:
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 226ms/step
[[ 3.61751343e+02  4.37374687e+00  4.70183075e+02  9.96324158e+02]
 [ 2.38657303e+02  4.71720123e+00  3.53916443e+02  1.00044708e+03]
 [ 1.19947411e+02  4.28417778e+00  2.32872391e+02  1.00281757e+03]
 [ 2.80680728e+00  3.24937606e+00  1.18191124e+02  1.00761493e+03]
 [ 9.47685242e+00 -2.04298973e+00  5.47523355e+00  2.49926662e+00]
 [ 5.50797319e+00 -2.12698174e+00  1.84664619e+00  3.75212121e+00]
 [ 4.71004391e+00  1.09558538e-01  7.83584213e+00  7.10538006e+00]
 [ 4.51294899e+00 -1.90369725e+00  8.51321793e+00  7.55034781e+00]
 [ 6.96509933e+00 -1.31059539e+00  4.06070614e+00  9.07032681e+00]
 [ 5.69514036e+00  7.32337117e-01  3.27979445e+00  1.01378269e+01]]
...
...
...
check out the penTrackingScript.ipynb for the complete code
```

### OUTPUT
![image](https://github.com/user-attachments/assets/daec24fd-ce12-4c99-ac0f-5df1990257e8)
![Screenshot 2024-08-26 072339](https://github.com/user-attachments/assets/a3da49e1-9948-47a5-9568-fd4066a3e529)

### 4. Phase_4 [Model Deployement, Building WebApplication]
#### >>> Using Python Flask Framework, the web application is built. However, project development is in process.
#### >>> On testing local server, a simple webpage:
![Screenshot 2024-08-27 123230](https://github.com/user-attachments/assets/35ba244d-d8fd-42ce-b738-8d86ea094f97)











