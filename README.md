## penTrackingModel [TensorFlow 2.x]
This is a TensorFlow based custom model [developed by Pankaj Kumar], and is on development phase.

### Navneet Enterprise   
Note: There is a bug in this model, that is it predicts 10 objects whether there are 10 objects or less. This bug can be fixed by re-training model.

# MAIN KEYPOINTS PEN DETECTION WEB APPLICATION

### 1. Phase_1 [Data Collections & Annotations]
#### >> Data Collections, such as collecting training datasets
#### >> Downloading Annotation Tools, for e.g., labelImg 
#### >> Making Annotations, using labelImg tool and saving all training data in training folder

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
    

print("Model training is completed!") ```



