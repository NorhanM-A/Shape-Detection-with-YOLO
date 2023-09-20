## Choosing YOLOv8

### YOLOv8 Shape Detection

Welcome to the YOLOv8 Shape Detection project! In this repository, I've implemented a shape detection model using YOLOv8, specifically tailored for shape detection. 

## Why YOLOv8?

When embarking on this project, I carefully considered various object detection models available in the computer vision landscape. YOLOv8 stood out as the ideal choice for several compelling reasons:

1. **State-of-the-Art Performance:** YOLOv8 is a state-of-the-art object detection model renowned for its remarkable accuracy and speed. Its multi-scale architecture allows it to detect objects of varying sizes efficiently.

2. **Versatility:** YOLOv8 is versatile and can be adapted to a wide range of object detection tasks. Whether you're detecting shapes in aerial imagery, identifying objects in real-time video streams, or any other computer vision application, YOLOv8's flexibility is a significant asset.

3. **Community Support:** YOLOv8 benefits from a vibrant open-source community. With an active developer community and ongoing updates, it's easy to find resources, tutorials, and pre-trained models to kickstart your project.

4. **Efficiency:** YOLOv8 has been designed with efficiency in mind. Its ability to achieve impressive detection results on resource-constrained devices makes it a practical choice for real-world applications.

5. **Robustness:** YOLOv8's robustness in handling occlusions, various lighting conditions, and complex scenes ensures reliable object detection performance.






## Setup
**Pip Install Ultralytics and Dependencies:**
- `pip install ultralytics`: This command uses the Python package manager `pip` to install the `ultralytics` library and its required dependencies.
- The `ultralytics` library is a powerful tool for computer vision tasks, including object detection and classification. Installing it via `pip` ensures you have access to its functionalities.

**Check Software and Hardware:**

- After installing `ultralytics`, you can use it to check your software and hardware environment, particularly relevant for deep learning tasks.
- The library provides functions to verify that your system meets the software and hardware requirements for running computer vision models effectively.

**Why This Matters:**

- Verifying software and hardware compatibility is crucial for ensuring that deep learning models, like YOLO, run efficiently on your machine.
- It helps identify any missing dependencies or issues that could affect the performance of computer vision tasks.
- By running this command, you prepare your environment for seamless work with computer vision models and avoid potential problems during model training and inference.

`%pip install ultralytics
import ultralytics
ultralytics.checks()`


## Dataset installation

**Install Roboflow and Download Dataset:**

- `!pip install roboflow`: This command installs the `roboflow` Python package, which is essential for managing computer vision datasets hosted on the Roboflow platform.

- The subsequent Python script utilizes Roboflow for the following tasks:
  - `from roboflow import Roboflow`: It imports the `Roboflow` class from the `roboflow` package, enabling interaction with Roboflow services.
  - `rf = Roboflow(api_key="Q3aYoueljqQ1S5nhgaYY")`: This line establishes a connection to Roboflow using an API key for secure authentication.
  - `project = rf.workspace("hku-uas-deprecated-sobt2").project("standard_object_shape")`: It specifies the target Roboflow workspace and project, specifically referring to the "standard_object_shape" project within the "hku-uas-deprecated-sobt2" workspace.
  - `dataset = project.version(2).download("yolov8")`: This action initiates the download of the dataset labeled "yolov8" from the specified project and version (version 2).

**Dataset Selection:**

- The choice of the dataset from Roboflow was deliberate due to its appropriateness for the project's objectives. This dataset encompasses aerial images captured by drones, featuring various shapes from an overhead perspective. This closely simulates the real-world scenarios encountered in SUAS competitions.




```python
!pip install roboflow
from roboflow import Roboflow
rf = Roboflow(api_key="Q3aYoueljqQ1S5nhgaYY")
project = rf.workspace("hku-uas-deprecated-sobt2").project("standard_object_shape")
dataset = project.version(2).download("yolov8")
```




## Data Exploration:

This code segment is responsible for exploring and analyzing a dataset, specifically designed for shape recognition tasks. The dataset is structured into training, testing, and validation sets, with each set residing in its respective directory.

- `train_dir`, `test_dir`, and `val_dir` variables store the file paths to the training, testing, and validation dataset directories, respectively.

- A `class_mapping` dictionary is defined, associating numeric labels (e.g., '0', '1', ...) with their corresponding class names (e.g., 'circle', 'cross', ...).

- The `explore_dataset` function is defined to perform the following tasks for each dataset (training, testing, validation):
  - Count the number of images and labels (annotations) present in the dataset directories.
  - Calculate the distribution of classes (shapes) within the dataset by parsing the labels and mapping them to class names using the `class_mapping` dictionary.
  - Generate a histogram illustrating the class distribution for visual analysis.

- The code concludes by calling the `explore_dataset` function for each dataset, providing insights into the dataset's composition and class distribution.



```python
import os
import numpy as np
import matplotlib.pyplot as plt

train_dir = '/content/standard_object_shape-2/train'
test_dir = '/content/standard_object_shape-2/test'
val_dir = '/content/standard_object_shape-2/valid'

class_mapping = {
    '0': 'circle',
    '1': 'cross',
    '2': 'heptagon',
    '3': 'hexagon',
    '4': 'octagon',
    '5': 'pentagon',
    '6': 'quarter circle',
    '7': 'rectangle',
    '8': 'semi circle',
    '9': 'square',
    '10': 'star',
    '11': 'trapezoid',
    '12': 'triangle'
}

def explore_dataset(dataset_dir, dataset_name):
    num_images = len(os.listdir(os.path.join(dataset_dir, 'images')))
    num_labels = len(os.listdir(os.path.join(dataset_dir, 'labels')))

    print(f"Exploring {dataset_name} dataset:")
    print(f"Number of images: {num_images}")
    print(f"Number of labels (annotations): {num_labels}")

    class_counts = {}
    for label_file in os.listdir(os.path.join(dataset_dir, 'labels')):
        with open(os.path.join(dataset_dir, 'labels', label_file), 'r') as label_file:
            for line in label_file:
                class_name = line.strip().split()[0]
                class_label = class_mapping.get(class_name, class_name)
                if class_label in class_counts:
                    class_counts[class_label] += 1
                else:
                    class_counts[class_label] = 1

    classes, counts = zip(*class_counts.items())
    plt.figure(figsize=(10, 5))
    plt.bar(classes, counts)
    plt.title(f'Class Distribution in {dataset_name} Dataset')
    plt.xlabel('Classes (Shapes)')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.show()

explore_dataset(train_dir, 'Training')
explore_dataset(test_dir, 'Testing')
explore_dataset(val_dir, 'Validation')

```

![train](https://github.com/NorhanM-A/Shape-Detection-with-YOLO/assets/72838396/c2a09bad-3b78-44eb-9a98-fb555648ff88)
![test](https://github.com/NorhanM-A/Shape-Detection-with-YOLO/assets/72838396/5c5a4210-5482-476a-9c63-1015a14b91d1)
![valid](https://github.com/NorhanM-A/Shape-Detection-with-YOLO/assets/72838396/d6fb1cf3-7a14-497f-be9c-e7b1f37f1ed0)

### Exploring Random Sample Images
```python
import cv2
import random

# Load and display random sample images
sample_images = random.sample(os.listdir(os.path.join(train_dir, 'images')), 5)  # Display 5 random images
plt.figure(figsize=(15, 5))
for i, image_filename in enumerate(sample_images):
    image = cv2.imread(os.path.join(train_dir, 'images', image_filename))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.subplot(1, 5, i+1)
    plt.imshow(image)
    plt.axis('off')
    plt.title(f'Sample Image {i+1}')
plt.show()
```
![sample](https://github.com/NorhanM-A/Shape-Detection-with-YOLO/assets/72838396/7133f2ec-c3d6-4d16-b6c5-8e7b603293e5)


### Initializing the YOLOv8 Model

In this section, we demonstrate how to initialize the YOLOv8 model using the Ultralytics library. The YOLO (You Only Look Once) model is a popular deep learning model for object detection.



```python
from ultralytics import YOLO

# Initialize the YOLOv8 model with pre-trained weights
model = YOLO("yolov8n.pt")
```


### Training the YOLOv8 Model

In this section, we demonstrate how to train the YOLOv8 model using the Ultralytics library. Training a model is a crucial step in building an accurate object detection system tailored to your specific requirements.



```python
# Train the YOLOv8 model using the provided data configuration and for a specified number of epochs
model.train(data="/content/standard_object_shape-2/data.yaml", epochs=3)
```

### Evaluating the YOLOv8 Model

### Confusion Matrix

A confusion matrix is a powerful tool for assessing the performance of an object detection model. It provides a detailed breakdown of the model's predictions compared to the ground truth labels in the test dataset. The confusion matrix is divided into four quadrants:

- True Positives (TP): Correctly predicted objects.
- True Negatives (TN): Correctly predicted background (no objects).
- False Positives (FP): Predicted objects where there are none (false alarms).
- False Negatives (FN): Missed actual objects.

The confusion matrix visually represents how well the model identifies and localizes objects.

![confusionmatrix](https://github.com/NorhanM-A/Shape-Detection-with-YOLO/assets/72838396/8912aed2-14df-4b2f-bc4f-a9bdcc299397)

### Precision-Recall Curve

The precision-recall curve is a valuable tool for evaluating the model's trade-off between precision (the fraction of true positive predictions among all positive predictions) and recall (the fraction of true positives identified correctly). It helps determine the optimal threshold for predictions based on the model's confidence scores.
![pr](https://github.com/NorhanM-A/Shape-Detection-with-YOLO/assets/72838396/7afa779c-b2ff-41fb-9185-071dbe4c6fa2)
### Confidence Curve

The confidence curve provides insights into the model's confidence levels for its predictions. By plotting confidence scores against the number of predictions, it reveals how well the model distinguishes between correct and incorrect predictions.

![confidencecurve](https://github.com/NorhanM-A/Shape-Detection-with-YOLO/assets/72838396/33420aad-a7c9-416a-a428-1a6e576896d6)
### Results Plot


![results](https://github.com/NorhanM-A/Shape-Detection-with-YOLO/assets/72838396/4c386318-81cc-4ef5-86ab-5b8f60e9e609)


### Initializing the Object Detection Model

The following code initializes an object detection model based on the YOLO architecture and loads pre-trained weights for inference:

```python
infer = YOLO("/content/runs/detect/train/weights/best.pt")
```
- YOLO: This code utilizes the YOLO class from an object detection framework.
- "/content/runs/detect/train/weights/best.pt": This path specifies the location of the pre-trained weights file that the model will use for inference. Pre-trained weights contain learned parameters from a model trained on a large dataset, enabling it to make predictions without further training.


### Performing Object Detection on Test Images

The following code performs object detection on a set of test images using the previously initialized object detection model:

```python
infer.predict("/content/standard_object_shape-2/test/images", save=True, save_txt=True)
```





## Predicted Images

In the following section, we explore visual results of object detection on the test images using the YOLOv8 model. These images have been annotated with bounding boxes, highlighting the detected shapes. These visualizations provide insights into the model's performance on real-world data.



The following images display the model's predictions:

![p10](https://github.com/NorhanM-A/Shape-Detection-with-YOLO/assets/72838396/66babfea-c834-4a5d-9cb0-4a111c705777)
![p1](https://github.com/NorhanM-A/Shape-Detection-with-YOLO/assets/72838396/0e613a53-7648-4c48-9c49-a93ec9881aa9)
![p2](https://github.com/NorhanM-A/Shape-Detection-with-YOLO/assets/72838396/2b60f08e-6bc1-4448-8b2a-dc1e7a58de6e)
![p3](https://github.com/NorhanM-A/Shape-Detection-with-YOLO/assets/72838396/9b417445-5511-44a5-b98a-840f8376179b)
![p4](https://github.com/NorhanM-A/Shape-Detection-with-YOLO/assets/72838396/28451b96-5a5c-4cc1-bb4b-2497c7a35b58)
![p5](https://github.com/NorhanM-A/Shape-Detection-with-YOLO/assets/72838396/24a9d977-05ce-4c39-8489-a80e3e8fec68)
![p6](https://github.com/NorhanM-A/Shape-Detection-with-YOLO/assets/72838396/236eff8c-2953-4ded-815b-f0709da25e27)
![p7](https://github.com/NorhanM-A/Shape-Detection-with-YOLO/assets/72838396/9bbc28bb-3f2e-4ca6-b7c8-6ce27cd8001f)
![p8](https://github.com/NorhanM-A/Shape-Detection-with-YOLO/assets/72838396/5fbc78f0-1356-4278-a57c-0b60a9e3ce51)
![p9](https://github.com/NorhanM-A/Shape-Detection-with-YOLO/assets/72838396/04375db7-623f-4cd3-babb-72c6df2f877c)


