# Shape-Detection-with-YOLO
Shape Detection with YOLO: A computer vision project that employs YOLO, a state-of-the-art deep learning framework, to accurately identify and locate various geometric shapes in images. Perfect for applications such as drone-based surveillance and object recognition.



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

![https://github.com/NorhanM-A/Shape-Detection-with-YOLO.git]![train](https://github.com/NorhanM-A/Shape-Detection-with-YOLO/assets/72838396/c2a09bad-3b78-44eb-9a98-fb555648ff88)
![https://github.com/NorhanM-A/Shape-Detection-with-YOLO.git]![test](https://github.com/NorhanM-A/Shape-Detection-with-YOLO/assets/72838396/5c5a4210-5482-476a-9c63-1015a14b91d1)
![https://github.com/NorhanM-A/Shape-Detection-with-YOLO.git]![valid](https://github.com/NorhanM-A/Shape-Detection-with-YOLO/assets/72838396/d6fb1cf3-7a14-497f-be9c-e7b1f37f1ed0)

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
![https://github.com/NorhanM-A/Shape-Detection-with-YOLO.git]![sample](https://github.com/NorhanM-A/Shape-Detection-with-YOLO/assets/72838396/7133f2ec-c3d6-4d16-b6c5-8e7b603293e5)


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



### Precision-Recall Curve

The precision-recall curve is a valuable tool for evaluating the model's trade-off between precision (the fraction of true positive predictions among all positive predictions) and recall (the fraction of true positives identified correctly). It helps determine the optimal threshold for predictions based on the model's confidence scores.

### Confidence Curve

The confidence curve provides insights into the model's confidence levels for its predictions. By plotting confidence scores against the number of predictions, it reveals how well the model distinguishes between correct and incorrect predictions.


### Results Plot

The results plot summarizes the overall performance of the YOLOv8 model on the test dataset. It typically includes metrics such as precision, recall, F1-score, and average precision (AP). This plot offers a comprehensive overview of the model's object detection performance.










