<div align="center">
  <h1>Proposed Title: Detection Of Hand Signals for Traffic Control Using Deep Learning MoveNet Model</h1>
</div><br>

---

### Development Model:

Different Traffic Gesture - Pose detection

## Group Members

| Reg.No    | Name                      |
| --------- | ------------------------- |
| 20BCI0090 | Vandit Gabani             |
| 20BCI0128 | Aditi Nitin Tagalpallewar |
| 20BCI0176 | Yash Bobde                |
| 20BCI0271 | Harsh Rajpal              |
| 20BCE2759 | Payal Maheshwari          |
| 20BCI0138 | Bagade Shaunak Rahul      |
| 20BCI0169 | Konark Patel              |
| 20BCI0159 | Nikhil Harshwardhan       |

---

## Roles - Responsibilities :

1.Collecting Data and Extracting Landmarks points and finding Angles for Different pose Using Mediapipe and Movenet Model - <br/>
<b>Konark Patel(20BCI0169) , Nikhil(20BCI0159)</b><br/><br/>

2.Build and train a pose classification model that takes landmark coordinates from a CSV file as input and outputs predicted labels-<br/>
<b>Vandit Gabani(20BCI0090) ,Aditi(20BCI0128) , Shaunak(20BCI0138)</b><br/><br/>

3.Convert the pose classification model to TFLite and Testing Using Movenet Model-<br/>
<b>Harsh(20BCI0271) , Payal(20BCE2759) , Yash(20BCI0176)</b><br/><br/>

---

## Abstract
<font size="10">
<div align="justify">
<ul>
  <li>Gesture recognition is one of the most difficult challenges in computer vision. While recognising traffic police hand signals, one must take into account the speed and dependability of the instructing signal. It is significantly easier to extract the three-dimensional coordinates of skeletons when depth information is given with the photos. Here, we present a method for detecting hand signals that does not rely on skeletons. Instead of skeletons, we employ basic object detectors that have been trained to respond to hand signals. 
<li>Autonomous vehicles require traffic police gesture recognition. Current traffic police gesture identification systems frequently extract pixel-level characteristics from RGB photos, which are incoherent owing to the absence of gesture skeleton features and can result in erroneous results. Existing object detection algorithms enable the detection of automobiles, trees, people, bicycles, animals, and so forth (YOLO).

<li>In this project, we will employ the Convolutional Neural Network (CNN) approach (Deep Learning) to recognise traffic police hand signals. As there are no acceptable datasets available, we shall attempt to generate our own. 
Mediapipe will be utilised in its development.
</ul>
</div></font>
<br>

***

## Introduction
<font size="10">
<div align="justify">
  <ul>
    <li>Artificial intelligence has been implemented in various industries, particularly in computer vision, improving object detection and classification. Traffic signals play a significant role in traffic flow and safety, and AI can be used to enhance their effectiveness.
    <li>Recognizing traffic police gestures is challenging due to the lack of interpretable features from RGB images and interference from complex environments. Gesture skeleton extractor (GSE) can be used to extract interpretable skeleton coordinate information, eliminating background interference. However, existing deep learning methods are not suitable for handling gesture skeleton features, requiring the development of new methods.
    <li>Previous studies faced limitations due to the reliance on handcrafted components or pixel-level information, resulting in poor recognition performance and weak generalization capability. Autonomous vehicles must also recognize hand signals from traffic controllers in real-time, which can be challenging due to the high speed of the vehicles. Many skeleton-based methods require high computational load or expensive devices, making them unsuitable for real-time processing.
    <li>Prior approaches used skeleton-based action recognition to identify hand signals in videos, requiring preprocessing and limiting generalizability to real-world problems. Accurate detection of hand signals requires distinguishing between intentional and non-intentional signals. The ability to differentiate situations affects the accuracy of detection.
    <li>The paper proposes a new method to recognize hand signals from raw video streams without preprocessing, overcoming the challenges of previous methods that relied on skeleton-based action recognition. The proposed method utilizes an attention-based spatial-temporal graph convolutional network (ASTGCN) that achieved higher accuracy than previous methods and can distinguish between intentional and non-intentional signals. The potential applications of this method include traffic management, public safety, and military operations. The paper highlights the significance of deep learning in recognizing hand signals in real-world environments.
  </ul>
</div></font>
<br>

***

## Division Of Work:
      
### Our Project Will be Divided into 3 parts:

<div align="justify">

<b>Part 1:</b> <br/>
DATASET Collection and Preprocess the pose classification training data into a CSV file,specifying the landmarks (body key points) and ground truth pose labelsrecognized by the Mediapipe and MoveNet.<br/>
<b>Konark Patel(20BCI0169) , Nikhil(20BCI0159)</b><br/><br/>
<ul><li>There are various processes involved in preparing pose classification training data into a CSV file, and they will be distributed among the team members. </li><li>Together, we will check the annotated data for accuracy and consistency, as well as the main points that were extracted and the CSV file that was produced.</li><li>To make the training data larger, we will also take into account data augmentation methods including flipping, rotating, and scaling.</li><li>Ultimately, the entire team will finish the preparation of posture categorization training data into a CSV file, which is an essential step in training a pose estimate model.</li></ul>

<b>Part 2:</b> <br/>
Build and train a pose classification model(MoveNet Model) that takes landmark coordinates from a CSV file as input and outputs predicted labels.<br/>
<b>Vandit Gabani(20BCI0090) ,Aditi(20BCI0128) , Shaunak(20BCI0138)</b><br/><br/>

<ul><li>There are various processes involved in creating and refining a pose categorization model, and they will be distributed among the team members.</li><li>To ensure that the model architecture and training parameters are properly stated, everyone will work together to construct and train the model.</li><li>They will also cooperate to assess the model's performance on the validation set and make any required modifications to increase its accuracy.</li><li>In general, developing and training a pose classification model necessitates a solid grasp of computer vision and deep learning, as well as a methodical approach to model choice, implementation, and training.</li></ul>
<b>Part 3:</b> <br/>
Convert the pose classification model to TFLite and Test and Deployment of Model.<br/>
<b>Harsh(20BCI0271) , Payal(20BCE2759) , Yash(20BCI0176)</b><br/><br/>

<ul>
<li>There are various processes involved in converting a pose classification model to TFLite, which may be broken down into jobs for each team member.</li>
<li>To guarantee that the conversion procedure goes without a hitch and that the TFLite model is appropriately optimised and validated, everyone will cooperate.</li>
<li>Overview: We will utilise our own dataset, which consists of 7 distinct traffic police/pose photographs, as there isn't a dataset readily available online. </li>
<li>Our dataset consists of almost 5000–6000 unique photos of the crucial 7 traffic postures that were narrowed down.</li>
</ul>
    
  *** 
      
 ## Timeline - Gantt Chart  

 The divison of project in sub phases using a Gant chart.</br>
<img src="https://github.com/harsh-rajpal/Hashers-TARP/blob/main/Timeline.png?raw=true">
      <br/>
 
 ## Workflow
Tasks in various phases using flowchart diagram</br>
<img src="Workflow Breakdown/Flowchart-5.png"><br/>

---

### 1.Planning and Modelling
<img src="Workflow Breakdown/Flowchart-1.png"><br/>

---

### 2.DataCollection and Real time detection usinf mediapipe
<img src="Workflow Breakdown/Flowchart-2.png"><br/>

---

### 3.Training Data model using MoveNet
<img src="Workflow Breakdown/Flowchart-3.png"><br/>

---

### 4.Testing and Deployment
<img src="Workflow Breakdown/Flowchart-4.png"><br/>

---

### Workflow Breakdown(Using Mediapipe):

  <img src="https://github.com/harsh-rajpal/Hashers-TARP/blob/main/Workflow%20Breakdown/mediapipeflowchart.png?raw=true" height="1000">
  
### Workflow Breakdown(Using MoveNet Model):
  
  <img src="https://github.com/harsh-rajpal/Hashers-TARP/blob/main/Workflow%20Breakdown/MoveNetflowchart.png?raw=true" height="1000">
  


## Tools and Software: Implementation
<h3>1. Mediapipe Library(for Realtime Detection)</h3>
<p><ul><li>MеdiаPiре is а Frаmеwоrk fоr building mасhinе lеаrning рiреlinеs fоr рrосеssing timе-sеriеs dаtа likе vidео, аudiо, еtс.</li><li> This сrоss-рlаtfоrm Frаmеwоrk wоrks in Dеsktор/Sеrvеr, Andrоid, iOS, аnd еmbеddеd dеviсеs likе Rаsрbеrrу Pi аnd Jеtsоn Nаnо.</li><li> MеdiаPiре Tооlkit соmрrisеs thе Frаmеwоrk аnd thе Sоlutiоns.</li><li> Hаndроsе rесоgnitiоn is а dеер lеаrning tесhniquе thаt аllоws уоu tо dеtесt diffеrеnt роints оn уоur hаnd.</li><li> Thеsе роints оn уоur hаnd аrе соmmоnlу rеfеrrеd tо аs lаndmаrks.</li><li> Thеsе lаndmаrks соnsist оf jоints, tiрs, аnd bаsеs оf уоur fingеrs.</li></ul></p>
<table cellpadding="0" cellspacing="0" border="0" width="100%">
  <tr>
    <td align="center"><img src="https://mediapipe.dev/images/mobile/pose_tracking_full_body_landmarks.png" ></td>
  </tr>
 </table>
<ol>
 <li>left_elbow_angle From left_shoulder , left_elbow and left_wrist.
 <li>right_elbow_angle From right_shoulder , right_elbow and right_wrist.
 <li>left_shoulder_angle From left_shoulder , left_elbow and left_hip.
 <li>right_shoulder_angle From right_shoulder , right_elbow and right_hip.
</ol>

<h2>2. MoveNet Model(Test and Train Dataset)</h3>
<p><ul></ul><li>MoveNet is a model that recognises 17 key spots on a body very quickly and precisely.</li><li>Two variations of the model, dubbed Lightning and Thunder, are available on TF Hub.</li><li>Thunder is designed for applications requiring great precision, whereas Lightning is designed for applications where latency is crucial.</li><li>For the majority of contemporary computers, laptops, and phones, both models operate faster than real time (30+ FPS), which is essential for live fitness, health, and wellness applications.</li></p>

<img src="https://www.marktechpost.com/wp-content/uploads/2021/05/Screen-Shot-2021-05-25-at-11.54.07-AM-768x505.png">
  </br>
  <ul>
    <li>Numpy and Pandas Library for CSV files.
    <li>opencv (cv2) for realtime video detection and extraction of landmark  points.
    <li>tensorflow : MovenetModel Training and Testing.
    <li>sklearn
    <li>Keras Model : Pose Classification
  </ul>
  
## How MoveNet Model Works?
<ol>
  <li><b>MoveNet Thunder:</b> The model chosen for the application is the Thunder variant of the Movenet model. MoveNet thunder, even though is slightly slower compared to its counterpart – lightening, is highly accurate, suitable for the proposed application.   
  <li><b>detect(input_tensor, inference_count=3):</b> Runs detection on an input image.<br>
    <b>Args :</b><br>
    <b>input_tensor :</b> A [height, width, 3] Tensor of type tf.float32.Note that height and width can be anything since the image will be immediately resized according to
    the needs of the model within thisfunction.<br>
      <b>inference_count : </b>Number of times the model should run repeatly on the same input image to improve detection accuracy.<br>
    <b>Returns :</b><br>A Person entity detected by the MoveNet.SinglePose. 
  
  <li><b>draw_prediction_on_image( image,person,crop_region=None,close_figure=True, keep_input_size=False):</b> Draws the keypoint predictions on image.<br>
  <b>Args :</b><br>
  <b>image:</b> An numpy array with shape [height, width, channel] representing the pixel values of the input image.<br>
<b>person:</b> A person entity returned from the MoveNet.SinglePose model. <br>
<b>close_figure:</b> Whether to close the plt figure after the function returns.<br>
<b>keep_input_size:</b> Whether to keep the size of the input image.<br>
    <b>Returns:<b>An numpy array with shape [out_height, out_width, channel] representing the image overlaid with keypoint predictions.
      
   <li>Class MoveNetPreprocessor(object):Helper class to preprocess pose sample images for classification.Creates a preprocessor to detection pose from images and save as CSV. 
   <li>Split Dataset into train and test.
   <li>load pose landmarks from images and store into csv files.
   <li>get center point for all landmarks and store it;s distance from every landmarks into csv file.
   <li>Now,normalize every coordinates by moving center to (0,0).
   <li>Define the Model and train Datset.
  </ol>
  

## Example of Dataset:
  </br><p>
  <img src="https://user-images.githubusercontent.com/79594169/228626817-b2d97684-f687-4b62-8785-bff8c4fabd14.jpg" height="500">
  </br>
  
## Dataset with overlapping model:
  </br><p>
  <img src="https://user-images.githubusercontent.com/79594169/228628410-c78704ba-5abf-45f8-91e8-ed993d9568eb.jpg" height="500">
  </br>
  
  <p>By identifying the 32 description points on the dataset image we are able to identify the angles and position the subject is forming which helps determine its gesture.</br>
  </br>
  
  <h4>NoPose</h4>
  <img src="https://github.com/harsh-rajpal/Hashers-TARP/blob/main/images/13.jpg" height="500" >
  
  <h4>Start Vehicle on T point</h4>
    <img src="https://github.com/harsh-rajpal/Hashers-TARP/blob/main/images/14.jpg">

  <h4>Start Vehicle From Left</h4>
    <img src="https://github.com/harsh-rajpal/Hashers-TARP/blob/main/images/15.jpg" height="500">

  <h4>Start Vehicle From Right</h4>
    <img src="https://github.com/harsh-rajpal/Hashers-TARP/blob/main/images/16.jpg">

  <h4>Stop Vehicles From behind</h4>
    <img src="https://github.com/harsh-rajpal/Hashers-TARP/blob/main/images/17.jpg">

  <h4>Stop Vehicles From Left and Right</h4>
    <img src="https://github.com/harsh-rajpal/Hashers-TARP/blob/main/images/18.jpg">

  <h4>Stop Vehicles From Front</h4>
    <img src="https://github.com/harsh-rajpal/Hashers-TARP/blob/main/images/19.jpg" height="500">

  <h4>Stop Vehicles From Left and Right</h4>
    <img src="https://github.com/harsh-rajpal/Hashers-TARP/blob/main/images/20.jpg" height="500">

  
  ## Code:
  
  <h4>Link:</h4>
  https://colab.research.google.com/drive/12VW66qOZcqRNwm26GZmab0se73G3cnDf
  <br>
  <br>


``` python
#os, random, shutil, and csv: 
#These libraries provide functionality to work with the file system, randomly generate values, and read and write CSV files.

#numpy, pandas, matplotlib, and cv2: 
#These libraries are commonly used in data science and machine learning workflows to manipulate, analyze, and visualize data.

#machine learning libraries such as tensorflow and keras. 
#These libraries provide functionality to build, train, and evaluate machine learning models. 
#The script also imports sklearn which provides additional machine learning utilities such as train-test splitting, accuracy calculation, and confusion matrix generation.

import os
import random
import shutil
import csv
import cv2
import itertools
import numpy as np
import pandas as pd
import sys
import tempfile
import tqdm
import time

from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
```

``` python
#setting up the path to a specific example in the TensorFlow Lite library for pose estimation on Raspberry Pi. 
#The first line uses the os library to join the current working directory with a subdirectory path examples/lite/examples/pose_estimation/raspberry_pi. 
#The resulting path is stored in the pose_sample_rpi_path variable.

#The second line of code uses the sys library to append the pose_sample_rpi_path to the system path, 
#allowing Python to locate and import modules from this directory.
pose_sample_rpi_path = os.path.join(os.getcwd(), 'examples/lite/examples/pose_estimation/raspberry_pi')
sys.path.append(pose_sample_rpi_path)
```

``` python
#The code also imports the Movenet class from the ml module, which is likely a wrapper class around the TensorFlow Lite model used for pose estimation. 
#The Movenet class takes a string argument specifying the type of Movenet model to use. In this case, the argument is 'movenet_thunder'.
import utils
from data import BodyPart
from ml import Movenet
movenet = Movenet('movenet_thunder')
```


``` python
def detect(input_tensor, inference_count=3):
  """Runs detection on an input image.
  Args:
    input_tensor: A [height, width, 3] Tensor of type tf.float32.
      Note that height and width can be anything since the image will be
      immediately resized according to the needs of the model within this
      function.
    inference_count: Number of times the model should run repeatly on the
      same input image to improve detection accuracy.

  Returns:
    A Person entity detected by the MoveNet.SinglePose.
  """
  channel,image_height, image_width = input_tensor.shape

  # Detect pose using the full input image
  movenet.detect(input_tensor.numpy(), reset_crop_region=True)

  # Repeatedly using previous detection result to identify the region of
  # interest and only croping that region to improve detection accuracy
  for _ in range(inference_count - 1):
    person = movenet.detect(input_tensor.numpy(),
                            reset_crop_region=False)

  return person
```

``` python
def draw_prediction_on_image(
    image, person, crop_region=None, close_figure=True,
    keep_input_size=False):
  """Draws the keypoint predictions on image.

  Args:
    image: An numpy array with shape [height, width, channel] representing the
      pixel values of the input image.
    person: A person entity returned from the MoveNet.SinglePose model.
    close_figure: Whether to close the plt figure after the function returns.
    keep_input_size: Whether to keep the size of the input image.

  Returns:
    An numpy array with shape [out_height, out_width, channel] representing the
    image overlaid with keypoint predictions.
  """
  # Draw the detection result on top of the image.
  image_np = utils.visualize(image, [person])

  # Plot the image with detection results.
  height, width, channel = image.shape
  aspect_ratio = float(width) / height
  fig, ax = plt.subplots(figsize=(12 * aspect_ratio, 12))
  im = ax.imshow(image_np)

  if close_figure:
    plt.close(fig)

  if not keep_input_size:
    image_np = utils.keep_aspect_ratio_resizer(image_np, (512, 512))

  return image_np
```

``` python
class MoveNetPreprocessor(object):
  """Helper class to preprocess pose sample images for classification."""

  def __init__(self,
               images_in_folder,
               images_out_folder,
               csvs_out_path):
    """Creates a preprocessor to detection pose from images and save as CSV.

    Args:
      images_in_folder: Path to the folder with the input images. It should
        follow this structure:
        yoga_poses
        |__ downdog
            |______ 00000128.jpg
            |______ 00000181.bmp
            |______ ...
        |__ goddess
            |______ 00000243.jpg
            |______ 00000306.jpg
            |______ ...
        ...
      images_out_folder: Path to write the images overlay with detected
        landmarks. These images are useful when you need to debug accuracy
        issues.
      csvs_out_path: Path to write the CSV containing the detected landmark
        coordinates and label of each image that can be used to train a pose
        classification model.
    """
    self._images_in_folder = images_in_folder
    self._images_out_folder = images_out_folder
    self._csvs_out_path = csvs_out_path
    self._messages = []

    # Create a temp dir to store the pose CSVs per class
    self._csvs_out_folder_per_class = tempfile.mkdtemp()

    # Get list of pose classes and print image statistics
    self._pose_class_names = sorted(
        [n for n in os.listdir(self._images_in_folder) if not n.startswith('.')]
        )

  def process(self, per_pose_class_limit=None, detection_threshold=0.1):
    """Preprocesses images in the given folder.
    Args:
      per_pose_class_limit: Number of images to load. As preprocessing usually
        takes time, this parameter can be specified to make the reduce of the
        dataset for testing.
      detection_threshold: Only keep images with all landmark confidence score
        above this threshold.
    """
    # Loop through the classes and preprocess its images
    for pose_class_name in self._pose_class_names:
      print('Preprocessing', pose_class_name, file=sys.stderr)

      # Paths for the pose class.
      images_in_folder = os.path.join(self._images_in_folder, pose_class_name)
      images_out_folder = os.path.join(self._images_out_folder, pose_class_name)
      csv_out_path = os.path.join(self._csvs_out_folder_per_class,
                                  pose_class_name + '.csv')
      if not os.path.exists(images_out_folder):
        os.makedirs(images_out_folder)

      # Detect landmarks in each image and write it to a CSV file
      with open(csv_out_path, 'w') as csv_out_file:
        csv_out_writer = csv.writer(csv_out_file, 
                                    delimiter=',', 
                                    quoting=csv.QUOTE_MINIMAL)
        # Get list of images
        image_names = sorted(
            [n for n in os.listdir(images_in_folder) if not n.startswith('.')])
        if per_pose_class_limit is not None:
          image_names = image_names[:per_pose_class_limit]

        valid_image_count = 0

        # Detect pose landmarks from each image
        for image_name in tqdm.tqdm(image_names):
          image_path = os.path.join(images_in_folder, image_name)

          try:
            image = tf.io.read_file(image_path)
            image = tf.io.decode_jpeg(image)
          except:
            self._messages.append('Skipped ' + image_path + '. Invalid image.')
            continue
          else:
            image = tf.io.read_file(image_path)
            image = tf.io.decode_jpeg(image)
            image_height, image_width, channel = image.shape

          # Skip images that isn't RGB because Movenet requires RGB images
          if channel != 3:
            self._messages.append('Skipped ' + image_path +
                                  '. Image isn\'t in RGB format.')
            continue
          person = detect(image)

          # Save landmarks if all landmarks were detected
          min_landmark_score = min(
              [keypoint.score for keypoint in person.keypoints])
          should_keep_image = min_landmark_score >= detection_threshold
          if not should_keep_image:
            self._messages.append('Skipped ' + image_path +
                                  '. No pose was confidentlly detected.')
            continue

          valid_image_count += 1

          # Draw the prediction result on top of the image for debugging later
          output_overlay = draw_prediction_on_image(
              image.numpy().astype(np.uint8), person, 
              close_figure=True, keep_input_size=True)

          # Write detection result into an image file
          output_frame = cv2.cvtColor(output_overlay, cv2.COLOR_RGB2BGR)
          cv2.imwrite(os.path.join(images_out_folder, image_name), output_frame)

          # Get landmarks and scale it to the same size as the input image
          pose_landmarks = np.array(
              [[keypoint.coordinate.x, keypoint.coordinate.y, keypoint.score]
                for keypoint in person.keypoints],
              dtype=np.float32)

          # Write the landmark coordinates to its per-class CSV file
          coordinates = pose_landmarks.flatten().astype(str).tolist()
          # coordinates = pose_landmarks.flatten().astype(np.str).tolist()
          csv_out_writer.writerow([image_name] + coordinates)

        if not valid_image_count:
          raise RuntimeError(
              'No valid images found for the "{}" class.'
              .format(pose_class_name))

    # Print the error message collected during preprocessing.
    print('\n'.join(self._messages))

    # Combine all per-class CSVs into a single output file
    all_landmarks_df = self._all_landmarks_as_dataframe()
    all_landmarks_df.to_csv(self._csvs_out_path, index=False)

  def class_names(self):
    """List of classes found in the training dataset."""
    return self._pose_class_names

  def _all_landmarks_as_dataframe(self):
    """Merge all per-class CSVs into a single dataframe."""
    total_df = None
    for class_index, class_name in enumerate(self._pose_class_names):
      csv_out_path = os.path.join(self._csvs_out_folder_per_class,
                                  class_name + '.csv')
      per_class_df = pd.read_csv(csv_out_path, header=None)

      # Add the labels
      per_class_df['class_no'] = [class_index]*len(per_class_df)
      per_class_df['class_name'] = [class_name]*len(per_class_df)

      # Append the folder name to the filename column (first column)
      per_class_df[per_class_df.columns[0]] = (os.path.join(class_name, '') 
        + per_class_df[per_class_df.columns[0]].astype(str))

      if total_df is None:
        # For the first class, assign its data to the total dataframe
        total_df = per_class_df
      else:
        # Concatenate each class's data into the total dataframe
        total_df = pd.concat([total_df, per_class_df], axis=0)

    list_name = [[bodypart.name + '_x', bodypart.name + '_y', 
                  bodypart.name + '_score'] for bodypart in BodyPart] 
    header_name = []
    for columns_name in list_name:
      header_name += columns_name
    header_name = ['file_name'] + header_name
    header_map = {total_df.columns[i]: header_name[i] 
                  for i in range(len(header_name))}

    total_df.rename(header_map, axis=1, inplace=True)

    return total_df
```

``` python
#read image from given dir and store content of image into tensor.pass it from detect function to detect object.
image = tf.io.read_file('Dataset//Stop Vehicles From Front//img996.jpg')
image = tf.io.decode_jpeg(image)
person = detect(image)= draw_prediction_on_image(image.numpy(), person, crop_region=None, 
                               close_figure=False, keep_input_size=True)
```


``` python
def processTensorImage(path):
    if(type(path) == str):
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image)
    else:
        image = tf.convert_to_tensor(path, dtype=tf.float32)
    return image


# def processTensorImageFromNumpy(image):
#     image = tf.convert_to_tensor(image, dtype=tf.float32)
#     return image
```


``` python
use_custom_dataset = True
is_skip_step_1 = False
dataset_is_split = False
```

``` python
def split_into_train_test(images_origin, images_dest, test_split):
  """Splits a directory of sorted images into training and test sets.

  Args:
    images_origin: Path to the directory with your images. This directory
      must include subdirectories for each of your labeled classes. For example:
      yoga_poses/
      |__ downdog/
          |______ 00000128.jpg
          |______ 00000181.jpg
          |______ ...
      |__ goddess/
          |______ 00000243.jpg
          |______ 00000306.jpg
          |______ ...
      ...
    images_dest: Path to a directory where you want the split dataset to be
      saved. The results looks like this:
      split_yoga_poses/
      |__ train/
          |__ downdog/
              |______ 00000128.jpg
              |______ ...
      |__ test/
          |__ downdog/
              |______ 00000181.jpg
              |______ ...
    test_split: Fraction of data to reserve for test (float between 0 and 1).
  """
  _, dirs, _ = next(os.walk(images_origin))

  TRAIN_DIR = os.path.join(images_dest, 'train')
  TEST_DIR = os.path.join(images_dest, 'test')
  os.makedirs(TRAIN_DIR, exist_ok=True)
  os.makedirs(TEST_DIR, exist_ok=True)

  for dir in dirs:
    # Get all filenames for this dir, filtered by filetype
    filenames = os.listdir(os.path.join(images_origin, dir))
    filenames = [os.path.join(images_origin, dir, f) for f in filenames if (
        f.endswith('.png') or f.endswith('.jpg') or f.endswith('.jpeg') or f.endswith('.bmp'))]
    # Shuffle the files, deterministically
    filenames.sort()
    random.seed(42)
    random.shuffle(filenames)
    # Divide them into train/test dirs
    os.makedirs(os.path.join(TEST_DIR, dir), exist_ok=True)
    os.makedirs(os.path.join(TRAIN_DIR, dir), exist_ok=True)
    test_count = int(len(filenames) * test_split)
    for i, file in enumerate(filenames):
      if i < test_count:
        destination = os.path.join(TEST_DIR, dir, os.path.split(file)[1])
      else:
        destination = os.path.join(TRAIN_DIR, dir, os.path.split(file)[1])
      shutil.copyfile(file, destination)
    print(
        f'Moved {test_count} of {len(filenames)} from class "{dir}" into test.')
  print(f'Your split dataset is in "{images_dest}"')
```

``` python
#SPLIT DATASET INTO TEST AND TRAIN

# dataset_in = 'Dataset'
#  # You can leave the rest alone:
# if not os.path.isdir(dataset_in):
#    raise Exception("dataset_in is not a valid directory")
# IMAGES_ROOT = dataset_in


dataset_in = 'Dataset'

if not os.path.isdir(dataset_in):
   raise Exception("dataset_in is not a valid directory")

dataset_out = 'split_' + dataset_in
split_into_train_test(dataset_in, dataset_out, test_split=0.2)
IMAGES_ROOT = dataset_out
```


``` python
#TRAINING DATASET

images_in_train_folder = os.path.join(IMAGES_ROOT, 'train') #path to train dataset
images_out_train_folder = 'poses_images_out_train' #tensor files for train datset
csvs_out_train_path = 'train_data.csv' #path to csv file of train dataset

#training model using MOVENETPREPROCESSOR
preprocessor = MoveNetPreprocessor(
    images_in_folder=images_in_train_folder,
    images_out_folder=images_out_train_folder,
    csvs_out_path=csvs_out_train_path,
)

#preprocessor.process(per_pose_class_limit=None, detection_threshold=0.0)
preprocessor.process(per_pose_class_limit=None)
```


``` python
#Now TESTING the Model Using Test Dataset

images_in_test_folder = os.path.join(IMAGES_ROOT, 'test')
images_out_test_folder = 'poses_images_out_test'
csvs_out_test_path = 'test_data.csv'

preprocessor = MoveNetPreprocessor(
    images_in_folder=images_in_test_folder,
    images_out_folder=images_out_test_folder,
    csvs_out_path=csvs_out_test_path,
)

#preprocessor.process(per_pose_class_limit=None,detection_threshold=0.0)
preprocessor.process(per_pose_class_limit=None)
```


``` python
def load_pose_landmarks(csv_path):
  """Loads a CSV created by MoveNetPreprocessor.

  Returns:
    X: Detected landmark coordinates and scores of shape (N, 17 * 3)
    y: Ground truth labels of shape (N, label_count)
    classes: The list of all class names found in the dataset
    dataframe: The CSV loaded as a Pandas dataframe features (X) and ground
      truth labels (y) to use later to train a pose classification model.
  """

  # Load the CSV file
  dataframe = pd.read_csv(csv_path)
  df_to_process = dataframe.copy()

  # Drop the file_name columns as you don't need it during training.
  df_to_process.drop(columns=['file_name'], inplace=True)

  # Extract the list of class names
  classes = df_to_process.pop('class_name').unique()

  # Extract the labels
  y = df_to_process.pop('class_no')

  # Convert the input features and labels into the correct format for training.
  X = df_to_process.astype('float64')
  y = keras.utils.to_categorical(y)

  return X, y, classes, dataframe
```

``` python
# Load the train data
X, y, class_names, _ = load_pose_landmarks(csvs_out_train_path)

# Split training data (X, y) into (X_train, y_train) and (X_val, y_val)
X_train, X_val, y_train, y_val = train_test_split(X, y,
                                                  test_size=0.15)
```

``` python
# Load the test data
X_test, y_test, _, df_test = load_pose_landmarks(csvs_out_test_path)
```

``` python
def get_center_point(landmarks, left_bodypart, right_bodypart):
  """Calculates the center point of the two given landmarks."""

  left = tf.gather(landmarks, left_bodypart.value, axis=1)
  right = tf.gather(landmarks, right_bodypart.value, axis=1)
  center = left * 0.5 + right * 0.5
  return center


def get_pose_size(landmarks, torso_size_multiplier=2.5):
  """Calculates pose size.

  It is the maximum of two values:
    * Torso size multiplied by `torso_size_multiplier`
    * Maximum distance from pose center to any pose landmark
  """
  # Hips center
  hips_center = get_center_point(landmarks, BodyPart.LEFT_HIP,
                                 BodyPart.RIGHT_HIP)

  # Shoulders center
  shoulders_center = get_center_point(landmarks, BodyPart.LEFT_SHOULDER,
                                      BodyPart.RIGHT_SHOULDER)

  # Torso size as the minimum body size
  torso_size = tf.linalg.norm(shoulders_center - hips_center)

  # Pose center
  pose_center_new = get_center_point(landmarks, BodyPart.LEFT_HIP,
                                     BodyPart.RIGHT_HIP)
  pose_center_new = tf.expand_dims(pose_center_new, axis=1)
  # Broadcast the pose center to the same size as the landmark vector to
  # perform substraction
  pose_center_new = tf.broadcast_to(pose_center_new,[tf.size(landmarks) // (17*2), 17, 2])

  # Dist to pose center
  d = tf.gather(landmarks - pose_center_new, 0, axis=0,
                name="dist_to_pose_center")
  # Max dist to pose center
  max_dist = tf.reduce_max(tf.linalg.norm(d, axis=0))

  # Normalize scale
  pose_size = tf.maximum(torso_size * torso_size_multiplier, max_dist)

  return pose_size


def normalize_pose_landmarks(landmarks):
  """Normalizes the landmarks translation by moving the pose center to (0,0) and
  scaling it to a constant pose size.
  """
  # Move landmarks so that the pose center becomes (0,0)
  pose_center = get_center_point(landmarks, BodyPart.LEFT_HIP,
                                 BodyPart.RIGHT_HIP)
  pose_center = tf.expand_dims(pose_center, axis=1)
  # Broadcast the pose center to the same size as the landmark vector to perform
  # substraction
  pose_center = tf.broadcast_to(pose_center,[tf.size(landmarks) // (17*2), 17, 2])
  landmarks = landmarks - pose_center

  # Scale the landmarks to a constant pose size
  pose_size = get_pose_size(landmarks)
  landmarks /= pose_size

  return landmarks


def landmarks_to_embedding(landmarks_and_scores):
  """Converts the input landmarks into a pose embedding."""
  # Reshape the flat input into a matrix with shape=(17, 3)
  reshaped_inputs = keras.layers.Reshape((17, 3))(landmarks_and_scores)
  # Normalize landmarks 2D
  landmarks = normalize_pose_landmarks(reshaped_inputs[:, :, :2])
  # Flatten the normalized landmark coordinates into a vector
  embedding = keras.layers.Flatten()(landmarks)
  return embedding
```

``` python
# Define the model - this is our keras - model (Using Movenet and Deep Learning)
inputs = tf.keras.Input(shape=(51))
embedding = landmarks_to_embedding(inputs)

#layer = keras.layers.Dense(128, activation=tf.nn.relu6)(embedding): This line creates a fully connected dense layer with 128 neurons and applies the relu6 activation function to the output. The input to this layer is the embedding tensor.
#layer = keras.layers.Dropout(0.5)(layer): This line applies a dropout layer to the output of the previous layer with a rate of 0.5. Dropout is a regularization technique that randomly drops out a fraction of the neurons in the layer during training to prevent overfitting.
#layer = keras.layers.Dense(64, activation=tf.nn.relu6)(layer): This line creates another fully connected dense layer with 64 neurons and applies the relu6 activation function to the output. The input to this layer is the output of the previous dropout layer.
#layer = keras.layers.Dropout(0.5)(layer): This line applies another dropout layer with a rate of 0.5 to the output of the previous dense layer.
#outputs = keras.layers.Dense(len(class_names), activation="softmax")(layer): This line creates the output layer with len(class_names) neurons and applies the softmax activation function to the output. The input to this layer is the output of the previous dropout layer, which is the final output of the model.
layer = keras.layers.Dense(128, activation=tf.nn.relu6)(embedding)
layer = keras.layers.Dropout(0.5)(layer)
layer = keras.layers.Dense(64, activation=tf.nn.relu6)(layer)
layer = keras.layers.Dropout(0.5)(layer)
outputs = keras.layers.Dense(len(class_names), activation="softmax")(layer)

model = keras.Model(inputs, outputs)
model.summary()
```


``` python
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Add a checkpoint callback to store the checkpoint that has the highest
# validation accuracy.
checkpoint_path = "weights.best.hdf5"
checkpoint = keras.callbacks.ModelCheckpoint(checkpoint_path,
                                             monitor='val_accuracy',
                                             verbose=1,
                                             save_best_only=True,
                                             mode='max')
earlystopping = keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                              patience=20)

# Start training
history = model.fit(X_train, y_train,
                    epochs=200,
                    batch_size=16,
                    validation_data=(X_val, y_val),
                    callbacks=[checkpoint, earlystopping])

```


``` python
# Visualize the training history to see whether you're overfitting.
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['TRAIN', 'VAL'], loc='lower right')
plt.show()
```


``` python
# Evaluate the model using the TEST dataset
loss, accuracy = model.evaluate(X_test, y_test)
```


``` python
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
  """Plots the confusion matrix."""
  if normalize:
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print("Normalized confusion matrix")
  else:
    print('Confusion matrix, without normalization')

  plt.imshow(cm, interpolation='nearest', cmap=cmap)
  plt.title(title)
  plt.colorbar()
  tick_marks = np.arange(len(classes))
  plt.xticks(tick_marks, classes, rotation=55)
  plt.yticks(tick_marks, classes)
  fmt = '.2f' if normalize else 'd'
  thresh = cm.max() / 2.
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, format(cm[i, j], fmt),
              horizontalalignment="center",
              color="white" if cm[i, j] > thresh else "black")

  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  plt.tight_layout()

# Classify pose in the TEST dataset using the trained model
y_pred = model.predict(X_test)

# Convert the prediction result to class name
y_pred_label = [class_names[i] for i in np.argmax(y_pred, axis=1)]
y_true_label = [class_names[i] for i in np.argmax(y_test, axis=1)]

# Plot the confusion matrix
cm = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
plot_confusion_matrix(cm,
                      class_names,
                      title ='Confusion Matrix of Pose Classification Model')

# Print the classification report
print('\nClassification Report:\n', classification_report(y_true_label,
                                                          y_pred_label))
```



    Classification Report:
                                        precision    recall  f1-score   support

                               NoPose       0.96      1.00      0.98        26
             Start Vehicle On T Point       0.97      0.97      0.97        79
             Start Vehicles From Left       0.95      0.99      0.97       102
            Start Vehicles From Right       0.94      0.93      0.93        97
            Start Vehicles On T Point       0.96      0.94      0.95       109
            Stop Vehicles From Behind       0.94      0.97      0.95       106
    Stop Vehicles From Left and Right       0.98      0.99      0.99       115
             Stop Vehicles from Front       0.99      0.92      0.95        95

                             accuracy                           0.96       729
                            macro avg       0.96      0.96      0.96       729
                         weighted avg       0.96      0.96      0.96       729

``` python
def processTensorImageFromNumpy(image):
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    return image

# def processTensorImage(path):
#     image = tf.io.read_file(path)
#     image = tf.image.decode_jpeg(image)
#     return image
```

``` python

# image = processTensorImage(image): This line calls the processTensorImage function to preprocess the input image tensor.
#person = detect(image): This line calls a function detect to detect a person in the input image using a pose detection algorithm. The detect function returns an object person that contains keypoints of the detected person.
#pose_landmarks = np.array([[keypoint.coordinate.x, keypoint.coordinate.y, keypoint.score] for keypoint in person.keypoints], dtype=np.float32): This line extracts the (x, y) coordinates and score of each keypoint in the detected person using a list comprehension, and converts the resulting list into a NumPy array pose_landmarks.
#coordinates = pose_landmarks.flatten().astype(str).tolist(): This line flattens the pose_landmarks array and converts it into a list of strings.
#df = pd.DataFrame([coordinates]).reset_index(drop=True): This line creates a Pandas DataFrame df from the coordinates list, with a single row and columns named after the keypoint indices.
#X = df.astype('float64'): This line converts the data type of the df DataFrame to float64, which is the data type required by the trained model.
#y = model.predict(X): This line passes the input data X to the trained model model to get the predicted pose classification probabilities y.
#y_pred = [class_names[i] for i in np.argmax(y, axis=1)]: This line finds the index of the maximum probability for each input sample in y using the NumPy argmax function, and then converts the indices into their corresponding pose classification labels using a list comprehension.
#return y_pred[0]: This line returns the predicted pose classification label for the input image tensor.
def classifyPose(image):
    image = processTensorImage(image)
    # image = processTensorImageFromNumpy(image)
    person = detect(image)
    pose_landmarks = np.array([[keypoint.coordinate.x, keypoint.coordinate.y, keypoint.score]
                              for keypoint in person.keypoints], dtype=np.float32)
    coordinates = pose_landmarks.flatten().astype(str).tolist()
    df = pd.DataFrame([coordinates]).reset_index(drop=True)
    X = df.astype('float64')
    y = model.predict(X)
    y_pred = [class_names[i] for i in np.argmax(y, axis=1)]
    return y_pred[0]
```

``` python
camera_video = cv2.VideoCapture(0)
camera_video.set(3, 1280)
camera_video.set(4, 960)
cv2.namedWindow('Traffic Sign Classification', cv2.WINDOW_NORMAL)
while camera_video.isOpened():
    ok, frame = camera_video.read()
    time1 = time.time()
    if not ok:
        continue
    frame_height, frame_width, _ = frame.shape
    frame = cv2.resize(
        frame, (int(frame_width * (320 / frame_height)), 320))
    pose = classifyPose(frame)
    frame = cv2.flip(frame, 1)
    time2 = time.time()
    fps = 0
    if (time2 - time1) > 0:
        fps = 1.0 / (time2 - time1)
    cv2.putText(frame, 'FPS: {}'.format(int(fps)), (500, 15),
                cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
    if pose=="NoPose":
        cv2.putText(frame, 'POSE: NoPose', (10, 15),
                cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
    else:
        cv2.putText(frame, 'POSE: {}'.format(pose), (10, 15),
                cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
    time1 = time2
    
    cv2.imshow('Traffic Sign Classification', frame)
    k = cv2.waitKey(1) & 0xFF
    if(k == 27):
        break
camera_video.release()
cv2.destroyAllWindows()
```


  ## Link to Project Demo(Video Presentation):
 
  ## Result and Discussion on Findings:
  
  <h4> Description about dataset used:</h4>
  Since the application is novel, no such dataset exits for the proposed application. Hence, we’ve prepared our own dataset consisting of
around 4000 images, that are classified and captures multiple environment scenarios for maximum accuracy. 

<h4> Experimental Results/Output:</h4>
 <img src="https://github.com/harsh-rajpal/Hashers-TARP/blob/main/images/Result.PNG">
 
 <h4>Accuracy</h4>
 <img src="https://github.com/harsh-rajpal/Hashers-TARP/blob/main/images/accuracy.png">
 
 <h4>Confusion Matrix</h4>
  <img src="https://github.com/harsh-rajpal/Hashers-TARP/blob/main/images/confusion.png">

 <h4>Classification Report</h4>
  <img src="https://github.com/harsh-rajpal/Hashers-TARP/blob/main/images/class.PNG">


 ## Issues in Existing System:
  <font size="10">
<div align="justify">
<ul>
  <li>Hand gesture recognition is an important task for traffic police as it enables them to control traffic flow more efficiently. In order to develop a deep learning model for hand gesture recognition, one approach is to use popular object detection models like YOLO. However, a slight change in the background can entirely change the image for the model, leading to false positives or true negatives in the predictions.To overcome this limitation, we need to train the model on a diverse set of images that capture different backgrounds, lighting conditions, and camera angles. This will enable the model to generalize better to new images that it has not seen before. 

<li>Autonomous vehicles require traffic police gesture recognition. Current traffic police gesture identification systems frequently extract pixel-level characteristics from RGB photos, which are incoherent owing to the absence of gesture skeleton features and can result in erroneous results. Existing object detection algorithms enable the detection of automobiles, trees, people, bicycles, animals, and so forth (YOLO).

<li>Another approach for hand gesture recognition is to use heuristic-based models that detect gestures on the basis of the location of feature points as perceived by the camera. However, this approach fails to perform with slight deviations in the relative position of the object on the image, slight shift in angle with respect to the camera and also the distance the object is from the camera. 
  
<li>To address this issue, we can use a combination of heuristic-based and deep learning-based
approaches. The heuristic-based model can be used as a pre-processing step to detect the
location of the hand in the image and estimate its orientation and distance from the camera.
This information can then be used to crop the image and feed it to a deep learning model
for gesture recognition. 
  
  <li>Moreover, data augmentation techniques such as random rotations, translations, and scaling can be applied to the training data to make the model more robust to variations in the input. In addition, we can also use techniques such as transfer learning, where a pretrained model on a large dataset can be fine-tuned on a smaller dataset of hand gestures to improve its performance.
    
  <li>Overall, developing an accurate and robust hand gesture recognition model for traffic police requires careful consideration of various factors such as the choice of model, diversity of training data, data augmentation techniques, and pre-processing steps.
</ul>
</div></font>
<br>
  
  ## Novelty of proposed work:
  <font size="10">
<div align="justify">
<ul>  
   
   <li>Nowdays In autonomous vehicles to detect different objects like other cars,trees,human,animals,water etc..YOLO algorithm is used,and this YOLO Algorithm has one drawback which is it can not detect traffic police gestures (Which is important in indian scenario) ,so our Movenet Model will overcome this issue and it is trained in such way that it can easily detect different traffic police gestures. 
     
   <li>As there are no dataset available for traffic police gestures we have collected around 5000 images and made our own custom dataset.We have collected this by making video of every pose and than extracted images from that video.
      
   <li>In Present in indian scenario there is no algorithm available which can able to detect this type of traffic police gestures,we are the first one to introduce this.
</ul>
</div></font>
<br>  

## Future Work:

  <font size="10">
<div align="justify">
<ul>  
   <li>As we are only detecting traffic police signals using movenet model,our model will only focus on the traffic police hand signals and it will not able to detect wheather the personal showing traffic hand signals is traffic police or any normal person.so we have to train our model in such way that it can also detect traffic police by their uniform by expanding our dataset.
    
   <li>Another Future work that can be done is we can merge our movenet model with YOLO Algorithm  so it can detect other traffic objects like trees,cars,humans,traffic signs,signals etc. and it can be used as fully working model for autonomus vehicles.
     
   <li>Add More images to Dataset to make our model more accurate in generating output and gussing the correct pose. 
   
   <li>For now our model can detect different signals perform by traffic police and it will inform/shown to autonomous vehicles about that signal.But we can modify it in such wawy that it can instruct autonomous vehicles about what signals traffic police showing as well as based on that what task that vehicle has to perform.
   <li>We can make one software for training traffic police or cadets about different traffic rules and gestures.

</ul>
</div></font>
<br> 

## Other Applications:

 ![image](https://user-images.githubusercontent.com/82700032/228662581-312cc764-69d0-4634-a8ab-80548ca258ff.png)
 
 ***
 
 ### 1. Yoga Pose Classification With TensorFlow’s MoveNet Model
      
 <div class="row">
  <div class="column">
    <img src="https://user-images.githubusercontent.com/82700032/228626302-26257f0d-7fbf-4255-a1cf-588bfd223141.png" style="width:50%">
  </div>
  <div class="column">
    <img src="https://user-images.githubusercontent.com/82700032/228626917-e2c5dc19-1ef0-4e83-a043-a39f812683ec.png" style="width:50%">
  </div>
</div>

<font size="10">
<div align="justify">

### The process for yoga pose detection using the Movenet model can be broken down into the following steps:

<b>1. Data Collection:</b> Collect a dataset of yoga pose images or videos. The dataset should include a diverse range of individuals with different body types and clothing.

<b>2. Data Preprocessing: </b> Resize and normalize the images to a fixed size. This step ensures that the model can handle input of a consistent size and shape.

<b>3. Model Training: </b> Train the Movenet model on the preprocessed dataset. The model can be trained using supervised or unsupervised learning.

<b>4. Inference: </b> Use the trained model to detect yoga poses in real-time or on pre-recorded videos. Inference involves processing the input data using the trained model to output the predicted pose.

<b>5. Post-processing: </b> Clean up the predicted pose by removing outliers and smoothing the pose over time. This step ensures that the final pose is accurate and smooth.

<b>6. Visualization: </b> Visualize the predicted pose by overlaying it on top of the input image or video. This step provides a visual representation of the predicted pose for evaluation and analysis.

  </div></font>
 
***

### 2. Instant Motion tracking using MediaPipe:
      
<img src="https://mediapipe.dev/images/mobile/instant_motion_tracking_android_small.gif">
<img src="https://4.bp.blogspot.com/-hZ5rq_NiERc/X0kYhGnu0iI/AAAAAAAAJVI/zUqHE8kZShQR14Oiax7MXNTBAbWmiEYXgCLcBGAsYHQ/s1600/unnamed%2B%25281%2529.gif">


<font size="10">
<div align="justify">

### Instant motion tracking using MediaPipe involves the following steps:

  <b>1. Install MediaPipe:</b> MediaPipe is an open-source framework developed by Google for building pipelines to process perceptual data. You can install MediaPipe by following the installation instructions provided on the official website.

<b>2. Import MediaPipe and relevant libraries:</b> Once you have installed MediaPipe, you can import it along with other relevant libraries such as OpenCV for image processing.

<b>3. Define the input source:</b> Define the input source, which could be a video file, a live video stream, or a series of images.

<b>4. Create a MediaPipe pipeline:</b> Create a MediaPipe pipeline that includes the motion tracking module. MediaPipe provides pre-trained models for motion tracking that you c<b>an use out-of-the-box.

<b>5. Process the input data:</b> Feed the input data to the pipeline and process it using the motion tracking module. MediaPipe will detect the motion and track it in real-time.

<b>6. Visualize the output:</b> Finally, visualize the output of the motion tracking module using OpenCV or any other visualization library.


  </div></font>

***

  ## Literature Survey

<table size="10">
  <tr>
    <th>Sr.No</th>        
    <th>Title/Author</th>
    <th>Techniques</th>
    <th>Future Work</th>
  </tr>
  <tr>
    <td>1.</td>
    <td><b>Objесt Dеtесtiоn in Sеlf Driving Cаrs Using Dеер Lеаrning</b><br/><br/> <b>Authоr</b> : Prаjwаl P, Prаjwаl D, Hаrish D H, Gаjаnаnа R, Jауаsri B S аnd S. Lоkеsh <br/>(IEE - 2021)</td>
    <td>
      <ul>
        <li>Convolutional Neural Networks (CNNs): CNNs are a type of deep learning neural network that are commonly used for image recognition and classification tasks. In the context of self-driving cars, CNNs are used to identify and classify objects in the environment, such as other vehicles, pedestrians, and traffic signs.</li>
        <li>Region Proposal Networks (RPNs): RPNs are a type of deep learning neural network that are used to identify potential object regions in an image. In the context of self-driving cars, RPNs are used to identify potential regions of interest in the environment, which can then be analyzed further using CNNs.</li>
        <li>Anchor Boxes: Anchor boxes are predefined bounding boxes that are used to predict the location of objects in an image. In the context of self-driving cars, anchor boxes are used to predict the location and size of objects in the environment.</li>
        <li>Non-Maximum Suppression (NMS): NMS is a technique used to filter out redundant object detections. In the context of self-driving cars, NMS is used to eliminate duplicate object detections and select the most accurate and relevant detections.
        </li>
        <li>Transfer Learning: Transfer learning is a technique used to leverage pre-trained models to solve new problems. In the context of self-driving cars, transfer learning is used to fine-tune pre-trained CNNs for object detection tasks in the environment.
        </li>
      </ul>
    </td>
    <td><ul><li>It саn оnlу able to dеtесt саr,реrsоn,аnimаl,trееs,Bus,dividеr,biсусlе еtс.</li></ul</td>
   </tr>
      
   <tr>
   <td>2.</td>
    <td><b>Trаffiс Pоliсе Gеsturе Rесоgnitiоn Bаsеd оn Gеsturе Skеlеtоn Eхtrасtоr аnd Multiсhаnnеl Dilаtеd Grарh Cоnvоlutiоn Nеtwоrk</b> <br/><br/> <b>Authоr</b> : Xin Xiоng , Hаоуuаn Wu , Wеidоng Min , Jiаnqiаng Xu , Qiуаn Fu аnd Chunjiаng Pеng<br/> (IEEE - 2021)</td>
    <td><ul><li>GSE - ехtrасts trаffiс роliсе skеlеtоn sеq. Frоm а vidео.(skеlеtоn сооrdinаtе)</li><li>MD-GCN - tаkе gеsturе skеlеtоn sеq. аs inрut аnd соnstruсt а grарh соnvоlutiоn.</li></ul></td>
    <td><ul><li>Duе tо thе diffеrеnсеs аnd сhаngеs in thе аnglе оf viеw, thе "lеft turn wаiting" might bе misсlаssifiеd аs "stор" аnd "slоw dоwn" might bе misсlаssifiеd аs "lеft turn"</li></ul</td>
   </tr>
      
      
  <tr>
    <td>3.</td>
    <td><b>Pоthоlе аnd Objесt Dеtесtiоn fоr аn Autоnоmоus Vеhiсlе Using YOLO </b><br/><br/> <b>Authоr</b> : Kаvithа R , Nivеthа S<br/>(IEEE - 2020)</td>
    <td><ul><li>YOLOv3</li><li>Cаmеrа сарturе thе оbjесt аs thе inрut imаgе.</li></ul></td>
    <td><ul><li>Dеtесt thе оbjесt сlаssеs likе: саr, реrsоn, truсk, bus, роthоlе, wеtlаnd, trаffiсlight, mоtоrсусlе.</li></ul</td>
   </tr>
      
      
      
   <tr>
    <td>4.</td>
    <td><b>Thе Rеаl-Timе Dеtесtiоn оf Trаffiс Pаrtiсiраnts Using YOLO Algоrithm </b><br/><br/> <b>Authоr</b> : Alеksа Ćоrоvić, Vеlibоr Ilić, Sinišа Đurić, Mаlišа Mаrijаn, аnd Bоgdаn Pаvkоvić  <br/>(IEEE - 2021)</td>
    <td><ul><li>YOLO.</li><li>First ехtrасt singlе imаgе frоm vidео 
strеаm аnd rеsizеd.</li><li>Imаgе gоеs tо CNN аnd bоunding bох 
is о/р.</li></ul></td>
    <td><ul><li>It саn оnlу dеtесt оbjесt likе саr, truсk, 
реdеstriаn, trаffiс signs, аnd lights.</li></ul</td>
   </tr>
      
      
      
   <tr>
    <td>5.</td>
    <td><b>Objесt Dеtесtiоn fоr Autоnоmоus Vеhiсlе using Singlе Cаmеrа with YOLOv4 аnd Mаррing Algоrithm </b><br/><br/> <b>Authоr</b> : Mосhаmmаd Sаhаl,Adе Oktаviаnus Kurniаwаn <br/>(IEEE - 2022)</td>
    <td><ul> <li>YOLOv4 with CSPDаrknеt-53.</li><li>Mаррing аlgоrithm fоr lосаtiоn.</li></ul></td>
    <td><ul><li>It саn оnlу able to dеtесt object like аnimаl,реорlе,trее,Vеhiсlе,tv.</li></ul</td>
   </tr>
      
      
      
      
   <tr>
    <td>6.</td>
    <td><b>On-rоаd оbjесt dеtесtiоn using Dеер Nеurаl Nеtwоrk </b><br/><br/> <b>Authоr</b> : Huiеun Kim, Yоungwаn Lее, Bуеоunghаk Yim, Eunsоо Pаrk, Hаkil Kim <br/>(IEEE - 2018)</td>
    <td><ul> <li>SSD(Singlе Shоt MultiBох Dеtесtоr).</li><li>RL(Rеinfоrсеmеnt Lеаrning)</li></ul></td>
    <td><ul><li>If dаtа is оbtаinеd frоm trаditiоnаl mеthоds suсh аs lоор dеtесtоrs, thеу will nоt рrоvidе ассurаtе оn-timе рrеdiсtiоns.
</li><li>lасk оf сlеаr роliсiеs, rеsistаnсе tо аdорting nеw tесhnоlоgiеs.</li></ul</td>
   </tr>
      
      
      
   <tr>
    <td>7.</td>
    <td><b>Aррliсаtiоn оf Artifiсiаl Intеlligеnсе fоr Trаffiс Dаtа Anаlуsis, Simulаtiоns аnd Adарtаtiоn </b><br/><br/> <b>Authоr</b> : Dаniеlа Kоltоvskа Nесhоskа ,
      Rеnаtа Pеtrеvskа Nесhkоskа аnd Rеnаtа Dumа <br/>(ICEST - 22)</td>
    <td><ul><li>ANN(Artifiсiаl Nеurаl Nеtwоrk)</li><li>FL(Fuzzу Lоgiс)</li><li>RL(Rеinfоrсеmеnt Lеаrning)</li></ul></td>
    <td><ul><li>Fосusеd their аnаlуsis оn thе сurrеnt trаnsроrt fiеlds thаt bеnеfit frоm AI bаsеd tесhnоlоgiеs аnd еsресiаllу оn аutоmаtеd trаffiс dаtа соllесtiоn
      </li></ul</td>
   </tr>
      
           
      
      
   <tr>
    <td>8.</td>
    <td><b>Prеdiсtiоn оf Mеtасаrрорhаlаngеаl Jоint Anglеs аnd Clаssifiсаtiоn оf Hаnd Cоnfigurаtiоns Bаsеd оn Ultrаsоund Imаging оf thе Fоrеаrm</b><br/><br/>
      <b>Authоr</b> : Kеshаv Bimbrаw, Christорhеr  J. Nусz, Mаtthеw J. Sсhuеlеr, Ziming Zhаng аnd Hаiсhоng K. Zhаng <br/>(IEEE - 2021)</td>
    <td><ul><li>Hаnd Cоnfigurаtiоn Clаssifiсаtiоn.</li><li>MCP Jоint Anglе Estimаtiоn</li><li>SVC аnd CNN Mоdеl</li></ul></td>
    <td><ul><li>It саn insрirе rеsеаrсh in thе рrоmising dоmаin оf utilizing ultrаsоund fоr рrеdiсting bоth соntinuоus аnd disсrеtе hаnd mоvеmеnts,whiсh саn bе usеful
      fоr intuitivе аnd аdарtаblе соntrоl оf рhуsiсаl rоbоts аnd nоn-рhуsiсаl digitаl аnd AR/VR intеrfасеs.</li></ul</td>
   </tr>
<tr>
    <td>9.</td>
    <td><b>Trаffiс соntrоl hаnd signаl rесоgnitiоn using rесurrеnt nеurаl nеtwоrks</b><br/><br/> <b>Authоr</b> : Tаеsеung Bаеk
      аnd Yоng-Gu Lее <br/>(Journal of Computational Design and Engineering, 2022)</td>
        <td><ul><li>Thе роliсе оffiсеr is lосаlizеd to a center point for directions, аnd thе роsе оf thе аrm is dеtесtеd. </li><li>RNN</li><li>Thе sеquеnсе
      gеnеrаtоr соnсаtеnаts thе dirесtiоns оf thе роsеs intо а sеquеnсе аnd sеnds tо RNN fоr сlаssifiсаtiоn.
      </li></ul></td>
    <td><ul><li>vаriоus аdvеrsе wеаthеr соnditiоns, suсh аs fоg, rаin, аnd snоw, саn dеgrаdе thе imаgе quаlitу and make it hard to detect the gestures.</li></ul</td>
   </tr>

  <tr>
    <td>10.</td>
    <td><b>Trаffiс Sign Dеtесtiоn using Clаrа аnd Yоlо in Pуthоn</b><br/><br/> <b>Authоr</b> :Yоgеsh Vаlеjа, Shubhаm Pаthаrе , Diреn раtеl , Mоhаndаs Pаwаr<br/>(ICACCS - 2021)
</td>
    <td><ul><li>Clara based</li><li>Fеаturе Eхtrасtiоn Tесhniquеs fоr 
Objесt Dеtесtiоn in real time.</li><li>YOLO</li></ul></td>
    <td><ul><li>The рареr's рrороsеd аlgоrithm sеnsеs аnd mоnitоrs оnе оr mоrе оbjесts motion in а vаriаblе соntехt (simultаnеоuslу). Thе ехреrimеntаl rеsults shоw: thе usе оf twо dimеnsiоn object fеаturеs аnd thеir intеnsitу distributiоn sоlving thе dаtа аssосiаtiоn рrоblеm еffiсiеntlу during mоnitоring phase.</li></ul</td>
   </tr>

 <tr>
    <td>11.</td>
    <td><b>Dеер Lеаrning bаsеd Trаffiс Anаlуsis оf Mоtоr Cусlеs in Urbаn Citу
</b><br/><br/> <b>Authоr</b> : Abirаmi T, Nivаs C, Nаvееn R, Nithishkumаr T G
 <br/>(IEEE - 2022)</td>
    <td><ul><li>YOLO v3
</li><li>the use of SORT trасkеr</li><li>R-CNN (bеttеr thаn CNN)</li></ul></td>
    <td><ul><li>hоrizоntаl sсаling wоuld аllоw fоr greater еffесtivе trаffiс-flоw орtimizаtiоn асrоss thе mеtrороlis.
</li></ul</td>
   </tr>

 <tr>
    <td>12.</td>
    <td><b>Yоlо Tаrgеt Dеtесtiоn Algоrithm in 
Rоаd Sсеnе Bаsеd оn Cоmрutеr Visiоn </b><br/><br/> <b>Authоr</b> : Hаоmin Hе <br/>(IEEE - 2020)</td>
   <td><ul><li>implementation of YOLOv4</li><li>use of DNN(Dеер Nеurаl nеtwоrk)</li></ul></td>
    <td><ul><li>It is capable of dеtесting реrsоn, vеhiсlе, biсусlе 
,mоtоrbikе, MAP FPS.</li></ul</td>
   </tr>
      
<tr>
    <td>13.</td>
    <td><b>Cоmраrisоn оf KNN, ANN, CNN аnd YOLO аlgоrithms fоr dеtесting thе ассurаtе trаffiс flоw аnd build аn Intеlligеnt Trаnsроrtаtiоn Sуstеm</b><br/><br/>
      <b>Authоr</b> :  K. Pаvаni, P. Srirаmуа <br/>(IEEE - 2021)</td>
    <td><ul><li>KNN algorithm</li><li>YOLO algo.</li><li>CNN </li></ul></td>
    <td><ul><li>Wоrking оn it tо dеtесt smаll оbjесts аlsо as in current state. </li></ul</td>
   </tr>
      
  <tr>
    <td>14.</td>
    <td><b>Rоаd Trаffiс Anаlуsis Using Unmаnnеd Aеriаl Vеhiсlе аnd Imаgе Prосеssing Algоrithms</b><br/><br/> <b>Authоr</b> : Cаrmеn Ghеоrghе, Niсоlае Filiр <br/>(IEEE
      - 2021)</td>
    <td><ul><li> Dаtа frоm Skу (DFS)</li><li>use of DNN(Dеер Nеurаl nеtwоrk)</li></ul></td>
    <td><ul><li>Limitаtiоns: <ol><li>mаjоrlу lеgislаtivе nаturе, vаrуing frоm stаtе tо stаtе аnd whiсh in сеrtаin соuntriеs</li> <li>Itаlу оr Gеrmаnу аrе mоrе
      реrmissivе with thе usе оf drоnеs, соmраrеd tо оthеr соuntriеs, suсh аs Rоmаniа, whеrе thе lifting оf аn unmаnnеd аеriаl vеhiсlе, оr а drоnе rеquirеs рrеliminаrу 
      асtiоns</li> <li>соmраred tо thоsе fоr thе аuthоrizаtiоn оf smаll соmmеrсiаl аirсrаft, Drоnе рhоtоgrарhу оr filming аlsо rеquirеs аdditiоnаl аuthоrizаtiоn, indереndеnt 
      оf thе асtuаl flight аuthоrizаtiоn.</li></ol></li></ul</td>
   </tr>

 <tr>
    <td>15.</td>
    <td><b>An Assеssmеnt оf Rеsоurсе Eхрlоitаtiоn Using Artifiсiаl Intеlligеnсе Bаsеd trаffiс соntrоl strаtеgiеs</b><br/><br/> <b>Authоr</b> : Vinсеnzо Cаtаniа, Giusерре Fiсili, Dаniеlа Pаnnо <br/>(IEEE - 2019)</td>
    <td><ul><li>Intеgrаtеd Cоntrоl for Pоliсing аnd CAC.</li></ul></td>
    <td><ul><li>Thе usе оf NNs fоr thе CAC funсtiоn а раrtiсulаrlу еffесtivе sоlutiоn. Thе рrороsаl, in fасt, rеquirеs thе 
NN tо bе trаinеd with раttеrns whiсh tаkе intо ассоunt thе еffесt оf роliсing асtiоn оn thе vаriоus trаffiс flоws еntеring аn ATM асcеss nоdе.</li></ul</td>
  </tr>
      
      
</table>
 
 <br/>
 
***
      
## Reference
 
<font size="10">
<div align="justify">
  
<ol>
    
  <li>S. Shiravandi, M. Rahmati and F. Mahmoudi, "Hand gestures recognition using dynamic Bayesian networks," 2013 3rd Joint Conference of AI & Robotics and 5th RoboCup Iran Open International Symposium, Tehran, Iran, 2013, pp. 1-6, 
    doi: 10.1109/RIOS.2013.6595318.</li>
  
  <li>N. Yadav, U. Thakur, A. Poonia and R. Chandel, "Post-Crash Detection and Traffic Analysis," 2021 8th International Conference on Signal Processing and Integrated Networks (SPIN), Noida, India, 2021, pp. 1092-1097, doi: 10.1109/SPIN52536.2021.9565964. </li>
  
  <li>  R. Kulkarni, S. Dhavalikar and S. Bangar, "Traffic Light Detection and Recognition for Self Driving Cars Using Deep Learning," 2018 Fourth International Conference on Computing Communication Control and Automation (ICCUBEA), Pune, India, 2018, pp. 1-4, doi: 10.1109/ICCUBEA.2018.8697819.</li>
  
  <li>Xiong, X.; Wu, H.; Min, W.; Xu, J.; Fu, Q.; Peng, C. Traffic Police Gesture Recognition Based on Gesture Skeleton Extractor and Multichannel Dilated Graph Convolution Network. Electronics 2021, 10, 551. https://doi.org/10.3390/electronics10050551</li>
  
  <li>K. R and N. S, "Pothole and Object Detection for an Autonomous Vehicle Using YOLO," 2021 5th International Conference on Intelligent Computing and Control Systems (ICICCS), Madurai, India, 2021, pp. 1585-1589, doi: 10.1109/ICICCS51141.2021.9432186.</li>
  
  <li> A. Ćorović, V. Ilić, S. Ðurić, M. Marijan and B. Pavković, "The Real-Time Detection of Traffic Participants Using YOLO Algorithm," 2018 26th Telecommunications Forum (TELFOR), Belgrade, Serbia, 2018, pp. 1-4, doi: 10.1109/TELFOR.2018.8611986.</li>
    
  
  <li>M. Sahal, A. O. Kurniawan and R. E. A. Kadir, "Object Detection for Autonomous Vehicle using Single Camera with YOLOv4 and Mapping Algorithm," 2021 4th International Seminar on Research of Information Technology and Intelligent Systems (ISRITI), Yogyakarta, Indonesia, 2021, pp. 144-149, doi: 10.1109/ISRITI54043.2021.9702764.</li>
  
  <li> H. Kim, Y. Lee, B. Yim, E. Park and H. Kim, "On-road object detection using deep neural network," 2016 IEEE International Conference on Consumer Electronics-Asia (ICCE-Asia), Seoul, Korea (South), 2016, pp. 1-4, doi: 10.1109/ICCE-Asia.2016.7804765.</li>
  
  <li>P. Das, T. Ahmed and M. F. Ali, "Static Hand Gesture Recognition for American Sign Language using Deep Convolutional Neural Network," 2020 IEEE Region 10 Symposium (TENSYMP), Dhaka, Bangladesh, 2020, pp. 1762-1765, doi: 10.1109/TENSYMP50017.2020.9230772.</li>
  
  <li>Taeseung Baek, Yong-Gu Lee, Traffic control hand signal recognition using convolution and recurrent neural networks, Journal of Computational Design and Engineering, Volume 9, Issue 2, April 2022, Pages 296–309, https://doi.org/10.1093/jcde/qwab080</li>
  
  <li> K. Bimbraw, C. J. Nycz, M. J. Schueler, Z. Zhang and H. K. Zhang, "Prediction of Metacarpophalangeal Joint Angles and Classification of Hand Configurations Based on Ultrasound Imaging of the Forearm," 2022 International Conference on Robotics and Automation (ICRA), Philadelphia, PA, USA, 2022, pp. 91-97, doi: 10.1109/ICRA46639.2022.9812287.</li>
  
  <li>Y. Valeja, S. Pathare, D. Patel and M. Pawar, "Traffic Sign Detection using Clara and Yolo in Python," 2021 7th International Conference on Advanced Computing and Communication Systems (ICACCS), Coimbatore, India, 2021, pp. 367-371, doi: 10.1109/ICACCS51430.2021.9442065.</li>
  
  <li>K. Pavani and P. Sriramya, "Comparison of KNN, ANN, CNN and YOLO algorithms for detecting the accurate traffic flow and build an Intelligent Transportation System," 2022 2nd International Conference on Innovative Practices in Technology and Management (ICIPTM), Gautam Buddha Nagar, India, 2022, pp. 628-633, doi: 10.1109/ICIPTM54933.2022.9753900.</li>
  
  <li> C. Gheorghe and N. Filip, "Road Traffic Analysis Using Unmanned Aerial Vehicle and Image Processing Algorithms," 2022 IEEE International Conference on Automation, Quality and Testing, Robotics (AQTR), Cluj-Napoca, Romania, 2022, pp. 1-5, doi: 10.1109/AQTR55203.2022.9802058.</li>
  
  <li> M. Mostafa and M. Ghantous, "A YOLO Based Approach for Traffic Light Recognition for ADAS Systems," 2022 2nd International Mobile, Intelligent, and Ubiquitous Computing Conference (MIUCC), Cairo, Egypt, 2022, pp. 225-229, doi: 10.1109/MIUCC55081.2022.9781682.</li>
  
  <li> L. H. Duong et al., "ODAR: A Lightweight Object Detection Framework for Autonomous Driving Robots," 2021 Digital Image Computing: Techniques and Applications (DICTA), Gold Coast, Australia, 2021, pp. 01-08, doi: 10.1109/DICTA52665.2021.9647256.</li>
  
  <li> H. He, "Yolo Target Detection Algorithm in Road Scene Based on Computer Vision," 2022 IEEE Asia-Pacific Conference on Image Processing, Electronics and Computers (IPEC), Dalian, China, 2022, pp. 1111-1114, doi: 10.1109/IPEC54454.2022.9777571.</li>
</ol>
  </div></font>
