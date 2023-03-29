<div align="center">
  <h1>Proposed Title: Traffic Police Hand Gesture Detection </h1>
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

<h2>Abstract</h2>
<ul>
  <li>Gesture recognition is one of the most difficult challenges in computer vision. While recognising traffic police hand signals, one must take into account the speed and dependability of the instructing signal. It is significantly easier to extract the three-dimensional coordinates of skeletons when depth information is given with the photos. Here, we present a method for detecting hand signals that does not rely on skeletons. Instead of skeletons, we employ basic object detectors that have been trained to respond to hand signals. 
<li>Autonomous vehicles require traffic police gesture recognition. Current traffic police gesture identification systems frequently extract pixel-level characteristics from RGB photos, which are incoherent owing to the absence of gesture skeleton features and can result in erroneous results. Existing object detection algorithms enable the detection of automobiles, trees, people, bicycles, animals, and so forth (YOLO).

<li>In this project, we will employ the Convolutional Neural Network (CNN) approach (Deep Learning) to recognise traffic police hand signals. As there are no acceptable datasets available, we shall attempt to generate our own. 
Mediapipe will be utilised in its development.
</ul>

<h2>Introduction</h2>
  <ul>
    <li>Artificial intelligence has been implemented in various industries, particularly in computer vision, improving object detection and classification. Traffic signals play a significant role in traffic flow and safety, and AI can be used to enhance their effectiveness.
    <li>Recognizing traffic police gestures is challenging due to the lack of interpretable features from RGB images and interference from complex environments. Gesture skeleton extractor (GSE) can be used to extract interpretable skeleton coordinate information, eliminating background interference. However, existing deep learning methods are not suitable for handling gesture skeleton features, requiring the development of new methods.
    <li>Previous studies faced limitations due to the reliance on handcrafted components or pixel-level information, resulting in poor recognition performance and weak generalization capability. Autonomous vehicles must also recognize hand signals from traffic controllers in real-time, which can be challenging due to the high speed of the vehicles. Many skeleton-based methods require high computational load or expensive devices, making them unsuitable for real-time processing.
    <li>Prior approaches used skeleton-based action recognition to identify hand signals in videos, requiring preprocessing and limiting generalizability to real-world problems. Accurate detection of hand signals requires distinguishing between intentional and non-intentional signals. The ability to differentiate situations affects the accuracy of detection.
    <li>The paper proposes a new method to recognize hand signals from raw video streams without preprocessing, overcoming the challenges of previous methods that relied on skeleton-based action recognition. The proposed method utilizes an attention-based spatial-temporal graph convolutional network (ASTGCN) that achieved higher accuracy than previous methods and can distinguish between intentional and non-intentional signals. The potential applications of this method include traffic management, public safety, and military operations. The paper highlights the significance of deep learning in recognizing hand signals in real-world environments.
  </ul>

<h2>Tools and Software: Implementation</h2>
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
  
  Example of Dataset:
  </br><p>
  <img src="https://user-images.githubusercontent.com/79594169/228626817-b2d97684-f687-4b62-8785-bff8c4fabd14.jpg">
  </br>
  
  Dataset with overlapping model:
  </br><p>
  <img src="https://user-images.githubusercontent.com/79594169/228628410-c78704ba-5abf-45f8-91e8-ed993d9568eb.jpg">
  </br>
  
  <p>By identifying the 32 description points on the dataset image we are able to identify the angles and position the subject is forming which helps determine its gesture.</br>
  </br>
# Other Applications:
      
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