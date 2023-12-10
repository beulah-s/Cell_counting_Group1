## Cell_counting_Group1
Codes and explanation to codes used in the cell counting project

### Aim

Automation of the cell counting task will save time spent on manually counting cells

Build a machine learning model with Convolutional Neural Networks to count cells from microscope images

Input an image and let the model output the number of cells

### Baseline models:

Used ResNet-18 and VGG19 as baseline models to run experiments on the training dataset.

Train/test = 80/20

Evaluation metrics = MAE (mean absolute error is a measure of errors between paired observations i.e., Prediction - True value) MAE = (1/n) Σ(i=1 to n) |y_i – ŷ_i|

### Models used:
### 1. Viresh Ranjan, Udbhav Sharma, Thu Nguyen, and Minh Hoai. 2021. Learning To Count Everything. Retrieved November 16, 2023from http://arxiv.org/abs/2104.08391

(codes adapted from https://github.com/cvlab-stonybrook/LearningToCountEverything)

1. Cellcount.py
2. Baseline.py
3. Resize.py

Steps followed:

Pre-training model on counting dataset FSC-147 with 147 classes 

Annotate 3-5 bounding boxes for each cell image

Generate heatmap

Prediction - sum of the densities

### 2. VGG16 (Adapted from https://github.com/ISU-NRT-D4/cell-analysis/tree/main/IDCIA/CNN%20Reg)

Train/Test/Val split = 70/20/10 (or) 175:50:25 images

Tried various parameters for learning rate and batch size. Tested with the test images from training data.

Finally used the model with learning rate 1e-3 and batch size 32.

The codes used are in IDCIA clone directory.

### Evaluation metrics

1. MAE = (1/n) Σ(i=1 to n) |y_i – ŷ_i| (mean absolute error is a measure of errors between paired observations i.e., Prediction - True value)

2. ACP = (1/n) * 100 Σ(i=1 to n) [[ |y_i – ŷ_i| </= 0.05 * y_i ]] (Acceptable Error Count Percent (ACP) - percentage of images whose predicted count is within a 5% difference from the true count by the domain expert)
