# Cell_counting_Group1
Codes and explanation to codes used in the cell counting project

### Aim

Automation of the cell counting task will save time spent on manually counting cells

Build a machine learning model with Convolutional Neural Networks to count cells from microscope images

Input an image and let the model output the number of cells

### Pre-train
Use ResNet-18 and VGG16 as baseline models to run experiments on the training dataset.

VGG16 (Adapted from https://github.com/ISU-NRT-D4/cell-analysis/tree/main/IDCIA/CNN%20Reg) - Train/Val/Test split = 70/20/10

ResNet-18 - Train/test = 80/20

Evaluation metrics = MAE (mean absolute error is a measure of errors between paired observations i.e., Prediction - True value) MAE = (1/n) Σ(i=1 to n) |y_i – ŷ_i|

### Model used
@inproceedings{m_Ranjan-etal-CVPR21,
  author = {Viresh Ranjan and Udbhav Sharma and Thu Nguyen and Minh Hoai},
  title = {Learning To Count Everything},
  year = {2021},
  booktitle = {Proceedings of the {IEEE/CVF} Conference on Computer Vision and Pattern Recognition (CVPR)},
}
https://github.com/cvlab-stonybrook/LearningToCountEverything

Steps followed:

Pre-training model on counting dataset FSC-147 with 147 classes 

Annotate 3-5 bounding boxes for each cell image

Generate heatmap

Prediction - sum of the densities

### Evaluation metrics
MAE = (1/n) Σ(i=1 to n) |y_i – ŷ_i| (mean absolute error is a measure of errors between paired observations i.e., Prediction - True value)

RMSE = √((1/n) Σ(i=1 to n) * (y_i – ŷ_i)^2) (The root mean square error (RMSE) measures the average difference between a statistical model's predicted values and the actual values)

ACP = (1/n) * 100 Σ(i=1 to n) [[ |y_i – ŷ_i| </= 0.05 * y_i ]] (Acceptable Error Count Percent (ACP) - percentage of images whose predicted count is within a 5% difference from the true count by the domain expert)
