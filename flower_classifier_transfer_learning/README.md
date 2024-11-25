# Flower Classifier Transfer Learning
In this assignment, I used the MobileNetV2 architecture, a neural network model often
used in transfer learning scenarios, particularly for image recognition tasks. Transfer learning
is a method where a pre-trained model, like MobileNetV2, is adapted to a new similar task.
The ModileNetV2 model is pre-trained on the ImageNet dataset and I adapted it in
order to recognize flower types from the ‘small_flower_dataset’ that are separated into five
categories. The main parameters that were investigated in this assignment are the learning
rate and the momentum used by the Stochastic Gradient Descent (SGD) optimizer during
the model’s training.