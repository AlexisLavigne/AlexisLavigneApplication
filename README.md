# ML Assignments Repository
This repository contains two assignments showcasing different machine learning techniques. Each assignment demonstrates the use of Python for data analysis, visualization, and model training.

## 1. Flower Classifier Transfer Learning
In this assignment, a transfer learning approach was used to classify flowers into five categories. 
Key highlights include:

- Model architecture: Used the MobileNetV2 model, pre-trained on the ImageNet dataset, as the base architecture
- Transfer learning: Adapted MobileNetV2 to recognize flower types from the small_flower_dataset
- Optimization: Explored the impact of learning rate and momentum on model performance using the Stochastic Gradient Descent (SGD) optimizer

## 2. ML Linear Regression
This assignment combines data collection, analysis, and machine learning to explore the relationship between city populations and museum visitor numbers. 
Key highlights include:

- Data retrieval:
Scraped a list of the most visited museums (from Wikipedia)and fetched city population data using a dataset from Kaggle
- Data integration:
Programmatically merged the museum visitor data with the corresponding city population
- Linear regression models:
Built linear regression models to analyze the correlation between city population and museum attendance
- Visualization:
Plotted the regression results and  relevant visualizations to interpret the data and findings

## Technologies used
- Python
- TensorFlow/Keras (for the flower classifier)
- Scikit-learn (for linear regression)
- Matplotlib and Plotly (for visualizations)
- Pandas and NumPy (for data manipulations)