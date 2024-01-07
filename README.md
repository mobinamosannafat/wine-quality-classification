# Wine Quality Classification

In the context of this project, our foremost goal is to use information on various wine samples, systematically categorized as either good or bad quality. 


## Dataset
Input Variables (Physicochemical Tests or x) in our problem are:

1.	Fixed Acidity
2.	Volatile Acidity
3.	Citric Acid
4.	Residual Sugar
5.	Chlorides
6.	Free Sulfur Dioxide
7.	Total Sulphur Dioxide
8.	Density
9.	pH
10.	Sulphates
11.	Alcohol

The output variable (Sensory Data or y) is Quality ('good' and 'bad'). These attributes represent features obtained from physicochemical tests, with the target variable being the wine quality derived from sensory data. The primary objective is to leverage this dataset for training and evaluating classification models to effectively categorize wines as either 'good' or 'bad' based on these attributes. Table 1 shows several samples of this dataset and Figure 1 shows the Correlation between different features of wines.

To facilitate the classification task, the categorical labels of the 'Quality' are encoded. Specifically, the label 'bad' is mapped to 0, and 'good' is mapped to 1. This numerical representation allows for seamless integration with classification algorithms.

![image](https://github.com/mobinamosannafat/wine-quality-classification/assets/52583295/c9cfc059-fe40-4c01-97af-c683a4261f5b)

![image](https://github.com/mobinamosannafat/wine-quality-classification/assets/52583295/58586654-3631-4e26-8bc8-000d7a681f35)


The dataset is initially split into three categories: train, test, and validate, with proportions set at 0.6, 0.2, and 0.2, respectively. Subsequently, normalization is applied to all three subsets.\\


## Classification Models

In our project, we implement three distinct classification models to address the wine quality categorization task. Each model offers unique characteristics suited for binary classification:

### Logistic Regression

	Type: A linear model designed for binary classification.
	Idea: Logistic Regression models the probability that an instance belongs to a particular class. It employs the logistic function to map a linear combination of input features to a value between 0 and 1.
	Hypotheses Class: logistic function (sigmoid function).
	Training/Learning Objective: Minimizing the logistic loss (binary cross-entropy) during training.


Support Vector Machines (SVM)

	Type: Linear and non-linear classifier.
	Idea: SVM aims to find a hyperplane that optimally separates data points of different classes in a high-dimensional space. It prioritizes maximizing the margin between classes, and for non-linear problems, it can utilize kernel functions to map features into a higher-dimensional space.
	Hypotheses Class: Decision function.
	Training/Learning Objective: Maximizing the margin between different classes while ensuring correct classification.


2-Layer Perceptron

	Type: Fully connected feedforward neural network tailored for binary classification.
	Idea: A neural network with one hidden layer processes input features through weights and biases, applying activation functions to introduce non-linearity. This architecture enables the model to capture more intricate relationships in the data compared to linear models.
	Hypotheses Class: Composition of multiple layers with non-linear activation functions.
	Training/Learning Objective: minimizing the cross-entropy loss during training.


The measure of success for all classification methods in this project is validation and test accuracy.

## Hyper Parameter Tuning

In the hyperparameter tuning phase, we systematically explored the impact of different values for the learning rate and batch size on each classification model based on the validation accuracy. Specifically, we considered three learning rates—0.001, 0.01, and 0.1—and three batch sizes—10, 100, and 1000.

The purpose of this exploration is to identify the combination of learning rate and batch size that optimizes the performance of each model. By varying these hyperparameters, we aim to fine-tune the training process and enhance the overall effectiveness of the models in classifying wine quality.

In the subsequent analysis, we will report and examine the model performance under each combination of learning rate and batch size, providing insights into the hyperparameter settings that yield the best results for our specific classification task.


## Results


Due to constraints in hardware resources and time limitations, we were restricted to training the models for up to 100,000 epochs. Notably, the deep neural network (DNN) model did not converge within this specified epoch limit. It is crucial to acknowledge that the final outcome may differ if we had the capacity to train the model for an extended number of epochs. Consequently, the subsequent reports and conclusions are based on the outcomes obtained within the given epoch constraint.

Training Details:

All classification models underwent training for a substantial 10,000 epochs, providing an extensive learning process for each model.


Validation Performance Analysis

Table 2 presents the validation performance (accuracy) of each model across different combinations of learning rates and batch sizes. Notably, the results indicate optimal hyperparameter settings for maximizing performance.

- Optimal Learning Rates:
  1. Logistic Regression: 0.01
  2. Support Vector Machines (SVM): 0.1
  3. Neural Networks: 0.1

- Optimal Batch Sizes:
  1. Logistic Regression: 10
  2. Support Vector Machines (SVM): 100
  3. Neural Networks: 1000


Performance Comparison:

Figure 2 visually represents the comparative performance of Logistic Regression, SVM, and Neural Networks for wine quality classification. The observed trend aligns with the ordered mention of models, indicating that Deep Neural Networks, Logistic Regression, and SVM perform better in the same order they were mentioned. 

![image](https://github.com/mobinamosannafat/wine-quality-classification/assets/52583295/cfa8a2d1-89c5-4c25-b212-e66fd2ff9c0d)
![image](https://github.com/mobinamosannafat/wine-quality-classification/assets/52583295/b19f4106-fcb4-49eb-be4b-19a3329e51d6)



