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

  Type: Linear model designed for binary classification.
  Idea: Logistic Regression models the probability that an instance belongs to a particular class. It employs the logistic function to map a linear combination of input features to a value between 0 and 1.
Hypotheses Class: logistic function (sigmoid function) which can be represented as $( h_θ (x)=  1/(1+ e^(-(θ^T x)) )$.
Training/Learning Objective: Minimizing the logistic loss (binary cross-entropy) during training, which can be represented as $(J(θ)= -1/M ∑_(i=1)^m▒〖[y^((i) )  log⁡(h_θ (x^((i) ) ))+(1-y^((i)))log⁡〖(1-h_θ (x^((i) )))〗])$.

Support Vector Machines (SVM)

	Type: Linear and non-linear classifier.
	Idea: SVM aims to find a hyperplane that optimally separates data points of different classes in a high-dimensional space. It prioritizes maximizing the margin between classes, and for non-linear problems, it can utilize kernel functions to map features into a higher-dimensional space.
	Hypotheses Class: Decision function which can be represented as f(x)= 〖argmax〗_k (〈w_(k,),x〉  + b_k).
	Training/Learning Objective: Maximizing the margin between different classes while ensuring correct classification, which can be represented as
 〖min〗_(w,b)   1/2  ‖w‖^2+C∑_(i=1)^m▒〖max⁡(0,1-y_i (〈w,x_i 〉+b))〗.


2-Layer Perceptron

	Type: Fully connected feedforward neural network tailored for binary classification.
	Idea: A neural network with one hidden layer processes input features through weights and biases, applying activation functions to introduce non-linearity. This architecture enables the model to capture more intricate relationships in the data compared to linear models.
	Hypotheses Class: Composition of multiple layers with non-linear activation functions which can be represented as h(x)= σ(W_2  .σ(W_1  .x+ b_1 )+ b_2) .
	Training/Learning Objective: minimizing the cross-entropy loss during training, which can be represented as
 J(W,b)= -1/M ∑_(i=1)^m▒〖[y^((i) )  log⁡(h_θ (x^((i) ) ))+(1-y^((i)))log⁡〖(1-h_θ (x^((i) )))〗]〗


The measure of success for all classification methods in this project is validation and test accuracy.


