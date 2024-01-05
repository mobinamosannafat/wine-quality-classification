import matplotlib 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def dataset_visualization(wine):

    print("\n\n\n ######################### Data Visualization ############################")

    print("\n\n")

    print(wine.head())

    print("\n\n")

    print(wine.info())

    print("\n\n")

    print(wine.describe())

    print("\n\n")

    print(wine['quality'].value_counts())

    print("\n\n")

    # corr = wine.corr()
    # plt.figure(figsize=(6,6))
    # sns.heatmap(corr, fmt='.1f',cmap='Reds', annot=True)
    # plt.savefig('dataset.png')


def preprocess(wine):

    X = wine.drop('quality', axis= 1)
    X = X.to_numpy()
    old_y = wine['quality']
    y = [0 if item == 'bad' else 1 for item in old_y]
    y = np.array(y)


    #Splitting

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1) # 0.25 x 0.8 = 0.2



    #Normalization

    norm = MinMaxScaler()
    norm_fit = norm.fit(X_train)
    new_xtrain = norm_fit.transform(X_train)
    new_xval = norm_fit.transform(X_val)
    new_xtest = norm_fit.transform(X_test)

    return new_xtrain, new_xval, new_xtest, y_train, y_val, y_test



class LogisticRegression:
    def __init__(self, N_class, alpha=0.01, batch_size=100, max_epoch=100000, decay=0.):
        self.N_class = N_class
        self.alpha = alpha
        self.batch_size = batch_size
        self.max_epoch = max_epoch
        self.decay = decay
        self.W = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def predict(self, X, t=None):
        z = np.dot(X, self.W)
        y = self.sigmoid(z)

        if t is not None:
            t_one_hot = np.eye(self.N_class)[t.flatten()]
            loss = -np.sum(t_one_hot * np.log(y + 1e-16)) / X.shape[0]
            t_hat = np.argmax(y, axis=1)
            acc = np.mean(t_hat == t.flatten())
            return y, t_hat, loss, acc
        else:
            return y

    def train(self, X_train, t_train, X_val, t_val):
        N_train = X_train.shape[0]
        d = X_train.shape[1] - 1

        self.W = np.random.randn(d+1, self.N_class)
        train_losses = []
        valid_accs = []
        W_best = None
        acc_best = 0.0
        epoch_best = 0

        for epoch in range(self.max_epoch):
            for i in range(0, N_train, self.batch_size):
                X_batch = X_train[i:i+self.batch_size]
                t_batch = t_train[i:i+self.batch_size]

                y_batch, _, _, _ = self.predict(X_batch, t_batch)  # y, t_hat, loss, acc
                grad = np.dot(X_batch.T, (y_batch - np.eye(self.N_class)[t_batch.flatten()])) / X_batch.shape[0]
                self.W -= self.alpha * (grad + self.decay * self.W)

            _, _, train_loss, _ = self.predict(X_train, t_train)  # y, t_hat, loss, acc
            train_losses.append(train_loss)

            _, _, _, val_acc = self.predict(X_val, t_val)  # y, t_hat, loss, acc
            valid_accs.append(val_acc)

            if val_acc > acc_best:
                acc_best = val_acc
                epoch_best = epoch
                W_best = np.copy(self.W)

        return epoch_best, acc_best, W_best, train_losses, valid_accs



class SVMClassifier:
    def __init__(self, N_class, alpha=0.01, batch_size=100, max_epoch=100000, decay=0.):
        self.N_class = N_class
        self.alpha = alpha
        self.batch_size = batch_size
        self.max_epoch = max_epoch
        self.decay = decay
        self.W = None

    def hinge_loss(self, y, t):
        margins = 1 - y * np.eye(self.N_class)[t.flatten()]
        margins[margins < 0] = 0
        loss = np.mean(np.sum(margins, axis=1))
        return loss

    def predict(self, X, t=None):
        scores = np.dot(X, self.W)
        t_hat = np.argmax(scores, axis=1)

        if t is not None:
            loss = self.hinge_loss(scores, t)
            acc = np.mean(t_hat == t.flatten())
            return scores, t_hat, loss, acc
        else:
            return scores

    def train(self, X_train, t_train, X_val, t_val):
        N_train = X_train.shape[0]
        d = X_train.shape[1] - 1

        self.W = np.random.randn(d+1, self.N_class)
        train_losses = []
        valid_accs = []
        W_best = None
        acc_best = 0.0
        epoch_best = 0

        for epoch in range(self.max_epoch):
            for i in range(0, N_train, self.batch_size):
                X_batch = X_train[i:i+self.batch_size]
                t_batch = t_train[i:i+self.batch_size]

                scores = np.dot(X_batch, self.W)
                margins = 1 - scores * np.eye(self.N_class)[t_batch.flatten()]
                grad = -np.dot(X_batch.T, (margins > 0) * np.eye(self.N_class)[t_batch.flatten()]) / X_batch.shape[0]
                self.W -= self.alpha * (grad + self.decay * self.W)

            _, _, train_loss, _ = self.predict(X_train, t_train)
            train_losses.append(train_loss)

            _, _, _, val_acc = self.predict(X_val, t_val)
            valid_accs.append(val_acc)

            if val_acc > acc_best:
                acc_best = val_acc
                epoch_best = epoch
                W_best = np.copy(self.W)

        return epoch_best, acc_best, W_best, train_losses, valid_accs




import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os, sys

class DeepNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, alpha=0.01, batch_size=100, max_epoch=100, decay=0.):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.alpha = alpha
        self.batch_size = batch_size
        self.max_epoch = max_epoch
        self.decay = decay
        self.model = None
        self.val_accuracy_history = []  # Store validation accuracy at each epoch

    def build_model(self):
        model = Sequential()
        model.add(Dense(self.hidden_size, input_dim=self.input_size, activation='relu'))
        model.add(Dense(self.output_size, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model


    def train(self, X_train, y_train, X_val, y_val):
        # Save the original standard output
        original_stdout = sys.stdout

        # Redirect standard output to /dev/null (Linux/Mac) or nul (Windows)
        sys.stdout = open(os.devnull, 'w')

        self.model = self.build_model()

        # Create ValidationAccuracyCallback instance
        val_accuracy_callback = self.ValidationAccuracyCallback(X_val, y_val)

        # Train the model using a separate validation dataset
        history = self.model.fit(
            X_train, y_train,
            epochs=self.max_epoch,
            batch_size=self.batch_size,
            validation_data=(X_val, y_val),
            callbacks=[val_accuracy_callback],
            verbose=0  # Set verbose to 0 to silence output during training
        )

        # Store the list of validation accuracies
        self.val_accuracy_history = val_accuracy_callback.val_accuracy_history

        # Restore standard output
        sys.stdout = original_stdout

        return history


    def evaluate(self, X_test, y_test):
        loss, accuracy = self.model.evaluate(X_test, y_test)
        return loss, accuracy
        #print(f'Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}')

    def get_best_accuracies(self, history):
        best_train_accuracy = max(history.history['accuracy'])
        best_val_accuracy = max(history.history['val_accuracy'])
        # print(f'Best Training Accuracy: {best_train_accuracy:.4f}')
        # print(f'Best Validation Accuracy: {best_val_accuracy:.4f}')
        return best_train_accuracy, best_val_accuracy

    def get_test_accuracy(self, X_test, y_test):
        _, test_accuracy = self.evaluate(X_test, y_test)
        return test_accuracy

   
    def get_train_loss(self, history):
      train_loss_history = history.history['loss']
      final_train_loss = train_loss_history[-1]
      return train_loss_history, final_train_loss

    class ValidationAccuracyCallback(keras.callbacks.Callback):
        def __init__(self, X_val, y_val):
            super().__init__()
            self.X_val = X_val
            self.y_val = y_val
            self.val_accuracy_history = []  # Store validation accuracies at each epoch

        def on_epoch_end(self, epoch, logs=None):
            val_predictions = self.model.predict(self.X_val)
            val_predictions_binary = (val_predictions > 0.5).astype(int)
            val_accuracy = np.mean(val_predictions_binary.flatten() == self.y_val.flatten())
            self.val_accuracy_history.append(val_accuracy)


def main():
    

    wine = pd.read_csv('wine.csv')

    #dataset_visualization(wine) #Uncomment for dataset visualiztion. It's better to observ visulizaton in jupyter notebook

    new_xtrain, new_xval, new_xtest, y_train, y_val, y_test = preprocess(wine)


    learning_rates = [0.001, 0.01, 0.1]
    bach_sizes = [10, 100, 1000]

    N_class = 2
    max_epoch = 10000#100000  # Maximum epoch
    decay = 0. #L2 regularization term



    lr_bests = [0, 0, 0]
    bs_bests = [0, 0, 0]


    lr = [0 for _ in range(5)]
    svm = [0 for _ in range(5)]
    nn = [0 for _ in range(4)]


    for i in range(len(learning_rates)):

      for j in range(len(bach_sizes)):
        print(i+j)


        # --- LogisticRegression ---

        lr_model = LogisticRegression(N_class, learning_rates[i], bach_sizes[j], max_epoch, decay)
        epoch_best_lr, acc_best_lr, W_best_lr, train_losses_lr, valid_accs_lr = lr_model.train(new_xtrain, y_train, new_xval, y_val)

        _, _, _, acc_test_lr = lr_model.predict(new_xtest, y_test)

        if i == 0:
          lr_bests[0] = i
          bs_bests[0] = j
          lr = [epoch_best_lr, acc_best_lr, acc_test_lr, train_losses_lr, valid_accs_lr]

        elif acc_best_lr > lr[1]:
          lr_bests[0] = i
          bs_bests[0] = j
          lr = [epoch_best_lr, acc_best_lr, acc_test_lr, train_losses_lr, valid_accs_lr]


        # --- SVM ---

        svm_model = SVMClassifier(N_class, learning_rates[i], bach_sizes[j], max_epoch, decay)
        epoch_best_svm, acc_best_svm, W_best_svm, train_losses_svm, valid_accs_svm = svm_model.train(new_xtrain, y_train, new_xval, y_val)

        _, _, _, acc_test_svm = svm_model.predict(new_xtest, y_test)


        if i == 0:
          lr_bests[1] = i
          bs_bests[1] = j
          svm = [epoch_best_svm, acc_best_svm, acc_test_svm, train_losses_svm, valid_accs_svm]

        elif acc_best_svm > svm[1]:
          lr_bests[1] = i
          bs_bests[1] = j
          svm = [epoch_best_svm, acc_best_svm, acc_test_svm, train_losses_svm, valid_accs_svm]


        # --- NeuralNetwork ---
        # Create an instance of the DeepNeuralNetwork class
        dnn = DeepNeuralNetwork(input_size=new_xtrain.shape[1], hidden_size=64, output_size=1, alpha=learning_rates[i], batch_size=bach_sizes[j], max_epoch=max_epoch, decay=decay) #nput_size, hidden_size, output_size, alpha=0.01, batch_size=100, max_epoch=100, decay=0.):

        # Train the model
        history = dnn.train(new_xtrain, y_train, new_xval, y_val)

        # Evaluate the model on the test set
        dnn.evaluate(new_xtest, y_test)

        # Get the best training accuracy, best validation accuracy, test accuracy, and train loss
        _, acc_best_nn = dnn.get_best_accuracies(history)
        acc_test_nn = dnn.get_test_accuracy(new_xtest, y_test)
        train_losses_nn, final_train_loss = dnn.get_train_loss(history)
        valid_accs_nn = dnn.val_accuracy_history




        if i == 0:
          lr_bests[2] = i
          bs_bests[2] = j
          nn = [acc_best_nn, acc_test_nn, train_losses_nn, valid_accs_nn]

        elif acc_best_nn > nn[0]:
          lr_bests[2] = i
          bs_bests[2] = j
          nn = [acc_best_nn, acc_test_nn, train_losses_nn, valid_accs_nn] 

        print("\n ------------------- learning rate = " + str(learning_rates[i]) + ", bach size = " + str(bach_sizes[j]) + " ------------------- \n")

        print("Validation performance (accuracy) in that epoch in ** Logistic Regression ** :", acc_best_lr)
        print("Validation performance (accuracy) in that epoch in ** SVM ** :", acc_best_svm)
        print("Validation performance (accuracy) in that epoch in ** Deep Neural Network ** :", acc_best_nn)

        print("\n\n")

        print("Test performance (accuracy) in that epoch in ** Logistic Regression ** :", acc_test_lr)
        print("Test performance (accuracy) in that epoch in ** SVM ** :", acc_test_svm)
        print("Test performance (accuracy) in that epoch in ** Deep Neural Network ** :", acc_test_nn)

        print("\n\n")


    print("\n\n\n\n*************************************************************************************************  \n")
    print("Best Learning Rate for LogisticRegression model is : " + str(learning_rates[lr_bests[0]]) + ", and best Bach Size for it is:  " + str(bach_sizes[bs_bests[0]]) +
          "\nBest Learning Rate for SVM model is : " + str(learning_rates[lr_bests[1]]) + ", and best Bach Size for it is:  " + str(bach_sizes[bs_bests[1]]) +
          "\nBest Learning Rate for Deep Neural Network model is : " + str(learning_rates[lr_bests[2]]) + ", and best Bach Size for it is:  " + str(bach_sizes[bs_bests[2]]))
    print("\n*************************************************************************************************  \n\n")

    print("Validation performance (accuracy) in that epoch in ** Logistic Regression ** with learning rate = " + str(learning_rates[lr_bests[0]]) + ' and bach size = ' + str(bach_sizes[bs_bests[0]]) + ' is << ' + str(lr[1]) + ' >>')
    print("Test performance (accuracy) in that epoch in ** Logistic Regression ** with learning rate = " + str(learning_rates[lr_bests[0]]) + ' and bach size = ' + str(bach_sizes[bs_bests[0]]) + ' is << ' + str(lr[2]) + ' >>')
    print("\n\n")

    print("Validation performance (accuracy) in that epoch in ** SVM ** with learning rate = " + str(learning_rates[lr_bests[1]]) + " and bach size = " + str(bach_sizes[bs_bests[1]]) + ' is << ' + str(svm[1]) + ' >>')
    print("Test performance (accuracy) in that epoch in ** SVM ** with learning rate = " + str(learning_rates[lr_bests[1]]) + " and bach size = " + str(bach_sizes[bs_bests[1]]) + ' is << ' + str(svm[2]) + ' >>')
    print("\n\n")

    print("Validation performance (accuracy) in that epoch in ** Deep Neural Network ** with learning rate = " + str(learning_rates[lr_bests[2]]) + " and bach size = " + str(bach_sizes[bs_bests[2]]) + " is << " + str(nn[0]) + ' >>')
    print("Test performance (accuracy) in that epoch in ** Deep Neural Network with learning rate = " + str(learning_rates[lr_bests[2]]) + " and bach size = " + str(bach_sizes[bs_bests[2]]) + " is << " + str(nn[1]) + ' >>')
    print("\n\n")


    # Plotting
    plt.figure(figsize=(24, 12))

    plt.subplot(1, 2, 1)
    plt.plot(lr[3], label='LogisticRegression (Learning Rate = ' + str(learning_rates[lr_bests[0]]) + ', Bach Size = ' + str(bach_sizes[bs_bests[0]]) + ')')
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss (Cross Entropy)')
    plt.legend()

    plt.subplot(1, 2, 1)
    plt.plot(svm[3], label='SVM (Learning Rate = ' + str(learning_rates[lr_bests[1]]) + ", Bach Size = " + str(bach_sizes[bs_bests[1]]) + ')')
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss (Cross Entropy)')
    plt.legend()

    plt.subplot(1, 2, 1)
    plt.plot(nn[2], label='Deep Neural Network (Learning Rate = ' + str(learning_rates[lr_bests[0]])+ ", Bach Size = " + str(bach_sizes[bs_bests[2]]) + ')')
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss (Cross Entropy)')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(lr[4], label='LogisticRegression (Learning Rate = ' + str(learning_rates[lr_bests[0]]) + ', Bach Size = ' + str(bach_sizes[bs_bests[0]]) + ')')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(svm[4], label='SVM (Learning Rate = ' + str(learning_rates[lr_bests[1]]) + ", Bach Size = " + str(bach_sizes[bs_bests[1]]) + ')')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(nn[3], label='Deep Neural Network (Learning Rate = ' + str(learning_rates[lr_bests[0]]) + ", Bach Size = " + str(bach_sizes[bs_bests[2]]) + ')')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy')
    plt.legend()

    plt.suptitle('Training Loss and Validation Accuracy.png')
    plt.savefig('Training_Loss_and_Validation_Accuracy.png')
    plt.show()






if __name__ == "__main__":
    main()