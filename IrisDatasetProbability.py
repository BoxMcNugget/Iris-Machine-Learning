#############################################
###                                       ###
### Used geeks for geeks website for help ###
###                                       ###
#############################################
import numpy as np
import pandas as pd

#################################################
### import slickit learn for machine learning ###
#################################################
from sklearn.model_selection import train_test_split


#########################
### Load Iris dataset ###
#########################
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
data = pd.read_csv(url, header=None, names=columns)


#######################################
### Select the feature for training ###
#######################################
feature = 'sepal_length'
X = data[[feature]].values
y = data['class'].values


################################################################################
### Split data for machine learning model, 60% for training, 40% for testing ###
################################################################################
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)


######################################################################
### Calculate mean and variance for each class in the training set ###
######################################################################
# uninque finds the unique values in the array y_train and initilalize a new  dictionary
classes = np.unique(y_train)
mean_var_per_class = {}

# for the classes in the dataset with the chosen feature
for cls in classes:
    # get the samples for that class
    X_class = X_train[y_train == cls]
    # then calculate the mean and variuance for that class
    mean_var_per_class[cls] = {
        'mean': np.mean(X_class),
        'var': np.var(X_class)
    }


#############################################
### Gaussian probability density function ###
#############################################
def gaussian_density(x, mean, stdDev):

#                     |  numerator     |   |  Denominator  |
    exponent = np.exp(-((x - mean) ** 2) / (2 * stdDev ** 2))

#          |Numerator|           Denominator        |  Exponent
    return (1         / stdDev * np.sqrt(2 * np.pi)) * exponent


###################################
### Predict class for test data ###
###################################
def predict(X_test):
    predictions = []
    # for every sample of data in the testing portion of the data
    for x in X_test:
        class_probabilities = {}
        # and for the classes in the dataset for the testing portion
        for cls in classes:
            # set the mean and variance
            mean = mean_var_per_class[cls]['mean']
            var = mean_var_per_class[cls]['var']
            # calculate the gaussian density using x(currentr data), mean, and variance
            class_probabilities[cls] = gaussian_density(x, mean, var)
        # add it to the predictions dictionary
        predictions.append(max(class_probabilities, key=class_probabilities.get))
    return predictions

# Perform predictions on the test set
y_pred = predict(X_test)


############################
### Accuracy calculation ###
############################
# calcuylate the accuracyt  by taking the predictions equal to the testing data 
# and devioding it by the testing data
accuracy = np.sum(y_pred == y_test) / len(y_test)
# tyhen multiply it by 100 to make it a useable number and set it to one decimal place
print(f"Accuracy: {accuracy * 100:.2f}%")


#####################################
### Report correct classifications###
#####################################
print("\nCorrectly Classified Samples:")
# for every value in y_test
for i in range(len(y_test)):
    # if the value equals the prediction then print the sample, predicted value,
    # actual value, and then the feature value
    if y_test[i] == y_pred[i]:
        print(f"Sample {i}: Predicted={y_pred[i]}, True={y_test[i]}, Feature Value={X_test[i][0]}")


#######################################
### Report incorrect classifications###
#######################################
print("\nIncorrectly Classified Samples:")
# for every value in y_test
for i in range(len(y_test)):
    # if the value does not equal the prediction then print the sample, predicted value,
    # actual value, and then the feature value
    if y_test[i] != y_pred[i]:
        print(f"Sample {i}: Predicted={y_pred[i]}, True={y_test[i]}, Feature Value={X_test[i][0]}")