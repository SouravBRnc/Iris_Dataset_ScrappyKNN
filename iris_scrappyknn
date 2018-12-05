from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.spatial import distance
# from sklearn.neighbors import KNeighborsClassifier
# we wont use KNeighborsClassifier directly.. we will create our own version of it..


# to calculate the euclidean distance between two rows
def euclid(a, b):
    return distance.euclidean(a, b)


# The class defined to make classifier
class ScrappyKNN():
    # function to load train data
    def fit(self, X_train, Y_train):
        self.X_train = X_train
        self.Y_train = Y_train

    # function to make predictions
    def predict(self, X_test):
        predictions = []
        # predictions is the list for predicted values
        for row in X_test:
            # for each row in the test set find the row closest to that row and return the label(response)
            label = self.closest(row)
            # appending prediction to the list
            predictions.append(label)
        return predictions

    # function to calculate the closest row to the test row(since KNN)
    def closest(self, row):
        # assuming first row of train is best and taking its index as the best index 
        best_dist = euclid(row, self.X_train[0])
        best_index = 0
        for i in range(1, len(self.X_train)):
            # for every other row in train set find the euclidean distance.. if less then the distance returned from 
            # euclid() is the best(least) distance and use the index  
            dist = euclid(row, self.X_train[i])
            if dist < best_dist:
                best_dist = dist
                best_index = i
        return self.Y_train[best_index]


iris = datasets.load_iris()
# load the iris dataset from sklearn.datasets

# features for x .. labels for y
x = iris.data
y = iris.target

# print(x)
# print(y)

# using train_test_split to split the iris dataset into train and test.. test_size can be varied accordingly
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.5)

# creating the ScrappyKNN object classifier
my_classifier = ScrappyKNN()

# loading up the train set
my_classifier.fit(X_train, Y_train)

# making the predictions.. the list in predict() i.e. predictions is returned back
predictions = my_classifier.predict(X_test)

# print(predictions)

# printing the accuracy of the classifier to understand how well the predictions worked out
print(accuracy_score(Y_test, predictions))
