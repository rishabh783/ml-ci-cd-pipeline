import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

#Load the data set
data = pd.read_csv("data/iris.csv")

# Preprocess the dataset
X = data.drop("species" , axis=1)
Y = data["species"]

#Split the data into traingin and test sets
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2,random_state=42)

#train a randomForest model
model= RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

#save the model
joblib.dump(model, "model/iris_model.pkl")


#Load the saved model
model = joblib.load("model/iris_model.pkl")

#Make predictions
Y_pred = model.predict(X_test)

# Evalutate the model
accuracy = accuracy_score(y_test ,y_pred)
print(f"Model accuracy: {accuracy:.2f}")



# Writing unit test 
import unittest
import joblib 
from sklearn.ensemble import RandomForestClassifier

class TestModelTraining(unitest.TestCase):
	def test_model_training(self):
		model= joblib.load("model/iris_model.pkl")
		self.assertIsInstance(model, RandomForestClassifier)
		self.assertGreaterEqual(len(modle.feature_importances_), 4)

if _name_== "_main_":
	unittest.main()