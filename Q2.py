
from sklearn import datasets, svm, metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import argparse
import matplotlib.pyplot as plt
import joblib

parser = argparse.ArgumentParser()

parser.add_argument("-cn", "--clf_name", help="Name of model, supported - svm or tree.")
parser.add_argument("-rs", "--random_state", help="Seed for the random number generator.")

args = parser.parse_args()

def train_test_split(ratio=0.1, seed=42):
	digits = datasets.load_digits()
	# flatten the images
	n_samples = len(digits.images)
	data = digits.images.reshape((n_samples, -1))
	return train_test_split(
	    data, digits.target, test_size=ratio, shuffle=True, random_state=seed
	)

def dcs_tree(data):
        X_train, X_test, y_train, y_test = data
        clf = DecisionTreeClassifier(random_state=42)
        clf.fit(X_train, y_train)

        return clf

def svm(data, gamma=0.01, C=5):
	X_train, X_test, y_train, y_test = data
	clf = svm.SVC(gamma=gamma, C=C)
	clf.fit(X_train, y_train)

	return clf



def test_model(model, data, args):
	_, X_test, __, y_test = data
	print("Model Accuracy:", metrics.accuracy_score(y_test, model.predict(X_test)))

	model_path = f'./models/{args.clf_name}_gamma=0.01_C=5_random_state={args.random_state}.joblib'

	with open(f"results/{args.clf_name}_{args.random_state}.txt", "w") as file:
		print("test accurancy:", metrics.accuracy_score(y_test, model.predict(X_test)), file=file)
		print("test macro-f1:", metrics.f1_score(y_test, model.predict(X_test), average="macro"), file=file)
		print(f"model saved at {model_path}", file=file)

	joblib.dump(model, model_path)

if __name__ == "__main__":
	data = train_test_split(seed=int(args.random_state))
	if args.clf_name == "svm":
		model = svm(data)
	elif args.clf_name == "tree":
		model = dcs_tree(data)
	else:
		print("Please enter a valid model name using --clf_name flag")

	test_model(model, data, args)
