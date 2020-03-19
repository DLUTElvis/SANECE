from sklearn.svm import LinearSVC
from sklearn.metrics import *
import warnings
import numpy as np

def multiclass_node_classification_eval(X, y, ratio=0.3, seed=0):
	state = np.random.get_state()
	np.random.seed(seed)
	y = np.array(y)
	warnings.filterwarnings("ignore")
	num_nodes = len(X)
	shuffle_indices = np.random.permutation(np.arange(num_nodes))
	train_idx = shuffle_indices[:int(ratio * num_nodes)]
	test_idx = shuffle_indices[int(ratio * num_nodes):]
	X_train = X[train_idx]
	y_train = y[train_idx]
	X_test = X[test_idx]
	y_test = y[test_idx]
	clf = LinearSVC()
	clf.fit(X_train, y_train)
	test_y_pred = clf.predict(X_test)
	test_macro_f1 = f1_score(y_test, test_y_pred, average = "macro")
	test_micro_f1 = f1_score(y_test, test_y_pred, average = "micro")
	np.random.set_state(state)
	return test_macro_f1, test_micro_f1

