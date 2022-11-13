import os
import pickle
from sklearn.tree import DecisionTreeClassifier

def load_model():
    models_base_path = 'model'
    # load Decision Tree
    filename = os.path.join(models_base_path, 'decision_tree_model.pkl')
    model = pickle.load(open(filename, 'rb'))
    return model