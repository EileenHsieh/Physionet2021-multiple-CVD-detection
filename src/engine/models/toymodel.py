import os
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from models.base import Classifier

def ToyArch(n_estimators, max_leaf_nodes, random_state):
    classifier = RandomForestClassifier(n_estimators=n_estimators, 
                                        max_leaf_nodes=max_leaf_nodes, 
                                        random_state=random_state)
    return classifier
    
class ToyModel(Classifier):
    def __init__(self, config, random_state):
        super(ToyModel, self).__init__(config, random_state)
        
        self._engine = ToyArch(self.n_estimators, self.max_leaf_nodes, self.random_state)

