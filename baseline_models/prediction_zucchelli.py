# Zucchelli, A., Apuzzo, R., Paolillo, C., Prestipino, V., De Bianchi, S., Romanelli, G., ... & Bellelli, G. (2021). 
# Development and validation of a delirium risk assessment tool in older patients admitted to the Emergency Department Observation Unit. 
# Aging Clinical and Experimental Research, 1-6.
# Emergeny department model 
# age 2 points, dementia 3 points, hearing impairment 2 points, psychotropic drugs 1 point, in total 8 points
# psychotropic drugs are defined as antidepressant, anipsychotics, benzodia, opioids, anti dimentioa, parkinson

import numpy as np

class ZucchelliPredictor:

    def __init__(self):
        self.dementia = 3/8
        self.hear_impair = 2/8
        self.psych_drugs = 1/8

    def __get_age(self, age):
        if age < 75:
            return 0
        else:
            return 2/8

    def predict_outcome(self, X):
        X = np.array(X)
        p = sum(np.array((
                        self.__get_age(X[0]),
                        X[1],
                        X[2],
                        X[3]
                    )))
        return p
