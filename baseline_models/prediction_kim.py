# Kim, E. M., Li, G., & Kim, M. (2020). 
# Development of a risk score to predict postoperative delirium in hip fracture patients. Anesthesia and analgesia, 130(1), 79.
# Initially developed as prediction score for hip fracture
# total 20 points, points assigned as fracture of total points 
# input X as preop delir, preop dement, age, medical co manag, asa, functional depend, smoking, sepsis, preop use mob
# preoperative medical consultation as medical co management, mobility aids may represent fraility (1-4 vs. > 4)
# functional dependence asured by impairments in the activities of daily living and was a significant predictor of PHFD implemented as barthel index
# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6917900/#SD4

import numpy as np

class KimPredictor:

    def __init__(self):
        self.preop_delir = 8/20
        self.preop_dim = 3/20
        self.med_co = 1/20
        self.barthel = 1/20
        self.smok = 1/20
        self.sepsis = 1/20

    def __get_age(self, age):
        if age >= 80:
            return 3/20
        elif age >= 70:
            return 2/20
        else:
            return 0
        
    # asa II vs. higher asa score
    def __get_asa(self, asa):
        if asa <=2:
            return 0
        else:
            return 1/20
    
    # fraility index <IV v.s higher
    def __get_frail(self, frail):
        if frail <=4:
            return 0
        else:
            return 1/20
        
    # barthel index <70 v.s. higher
    def __get_barthel(self, frail):
        if frail <=60:
            return 1/20
        else:
            return 0

    def predict_outcome(self, X):
        X = np.array(X)
        p = sum(np.array((
                        X[0],
                        X[1],
                        self.__get_age(X[2]),
                        X[3],
                        self.__get_asa(X[4]),
                        self.__get_barthel(X[5]),
                        X[6],
                        X[7],
                        self.__get_frail(X[8]))
                    ))
        return p
