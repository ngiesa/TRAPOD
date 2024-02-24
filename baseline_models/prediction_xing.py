# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6858207/#:~:text=A%20score%20of%204%20points,not%20result%20in%20this%20diagnosis.
# Xing, H., Zhou, W., Fan, Y., Wen, T., Wang, X., & Chang, G. (2019). 
# Development and validation of a postoperative delirium prediction model for patients admitted to an intensive care unit in China:
#  a prospective study. BMJ open, 9(11), e030733.
# inputs as acid imbalace (apache-III), diabetes history, hypertension history, coma history, POSSUM score 
# POSSUM score https://pubmed.ncbi.nlm.nih.gov/17192224/#:~:text=Introduction%3A%20The%20POSSUM%20scale%20(Physiological,wide%20variety%20of%20surgical%20procedures.
# score is replaced with surrogate: the mean physiological score 23.4 points (range: 12-40 points), while the mean surgical score was 11.3 points (range: 6-24 points).

import numpy as np

class XingPredictor:

    def __init__(self):
        self.inter = (-6.963)
        self.est_apache = 0.310
        self.est_hst_diab = 1.228
        self.est_hst_hypert = 1.308
        self.est_hst_coma = 3.428
        self.est_surgrisk = 0.113

    # substitute POSSUM with asa
    def __get_surgrisk(self, asa):
        if asa < 4:
            return 23.4
        else:
            return 11.3
    
    # use coma var from boogard 
    def __get_surgrisk(self, coma):
        if coma == 0:
            return 0
        else:
            return 1

    def predict_outcome(self, X):
        X = np.array(X)
        #X[-1] = self.__get_surgrisk(X[-1])
        X[3] = self.__get_surgrisk(X[3])
        W = np.array((self.est_apache,
                      self.est_hst_diab,
                      self.est_hst_hypert,
                      self.est_hst_coma,
                      self.est_surgrisk
                      ))
        Z = np.dot(W.T, X.T)+self.inter
        p = 1/(1+np.exp(-Z))
        return p