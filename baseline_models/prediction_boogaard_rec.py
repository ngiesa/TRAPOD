# Prediction model
# van den Boogaard, M. H. W. A., Schoonhoven, L., Maseda, E., Plowright, C., Jones, C., Luetz, A., ... & Pickkers, P. (2014).
# Recalibration of the delirium prediction model for ICU patients (PRE-DELIRIC): a multinational observational study. Intensive care medicine, 40(3), 361-369.
# using the new values of linear predictors .. maybe splitting in recal and normal models
# urea must be highest value in mm/L

import numpy as np

class BoogaardPredictorRecalibrated:

    def __init__(self):
        self.inter = (-4.0369)
        self.est_age = 0.0183
        self.est_apache = 0.0272
        self.est_infect = 0.4965
        self.est_meta_acid = 0.1378
        self.est_seda = 0.6581
        self.est_urea = 0.0141
        self.est_urg = 0.1891

    # coma 0 -> no, 1 -> drug, 2 -> other, 3 -> combination
    def __get_est_coma(self, coma):
        if coma == 1:
            return 0.2578
        elif coma == 2:
            return 1.0721
        elif coma == 3:
            return 1.3361
        else:
            return 0

    # adm cat 1 -> surgery, 2 -> medical, 3 -> trauma, 4 -> neuro/surgery
    def __get_est_adm_cat(self, cat):
        if cat == 2:
            return 0.1446
        elif cat == 3:
            return 0.5316
        elif cat == 4:
            return 0.6516
        else:
            return 0

    # enter the commulated morphine dosis in mg
    def __get_est_morph(self, morph):
        if (morph >= 0.01) and (morph <= 7.1):
            return 0.1926
        elif (morph >= 7.2) and (morph <= 18.6):
            return 0.0625
        elif (morph > 18.6):
            return 0.2414
        else:
            return 0

    def predict_outcome(self, X):
        X = np.array(X)
        W = np.array((self.est_age,
                      self.est_apache,
                      self.__get_est_coma(X[2]),
                      self.__get_est_adm_cat(X[3]),
                      self.est_infect,
                      self.est_meta_acid,
                      self.__get_est_morph(X[6]),
                      self.est_seda,
                      self.est_urea,
                      self.est_urg
                      ))
        Z = np.dot(W.T, X.T)+self.inter
        p = 1/(1+np.exp(-Z))
        return p
