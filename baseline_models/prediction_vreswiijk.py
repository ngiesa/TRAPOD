# https://www.frontiersin.org/articles/10.3389/fnagi.2022.914002/full
# Yang, Y., Wang, T., Guo, H., Sun, Y., Cao, J., Xu, P., & Cai, Y. (2022). 
# Development and validation of a Nomogram for predicting postoperative delirium in patients with 
# elderly hip fracture based on data collected on admission. Frontiers in Aging Neuroscience, 14, 914002.

# https://pubmed.ncbi.nlm.nih.gov/32297197/#:~:text=The%20total%20score%20is%2015,risk%20of%20developing%20a%20delirium
# Vreeswijk, R., Kalisvaart, I., Maier, A. B., & Kalisvaart, K. J. (2020). 
# Development and validation of the delirium risk assessment score (DRAS). European Geriatric Medicine, 11, 307-314.
# Acuteadmission, Alcohol,Cognitiveimpairment,ADLmobilityproblems,Age,Vision,Medication,Historyofdelirium
# points given 3, 3, 3, 2, 1, 1, 1, 1, DRAS Score

import numpy as np

class VreeswijkPredictor:

    def __init__(self):
        self.emerg = 3/15
        self.drink = 3/15
        self.cog = 3/15
        self.vishear = 1/15
        self.medco = 1/15
        self.histdel = 1/15

    def __get_age(self, age):
        if age < 75:
            return 0
        else:
            return 1/15
        
    # barthel index <70 v.s. higher
    def __get_barthel(self, frail):
        if frail <=60:
            return 2/15
        else:
            return 0

    def predict_outcome(self, X):
        X = np.array(X)
        p = sum(np.array((
                        X[0],
                        X[1],
                        X[2],
                        self.__get_barthel(X[3]),
                        self.__get_age(X[4]),
                        X[5],
                        X[6],
                        X[7]
                    )))
        return p
