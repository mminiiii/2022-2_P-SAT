
"""
정규화 방법 중 하나인 Minmax Scaling을 구현한 모듈입니다.

1. minmax_scaling: 독립변수들을 입력받아 minmax scaling한 결과를 반환합니다.
"""

import numpy as np
import pandas as pd


class Scaling(object):
    def __init__(self,
                 inputs: pd.DataFrame):
        self.X = inputs.iloc[:, 1:].copy()
        self.target = inputs.iloc[:, 0].copy()

    def minmax_scaling(self):
        dim = self.X.shape[1]
        #test1 = test_data.copy()

        for k in range(dim):
            col1 = self.X.iloc[:, k]
            #col2 = test1.iloc[:, k]

            minimum = np.min(col1)
            maximum = np.max(col1)

            col1 = (col1 - minimum) / (maximum - minimum)
            #col2 = (col2 - minimum) / (maximum - minimum)

            self.X.iloc[:, k] = col1
            #test1.iloc[:, k] = col2
            
        self.X.insert(0, 'Region', self.target, True)

        return self.X.copy()
    
    
    