
"""
Chapter 2: 데이터 전처리 및 시각화

Minmax Scaling을 사용하기 위한 모듈입니다.
주어진 라이브러리 이외에 다른 라이브러리는 사용하지 않습니다.

1. minmax_scaling: train 데이터로 min과 max를 계산한 후, train 데이터와 test 데이터에 대해 한 번에 minmax_scaling을 진행합니다.
"""

import numpy as np
import pandas as pd


class Scaling(object):
    def __init__(self, x):
        """
        :param
        inputs: train 데이터(X)를 입력으로 받습니다. (형식: pd.DataFrame)
        """

        # 1. 입력받은 train 데이터의 복사본을 만들어 self.X에 저장합니다. (HINT: copy()를 활용하세요.)
        self.X = x.copy()

    def minmax_scaling(self, test_data):
        """
        :param
        test_data: test 데이터(X)를 입력으로 받습니다. (형식: pd.DataFrame)
        :returns Minmax Scaling이 진행된 train 데이터와 test 데이터를 반환합니다.
        """

        # 1. train 데이터의 속성의 개수를 dim 객체에 저장하고, test 데이터의 복사본을 test1에 저장하세요.
        # 2. train 데이터의 속성별 min과 max를 찾아 train과 test 데이터 모두에 minmax scaling을 진행하세요. (HINT: for문을 사용하세요.)
        
        
        dim = len(self.X.columns)
        test1 = test_data.copy()
        
        for i in range(dim):
            mi = min(self.X.iloc[:,i])
            ma = max(self.X.iloc[:,i])
            
            for j, v in enumerate(self.X.iloc[:,i]):
                newv = (v-mi)/(ma-mi)
                self.X.iloc[j,i] = newv
                
            for j, v in enumerate(test1.iloc[:,i]):
                newv = (v-mi)/(ma-mi)
                test1.iloc[j,i] = newv
        
        rst_tr = self.X.copy()
        
        return rst_tr, test1
        
        
        
        
        
