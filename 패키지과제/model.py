
"""
Chapter 3: 모델링 - Normal Equation

이후 만들게 될 회귀모형들의 부모 클래스가 되는 Model 클래스를 먼저 만들어 보겠습니다.
Model 클래스는 다음의 3가지 함수를 가지고 있습니다.
주어진 라이브러리 이외에 다른 라이브러리는 사용하지 않습니다.

1. __init__       : class가 처음 정의될 때 자동으로 생성되는 variable들을 정의합니다.
2. describe       : X와 Y의 기술통계량을 선택적으로 확인합니다.
3. predict        : test 데이터에 대한 모델의 예측값을 반환합니다.
"""

import pandas as pd
import numpy as np


class Model(object):
    def __init__(self, data1, data2):
        """
        :param
        data1: X값들을 입력으로 받습니다. (형식: pd.DataFrame)
        data2: y값들을 입력으로 받습니다. (형식: pd.DataFrame)
        """
        self.X = data1
        self.Y = data2

        # 1. data1의 row 개수를 self.m, data1의 col 개수를 self.n에 각각 저장하세요.
        # 2. data1의 오른쪽에 1로 채워진 열을 붙여 array로 바꾼 후 self.X_mat에 새로 저장하세요.
        # 3. self.Y를 array로 바꾼 후 self.Y_mat에 새로 저장하세요.
        # 4. 회귀계수를 저장하기 위해 임시로 1로 채워진 array를 self.theta에 저장하세요.
        # 5. 예측값을 저장하기 위한 빈 객체로 self.train_pred와 self.test_pred를 생성하세요.
        
        self.m = len(self.X.index)
        self.n = len(self.X.columns)
        self.X['one'] = 1
        self.X_mat = np.array(self.X)
        self.Y_mat = np.array(self.Y)
        self.theta = np.ones(self.n)
        self.train_pred = None
        self.test_pred = None

    def describe(self, which):
        """
        :param
        which: 'X' 혹은 'y'를 입력으로 받습니다. (형식: str)
        :return: pd.DataFrame.info()
        """

        # 1. if문을 활용하여 'X'가 입력되었을 때는 data1의 info를, 'y'가 입력되었을 때는 data2의 info를 반환하도록 하세요.
        # 2. 만약 2가지 이외의 무언가가 입력되었을 때는 "Not Defined"라는 예외를 띄우도록 하세요. (HINT: raise Exception을 사용하세요.)
        
        if which == 'X':
            return data1.info()
        elif which == 'y':
            return data2.info()
        else:
            raise Exception("Nont Defined")
        

    def predict(self, test1):
        """
        :param
        test1: test 데이터를 입력으로 받습니다.        (형식: pd.DataFrame)
        :return: test 데이터에 대한 예측 결과를 반환합니다. (형식: list)
        """

        # 1. test 데이터에 self.X_mat과 동일한 처리를 진행하여 test_data 객체에 저장하세요.
        # 2. test_data와 self.theta를 곱해서 test 데이터에 대한 예측값을 출력하세요. (HINT: np.matmul을 사용하세요.)
        
        self.test_data = test1
        self.test_data['one'] = 1
        self.test_data = np.array(self.test_data)
        
        self.test_pred = np.matmul(self.test_data, self.theta)
        return self.test_pred
        
        
        
        
        
        

