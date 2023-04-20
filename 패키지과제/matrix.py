
"""
Chapter 3: 모델링 - Normal Equation

행렬 연산을 통한 최소제곱법을 구현하기 위해 Normal Equation의 연산 과정을 모듈로 만들어 보겠습니다.
Model 클래스를 상속받아 사용하겠습니다.
주어진 라이브러리 이외에 다른 라이브러리는 사용하지 않습니다.

1. normal_eq       : Model 클래스를 통해 입력받은 데이터에 대해 회귀모형을 적합하고, 예측값과 회귀계수를 반환합니다.
"""

import numpy as np
from model import Model


class LSE(Model):
    # 1. Model 클래스가 LSE 클래스의 부모 클래스가 될 수 있도록 코드를 작성해 주세요. (HINT: 괄호 안에 코드를 작성하세요.)

    def normal_eq(self):
        """
        :return: self.theta, self.train_pred
        """
        # 2. X의 전치행렬과 X를 곱한 후 그 역행렬을 구하여 inverse 객체에 저장하세요 (HINT: np.matmul, np.linalg.inv를 사용하세요.)
        # 3. 2번의 역행렬과 X의 전치행렬을 곱한 후 Y값과 곱하여 회귀계수를 구하고, self.theta에 저장하세요. (HINT: np.matmul을 사용하세요.)
        # 4. X와 회귀계수를 곱하여 얻은 예측값을 self.train_pred 객체에 저장하세요.
        
        #inv(X'X)X'y
        
        self.inverse = np.linalg.inv(np.matmul(self.X_mat.T, self.X_mat))
        self.theta = np.matmul(np.matmul(self.inverse, self.X_mat.T), self.Y_mat)
        self.train_pred = np.matmul(self.theta, self.X_mat.T)
        
        return self.theta, self.train_pred
        
        
        
        
        
