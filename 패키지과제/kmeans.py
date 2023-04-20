"""
Chapter 3. 모델링 - K-Means Clustering

KMeans 클래스를 완성하여 K-Means Clustering 알고리즘을 구현해 주세요.

1. init_centroid    : K-Means Clustering은 초기 centroid가 어떻게 정해지는지에 따라 군집화의 결과가 크게 달라집니다.
                      따라서, 단순히 랜덤하게 centroid를 선정하는 대신 확률적인 방법으로 centroid를 선정하는 함수입니다.
2. fit              : K-Means Clustering은 일정한 거리 척도에 따라 centroid들과 가까운 데이터들을 cluster에 배정하고,
                      배정된 결과에 따라 centroid를 업데이트하는 방식으로 학습됩니다.
3. predict          : fit()을 통해 얻은 결과를 반환합니다.
4. wcss             : WCSS(Within Clusters Sum of Squares)를 계산하는 함수입니다.
                      Elbow Method로 최적의 클러스터 개수를 파악하기 위해 사용됩니다.
"""

import random
import numpy as np
import pandas as pd
import random


class KMeans(object):
    def __init__(self,
                 data: pd.DataFrame,
                 K: int):
        """
        :param data: Region 변수를 제외한 데이터를 받습니다. pd.DataFrame의 value들만을 입력으로 받습니다.
        :param K: 군집의 개수를 의미합니다.
        """
        self.data = data
        self.result = {}
        self.centroids = np.zeros(shape=(self.data.shape[1], 0)) #(9, 0)
        self.k = K #군집 개수

    def init_centroid(self):
        # 1. 임시 Centroid 1개 선정 (랜덤)
        c_index = random.randint(0, self.data.shape[0]) #0~400 중 1개 추출
        c_temp = np.array([self.data[c_index]]) #해당 인덱스의 각 컬럼 값 추출

        # 2. 임시 Centroid (k - 1)개 선정
        for num in range(1, self.k):
            distance = np.array([])

            """
            문제 1. 
            for문을 사용하여 각 학습 데이터와 모든 임시 centroid의 유클리드 거리를 구하고, 그 중 가장 작은 값을 distance에 저장하세요.
            """
            ##################################
            for i in self.data:
                euc_dist = np.sqrt(np.sum((i - c_temp)**2, axis=1))
                distance = np.append(distance, np.min(euc_dist))
            ##################################

            """
            문제 2.
            prob에 distance의 총합으로 scaling된 distance를 저장하고, cum_prob에 prob의 누적값을 저장하세요.
            []
            (HINT) np.cumsum을 사용하면 편합니다.
            """
            ##################################
            prob = distance / np.sum(distance)
            cum_prob = np.cumsum(prob)
            ##################################
            m = random.random()
            c_index = 0

            for idx, value in enumerate(cum_prob):
                if m < value:
                    c_index = idx
                    break

            """
            문제 3.
            init_centroid() 함수가 어떤 방식으로 초기 centroid를 설정하는지 설명해주세요.
            """

            c_temp = np.append(c_temp, [self.data[c_index]], axis=0)
            # 초기 중심점 1개 랜덤 추출, 나머지 k-1개의 초기 중심점은 첫번째 초기 중심점과 데이터 간의 거리를 확률적으로 고려하여 할당, 최종적으로 k개의 초기 중심점이 담긴 c_temp 반환

        return c_temp.T

    def fit(self,
            n_iter: int):
        self.centroids = self.init_centroid()
        #print(self.centroids)

        for n in range(n_iter):
            euc_dist = np.array([]).reshape(self.data.shape[0], 0)

            """
            문제 4.
            for문을 사용하여 각 centroid와 모든 데이터의 거리를 구하여 temp에 저장하고, 이를 euc_dist에 차례대로 저장해 주세요.
            
            (HINT) 
            for문이 시작하기 전 euc_dist의 shape은 (data의 개수, 0)이지만,
            for문이 종료된 후 euc_dist의 shape은 (data의 개수, centroid의 개수)입니다.
            """
            ##################################
            for k in range(self.k):
                temp = np.sqrt(np.sum((self.data - self.centroids.T[k])**2, axis=1))
                #print(temp.shape) # (400, 0)
                temp = temp.reshape(-1, 1)
                euc_dist = np.hstack((euc_dist, temp)) #(400, 3)
            ##################################
            #print(euc_dist.shape)
            #print(euc_dist)

            """
            문제 5.
            각 데이터가 어떤 centroid와 가까운지 확인하여 cluster에 저장하세요.
            
            (HINT) euc_dist는 각 데이터와 모든 centroid 간의 거리가 저장된 array입니다. 여기에 np.argmin() 함수를 사용하면 편합니다.
            """
            ##################################
            cluster = np.argmin(euc_dist, axis=1) + 1
            ##################################
            #print(cluster.shape) # (400, 0)

            points = {}
            by_index = {}

            for k in range(self.k):
                points[k + 1] = np.array([]).reshape(self.data.shape[1], 0)
                by_index[k + 1] = []
            for i in range(self.data.shape[0]):
                points[cluster[i]] = np.c_[points[cluster[i]], self.data[i]]
                by_index[cluster[i]].append(i)
            for k in range(self.k):
                points[k + 1] = points[k + 1].T
            for k in range(self.k):
                self.centroids[:, k] = np.mean(points[k + 1], axis=0)

            """
            문제 6. fit() 함수가 어떤 방식으로 군집화를 진행하는지 설명해 주세요.
            """
            # 전체 데이터마다 각 군집 별 거리를 구한다. 각 데이터마다 가장 가까운 군집을 찾는다. 데이터를 군집에 할당해주고, 각 군집의 평균을 새로운 중심점으로 지정한다. 해당 과정을 n번 반복한다.
            self.result = points
            self.index = by_index

    def predict(self):
        return self.result, self.centroids.T, self.index

    def wcss(self):
        wcss = 0

        for k in range(self.k):
            wcss += np.sum((self.result[k + 1] - self.centroids[:, k]) ** 2)

        return wcss

    
    
    
    
    
