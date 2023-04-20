
"""
Chapter 2: 데이터 전처리 및 시각화

Outlier 제거를 위한 모듈입니다.
주어진 라이브러리 이외에 다른 라이브러리는 사용하지 않습니다.

1. del_outlier: 변수별로 outlier의 index를 추출한 후, 해당 데이터들을 제거합니다.
"""

import pandas as pd


def del_outlier(data, y):
    """
    :param
    data   : train 데이터의 X값 (형식: pd.DataFrame)
    y      : train 데이터의 y값 (형식: pd.DataFrame)
    :return: Outlier가 모두 제거된 x값과 y값 (형식: pd.DataFrame)
    """

    # 1. for문을 사용하여 변수별로 outlier의 index를 저장한 후, 중복되는 행들을 처리하여 최종적으로 이상치를 제거하세요.
    # (HINT: for문을 사용하고, 중복되는 행들을 처리하고 위해 set()을 사용하세요.
    
    data = data.reset_index(drop=True)
    y = y.reset_index(drop=True)
    
    df = data.copy()
    df['y'] = y
    
    stat = df.describe()
    
    outliers = []
    for i in range(len(df.columns)):
        q1 = stat.iloc[4,i]
        q3 = stat.iloc[6,i]
        iqr = q3 - q1
        
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        for j in range(len(df)):
            if (df.iloc[j,i] < lower) | (df.iloc[j,i] > upper):
                outliers.append(j)
        
    out = set(outliers)
    
    data.drop(index=out, axis=0, inplace=True)
    y.drop(index=out, axis=0, inplace=True)
    
    return data, y




