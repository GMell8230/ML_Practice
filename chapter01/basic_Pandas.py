import numpy as np
import pandas as pd
import os
import random
# obj = pd.Series([1,2,-3,5])
# print(obj[obj>2])
# print(obj.values)
# print(obj.index)

# data = {'b':90, 'c':88,'a' : 30 ,  'd':0}
# obj1 = pd.Series(data)
# print(obj1)
# 数据读取以及列选择
# data = pd.read_csv(os.path.abspath(".") + "\\data_example\\ad-dataset\\ad.data", header=None)
# data[data[1558] == 'ad.'].head(4)
# print(data[data[1558] == 'ad.'].head(4))

data = pd.read_csv(os.path.abspath(".") + "\\data_example\\ad-dataset\\ad.data", header=None)
# print(data.iloc[:3]) 行
row = [random.randint(0,1) for r in xrange(1588)] + ['ad.']
data = data.append(pd.Series(row, index= data.columns), ignore_index=True)
