import time
import numpy as np

def sum_trad():
    start = time.time()
    X = range(10000000)
    Y = range(10000000)
    Z = []
    for i in range(len(X)):
        Z.append(X[i] + Y[i])
    return time.time()-start

def sum_numpy():
    start = time.time()
    X = np.arange(10000000)
    Y = np.arange(10000000)
    Z = X+Y
    return time.time() - start


def main():
    # print("this sum_trad", sum_trad())
    # print("this sum_numpy", sum_numpy())

    # arr = np.array(range(1,9), int)
    # print(arr, type(arr))
    #
    # arr.tolist()
    # z = list(arr)
    # print("z:",z)

    # arr = np.random.permutation(100)
    # arr = np.zeros((2,3), dtype=int)
    # arr1 = np.array([1,2,4])
    # arr2 = np.array([2,6,8])
    # arr3 = np.array([2, 6, 8])
    # arr4 = np.array([2, 6, 8])
    # arr_comp = np.vstack([arr1, arr2, arr3, arr4])
    # print(arr_comp )
    # name = input()
    # print(name)

#     数组操作
    # arr = np.array([1,5,9,2,0,1,1,5,5])
    # print(np.unique(arr)) #不重复元素
    # print(np.sort(arr))
    # print(arr.max())
    #
    # arr = np.array([1,  9, 2, 0, 1, 1, 5, 5])
    # print(arr)
    # z = arr.reshape(4,2)#将数组调整为4行两列
    # print(z) #将数组调整为4行两列
    # tran_z = z.transpose() #矩阵转置
    # print(tran_z)
    # print(tran_z.T) #T属性转置
#      线性代数运算
    X = np.arange(15).reshape((3,5))
    X.fill(1)
    print(X)
    X_T = X.T
    print(X_T)
    res = np.dot(X_T, X)
    print(res)


if __name__ == '__main__':
    main()







