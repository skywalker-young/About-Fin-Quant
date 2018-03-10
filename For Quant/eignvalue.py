import numpy as np

x=np.diag((1,2,3,4))

a,b=np.linalg.eig(x) #a特征值 b 特征向量

'''
使用循环的方法，我们来验证一下特征值和特征向量，验证的方法是特征值和特征向量的定义法：
设A为n阶矩阵，若存在常数λ及非零的n维向量x，使得
Ax=λx，
'''
for i in range(3):
    if (np.dot(x,b[:][i])==a[i]*b[i]).all():
        print('right')
    else:
        print('wrong')
