import matplotlib.pyplot as plt
import numpy as np



def curve(x):
    y=x**2-x+1
    return y

def line (x):
    y=x+1
    return y

a=np.linspace(-2,3,500)

'''
y1=curve(a)
y2=line(a)

plt.plot(a,y1)
plt.plot(a,y2)
plt.show()
'''
###在区间[0,2]之间求面积 a=0,b=2
def area (a,b,n):
    results=0
    length=(b-a)/n
    for i in range(n):
        results=results+line((i)*length)-curve(i*length)

    return results*length

print(area(0,2,200))

##baidu 是4/3==1.3333接近
