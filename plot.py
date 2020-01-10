import matplotlib.pyplot as plt
import numpy as np
def plot(x,y):
    plt.plot(x,y)
    plt.show()

if __name__ == "__main__":
    x = [i for i in range(5)]    # 0,1,2,3,4
    x = np.array(x)
    print(x)
    y = x+1
    print(y)
    plot(x,y)