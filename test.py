import numpy as np

a = np.arange(0, 10)
for i in range(100):
    b = np.random.choice(a, 4, replace=False)
    b.sort()
    if np.unique(b).size != b.size:
        print(i, 'hehe')
    print(b, b[np.array((0,1,3))])
