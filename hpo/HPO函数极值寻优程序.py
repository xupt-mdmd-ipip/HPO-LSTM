import numpy as np
from matplotlib import pyplot as plt

'''定义目标函数用户可选fun1 - fun6 , 也可以自己定义自己的目标函数'''


def fun1(X):
    O = np.sum(X * X)
    return O


def fun2(X):
    O = np.sum(np.abs(X)) + np.prod(np.abs(X))
    return O


def fun3(X):
    O = 0
    for i in range(len(X)):
        O = O + np.square(np.sum(X[0:i + 1]))
    return O


def fun4(X):
    O = np.max(np.abs(X))
    return O


def fun5(X):
    X_len = len(X)
    O = np.sum(100 * np.square(X[1:X_len] - np.square(X[0:X_len - 1]))) + np.sum(np.square(X[0:X_len - 1] - 1))
    return O


def fun6(X):
    O = np.sum(np.square(np.abs(X + 0.5)))
    return O


def pdist2(Y, X):
    dis = np.zeros((X.shape[0]))
    for i in range(X.shape[0]):
        dis[i] = np.sqrt(np.sum((X[i, :] - Y) ** 2))
    return dis


def Bounds(x, lb, ub):
    dim = len(x)
    for i in range(dim):
        if x[i] > ub[i] or x[i] < lb[i]:
            x[i] = (ub[i] - lb[i]) * np.random.rand() + lb[i]
    return x


def main(nPop, MaxIt, lb, ub, dim, CostFunction):
    # MaxIt    #Maximum Nomber of Iterations
    # nPop         # Population Size
    Convergence_curve = np.zeros((MaxIt,))

    # Constriction Coefeicient
    B = 0.1

    # Initialization
    HPpos = np.random.rand(nPop, dim) * (ub - lb) + lb
    print(HPpos.shape)
    # HPpos = np.array([2, 2, 6, 3, 2, 1, 0.3])

    # Evaluate
    HPposFitness = np.zeros((nPop,))
    for i in range(nPop):
        HPposFitness[i] = CostFunction(HPpos[i])

    indx = np.argmin(HPposFitness)
    Target = HPpos[indx, :]  # Target HPO
    TargetScore = HPposFitness[indx]
    Convergence_curve[0] = TargetScore

    # Main Loop
    for it in range(1, MaxIt):

        c = 1 - it * ((0.98) / MaxIt)  # Update C Parameter
        kbest = int(nPop * c)  # Update kbest
        for i in range(nPop):
            r1 = (np.random.rand(dim) < c).astype(np.int32)
            r2 = np.random.rand()
            r3 = np.random.rand(dim)
            idx = (r1 == 0)
            z = r2 * idx + r3 * ~idx

            if np.random.rand() < B:
                xi = np.mean(HPpos, axis=0)
                dist = pdist2(xi, HPpos)
                idxsortdist = np.argsort(dist)
                SI = HPpos[idxsortdist[kbest], :]
                HPpos[i, :] = HPpos[i, :] + 0.5 * (
                        (2 * c * z * SI - HPpos[i, :]) + (2 * (1 - c) * z * xi - HPpos[i, :]))
            else:
                for j in range(dim):
                    rr = -1 + 2 * z[j]
                    HPpos[i, j] = 2 * z[j] * np.cos(2 * np.pi * rr) * (Target[j] - HPpos[i, j]) + Target[j]

            HPpos[i, :] = Bounds(HPpos[i, :], lb, ub)
            # Evaluation
            HPposFitness[i] = CostFunction(HPpos[i, :])
            # Update Target
            if HPposFitness[i] < TargetScore:
                Target = HPpos[i, :]
                TargetScore = HPposFitness[i]

        Convergence_curve[it] = TargetScore
        # print('Iteration: ',it,' Best Cost = ',TargetScore);
    return TargetScore, Target, Convergence_curve


# In[]
if __name__ == "__main__":
    # 设置参数
    pop = 7 * 24   # 种群数量
    MaxIter = 100  # 最大迭代次数
    dim = 2  # 维度
    lb = 0 * np.ones([dim, ])  # 下边界
    ub = 100 * np.ones([dim, ])  # 上边界
    # 适应度函数选择
    fobj = fun3

    GbestScore1, GbestPositon1, Curve1 = main(pop, MaxIter, lb, ub, dim, fobj)

    print('最优适应度值：', GbestScore1)
    print(GbestPositon1)

    # 绘制适应度曲线
    plt.figure()
    plt.semilogy((1 / Curve1), 'b-', linewidth=2)
    # plt.plot(Curve1,'b-',linewidth=2)
    plt.xlabel('Iteration', fontsize='medium')
    plt.ylabel("Fitness", fontsize='medium')
    # plt.grid()
    plt.title('Fitness', fontsize='large')
    plt.show()
