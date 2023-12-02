import numpy as np
import matplotlib.pyplot as plt
import math

# Tomando n = 5 y d = 100 genere matrices A y vectores b aleatorios y resuelva el problema minimizando
# F y F2 . Tome δ_2 = 10−2 σmax cuando trabaje con F2 . En todos los casos utilice s = 1/λmax ,
# una condición inicial aleatoria y realice 1000 iteraciones. Estudiar como evoluciona la solución del
# gradiente descendiente iterativamente. Compare con la solución obtenida mediante SVD. Analice los
# resultados. ¿Por qué se elige este valor de s? ¿Qué sucede si se varían los valores de δ2 ?


def generate_random(n, d):
    return np.random.uniform(size=(n, d)), np.random.uniform(size=n)

# F (x) = (Ax − b)^T (Ax − b).
def F(A, b, x):
    return (A@x - b).T @ (A@x - b)

def F_gradient(A, b, x):
    return 2 * A.T @ (A @ x - b)

def max_singular_value(A):
    _, D, _ = np.linalg.svd(A)
    return D[0]

def F_2(A, b, x):
    S_2 = 1e-2 * max_singular_value(A)
    return F(A, b, x) + S_2 * np.linalg.norm(x)**2

def F_2_gradient(A, b, x):
    S_2 = 1e-2 * max_singular_value(A)
    return F_gradient(A, b, x) + 2 * S_2 * x

def H(A):
    return 2*A@A.T

def max_eigenvector(A):
    vals, _  = np.linalg.eig(A)
    return vals[0]

def gradient_descent(A, b, func, gradient,  iter = int(1e3), step_size = None):
    x = np.random.uniform(size=A.shape[1])
    if(step_size==None):
        step_size = 1 / max_eigenvector(H(A))
    cost = func(A, b, x)
    norms, costs = [],[]
    for _ in range(iter):
        x = x - step_size*gradient(A, b, x)
        cost = func(A, b, x)
        costs.append(cost)
        norms.append(np.linalg.norm(x, ord = 2))
        
    return (np.linalg.norm(x, ord = 2), F(A, b, x), norms, costs, x)

def gradient_descent_momentum(A, b, func, gradient, beta=0.9, iter=int(1e3)):
    x = np.random.uniform(size=A.shape[1])
    velocity = np.zeros(A.shape[1])
    step_size = 1 / max_eigenvector(H(A))
    costs = []

    for _ in range(iter):
        grad = gradient(A, b, x)
        velocity = beta * velocity + (1 - beta) * grad
        x -= step_size * velocity
        costs.append(func(A, b, x))
        
    return x, costs

def solve_svd(A, b):
    U, S, VT = np.linalg.svd(A)
    S_inv = np.zeros((A.shape[1], A.shape[0]))
    S_inv[:len(S), :len(S)] = np.diag(1 / S)
    A_pseudo_inverse = VT.T @ S_inv @ U.T
    x = A_pseudo_inverse @ b
    return x


def main():
    iteraciones, n, d = 1, 5, 100
    norm_sum_F, norm_sum_F_2, norm_sum_SVD = 0, 0, 0
    SOL = np.zeros(shape=10)
    for _ in range(iteraciones):
        random_A, random_b = generate_random(n, d)
        X1, X2, X_SVD, X1_Momentum, X2_Momentum = gradient_descent(random_A, random_b, F, F_gradient), gradient_descent(random_A, random_b, F_2, F_2_gradient),solve_svd(random_A, random_b), gradient_descent_momentum(random_A, random_b, F, F_gradient), gradient_descent_momentum(random_A, random_b, F_2, F_2_gradient)
        #step size experiment
        costs, step_sizes = [], []
        original_step_size = 1 / max_eigenvector(H(random_A))
        start, end = -200*original_step_size, 200*original_step_size
        step = (end - start)/100
        for s in np.arange(start, end, step):
            x1 = gradient_descent(random_A, random_b, F, F_gradient)
            costs.append(F(random_A, random_b, x1[-1]))
            step_sizes.append(s)
        
        plt.axvline(x=original_step_size, color='red', linestyle='--')
        # Add a label to the vertical line
        # plt.text(original_step_size, plt.ylim()[1], 'Step Size Original', rotation=90, verticalalignment='top')
        plt.xlabel("step-size")
        plt.ylabel("Costo")
        plt.legend()
        plt.plot(step_sizes, costs)
        plt.show()
        ##########PLOT cost vs iteraciones de F1 F2#########################
        # plt.plot(X1[3], label="x conseguido con F_1")
        # plt.plot(X2[3], label="x conseguido con F_2")
        # plt.xlabel("iteraciones")
        # plt.ylabel("Costo")
        # plt.title("Costo del vector solucion en base a la cantidad de iteraciones")
        # plt.legend()
        # plt.show()
        SOL += (X1[0], X1[1], X2[0] , X2[1], np.linalg.norm(X_SVD, ord=2), F(random_A, random_b, X_SVD), np.linalg.norm(X1_Momentum[-1], ord=2), X1_Momentum[1][-1], np.linalg.norm(X2_Momentum[0], ord=2), X2_Momentum[1][-1])
        norm_sum_F += X1[0]
        norm_sum_F_2 += X2[0]
    SOL/=iteraciones
    print(SOL)

if __name__ == "__main__":
    main()