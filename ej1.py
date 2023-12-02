import numpy as np
import matplotlib.pyplot as plt

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

def gradient_descent(A, b, func, gradient,  iter = int(1e3)):
    x = np.random.uniform(size=A.shape[1])
    step_size = 1 / max_eigenvector(H(A))
    cost = func(A, b, x)
    norms, costs = [],[]
    for _ in range(iter):
        x = x - step_size*gradient(A, b, x)
        costs.append(func(A, b, x))
        norms.append(np.linalg.norm(x, ord = 2))
        cost = F(A, b, x)
    return (np.linalg.norm(x, ord = 2), cost, norms, costs)

def main():
    iterations, n, d = 1, 5, 100
    norm_sum_F, norm_sum_F_2, norm_sum_SVD = 0, 0, 0
    for _ in range(iterations):
        random_A, random_b = generate_random(n, d)
        X1, X2 = gradient_descent(random_A, random_b, F, F_gradient), gradient_descent(random_A, random_b, F_2, F_2_gradient)
        plt.plot(X1[2], label="x conseguido con F_1")
        plt.plot(X2[2], label="x conseguido con F_2")
        plt.xlabel("iteraciones")
        plt.ylabel("norma L2")
        plt.title("Norma L2 del vector solucion en base a la cantidad de iteraciones")
        plt.legend()
        plt.show()
        norm_sum_F += X1[0]
        norm_sum_F_2 += X2[0]
        
    print(norm_sum_F/iterations, norm_sum_F_2/iterations)

if __name__ == "__main__":
    main()