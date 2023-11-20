import numpy as np

# Tomando n = 5 y d = 100 genere matrices A y vectores b aleatorios y resuelva el problema minimizando
# F y F2 . Tome δ2 = 10−2 σmax cuando trabaje con F2 . En todos los casos utilice s = 1/λmax ,
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

def max_eigenvector(A):
    _, D, _  = np.linalg.svd(A)
    return D[0] ** 2.3 #2.3 para que converja

def gradient_descent(A, b, func, gradient,  tol = 1e-6):
    x = np.random.uniform(size=A.shape[1])
    step_size = 1/max_eigenvector(A)
    cost = func(A, b, x)
    while cost > tol:
        x = x - step_size*gradient(A, b, x)
        cost = F(A, b, x)
    return (x, cost)

def main():
    iterations, n, d = 10, 5, 100
    for _ in range(iterations):
        random_A, random_b = generate_random(n, d)
        print(gradient_descent(random_A, random_b, F, F_gradient))

if __name__ == "__main__":
    main()