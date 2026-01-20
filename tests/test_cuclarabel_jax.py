import cvxpy as cp
import jax

from cvxpylayers.jax import CvxpyLayer

def main():
    n, m = 2, 3
    x = cp.Variable(n)
    A = cp.Parameter((m, n))
    b = cp.Parameter(m)
    constraints = [x >= 0]
    objective = cp.Minimize(0.5 * cp.pnorm(A @ x - b, p=1))
    problem = cp.Problem(objective, constraints)
    assert problem.is_dpp()

    layer = CvxpyLayer(problem, parameters=[A, b], variables=[x], solver=cp.CUCLARABEL)
    key = jax.random.PRNGKey(0)
    key, k1, k2 = jax.random.split(key, 3)
    A_jax = jax.random.normal(k1, shape=(m, n))
    b_jax = jax.random.normal(k2, shape=(m,))

    (solution,) = layer(A_jax, b_jax)

    # compute the gradient of the summed solution with respect to A, b
    dlayer = jax.grad(lambda A, b: sum(layer(A, b)[0]), argnums=[0, 1])
    gradA, gradb = dlayer(A_jax, b_jax)
    print(f"gradA: {gradA}")
    print(f"gradb: {gradb}")


if __name__ == "__main__":
    main()