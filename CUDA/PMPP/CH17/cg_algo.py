import numpy as np


def conjugate_gradient(A, b, x0=None, tol=1e-10, max_iter=None):
    """
    Solve the linear system Ax = b using the Conjugate Gradient method.

    Parameters:
    -----------
    A : numpy.ndarray
        The coefficient matrix (must be symmetric and positive-definite)
    b : numpy.ndarray
        The right-hand side vector
    x0 : numpy.ndarray, optional
        Initial guess for the solution (default: zero vector)
    tol : float, optional
        Error tolerance (default: 1e-10)
    max_iter : int, optional
        Maximum number of iterations (default: size of A)

    Returns:
    --------
    x : numpy.ndarray
        The solution vector
    residuals : list
        The residual norm at each iteration
    """
    n = len(b)
    if max_iter is None:
        max_iter = n

    # Initialize
    if x0 is None:
        x = np.zeros_like(b, dtype=float)
    else:
        x = x0.copy()

    # Calculate initial residual r = b - Ax
    r = b - A @ x
    p = r.copy()

    # Initialize residual norm list
    residuals = [np.linalg.norm(r)]

    # Print initial state
    print("Iteration 0:")
    print(f"  x = {x}")
    print(f"  r = {r}")
    print(f"  |r| = {residuals[0]}")
    print(f"  p = {p}")
    print("-" * 40)

    # Iterate until convergence or max iterations
    for k in range(max_iter):
        # Compute A*p
        Ap = A @ p

        # Calculate step size alpha = (r^T * r) / (p^T * A * p)
        rTr = r @ r
        alpha = rTr / (p @ Ap)

        # Update solution x = x + alpha*p
        x = x + alpha * p

        # Update residual r = r - alpha*A*p
        r_new = r - alpha * Ap

        # Calculate convergence criterion
        residual_norm = np.linalg.norm(r_new)
        residuals.append(residual_norm)

        # Print current state
        print(f"Iteration {k+1}:")
        print(f"  alpha = {alpha}")
        print(f"  x = {x}")
        print(f"  r = {r_new}")
        print(f"  |r| = {residual_norm}")

        # Check for convergence
        if residual_norm < tol:
            print(f"Converged after {k+1} iterations!")
            break

        # Calculate beta = (r_new^T * r_new) / (r^T * r)
        beta = (r_new @ r_new) / rTr

        # Update search direction p = r_new + beta*p
        p = r_new + beta * p

        print(f"  beta = {beta}")
        print(f"  p = {p}")
        print("-" * 40)

        # Update residual for next iteration
        r = r_new

    return x, residuals


# Example usage with a simple 3x3 system
if __name__ == "__main__":
    # Define system: A*x = b
    A = np.array([[4, 1, 0], [1, 3, 2], [0, 2, 6]], dtype=float)

    b = np.array([1, -2, 3], dtype=float)

    # Solve using conjugate gradient
    print("Solving system using Conjugate Gradient method:")
    print(f"A = \n{A}")
    print(f"b = {b}")
    print("-" * 40)

    x, residuals = conjugate_gradient(A, b)

    print("\nFinal solution:")
    print(f"x = {x}")

    # Verify solution
    print("\nVerification:")
    print(f"A*x = {A @ x}")
    print(f"b = {b}")
    print(f"Error = {np.linalg.norm(A @ x - b)}")

    # Compare with direct solution
    direct_sol = np.linalg.solve(A, b)
    print("\nDirect solution:")
    print(f"x = {direct_sol}")
