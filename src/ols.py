"""Ordinary least squares, written by hand. Walkthrough: Lesson 1 in lessons.py."""

import numpy as np


def add_intercept(x):
    """Put a column of 1s in front of x (1-D or 2-D). The result is the design matrix X."""
    # Why: the column of 1s gives each industry its own average return (alpha),
    # so beta and gamma don't get bent to absorb it.
    ones = np.ones(len(x))
    return np.column_stack([ones, x])


def solve(A, c):
    """Solve A b = c for b by Gaussian elimination with partial pivoting.

    A is square (k x k). c is a vector (k,) or a matrix (k, m); b has the same shape as c.
    """
    A = np.array(A, dtype=float)  # copies, so the caller's arrays aren't changed
    c = np.array(c, dtype=float)
    k = A.shape[0]
    scale = np.abs(A).max()

    # Phase 1, forward elimination: zeros below the diagonal
    for col in range(k):
        # Why: swapping in the biggest pivot avoids dividing by a tiny number (rounding blow-up)
        pivot_row = col + np.argmax(np.abs(A[col:, col]))
        if abs(A[pivot_row, col]) < 1e-12 * scale:
            raise ValueError("No unique solution: the columns of X are linearly dependent")
        if pivot_row != col:
            A[[col, pivot_row]] = A[[pivot_row, col]]
            c[[col, pivot_row]] = c[[pivot_row, col]]

        for row in range(col + 1, k):
            factor = A[row, col] / A[col, col]
            A[row] = A[row] - factor * A[col]
            c[row] = c[row] - factor * c[col]

    # Phase 2, back substitution: last unknown first, then work upward
    b = np.zeros_like(c)
    for row in range(k - 1, -1, -1):
        already_solved = A[row, row + 1:] @ b[row + 1:]
        b[row] = (c[row] - already_solved) / A[row, row]
    return b


def ols(X, y):
    """OLS coefficients: solve the normal equations X'X b = X'y.

    X: (days, k) with the column of 1s included. y: (days,) or (days, n_series).
    Returns b: (k,) or (k, n_series), one row per column of X.
    """
    XtX = X.T @ X
    Xty = X.T @ y
    return solve(XtX, Xty)


def residuals(X, y, b):
    """Misses: actual minus predicted."""
    return y - X @ b
