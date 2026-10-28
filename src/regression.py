import numpy as np

def ols_simple(x, y):

    if len(x) != len(y):
        raise ValueError("x and y must be the same length")

    n = len(x)
    x_bar = sum(x) / n
    y_bar = sum(y) / n

    sum_cross_dev = 0
    for x_i, y_i in zip(x,y):
        sum_cross_dev += (x_i - x_bar) * (y_i - y_bar)


    sum_sq_x_dev = 0
    for x_i in x:
        sum_sq_x_dev += (x_i - x_bar)**2

    if sum_sq_x_dev == 0:
      raise ValueError("x has no variation; slope is undefined")

    b1 = sum_cross_dev / sum_sq_x_dev
    b0 = y_bar - b1 * x_bar

    return b0, b1

def transpose(A):

    num_rows = len(A)
    num_cols = len(A[0])

    trans_A = []
    for j in range(num_cols):
        trans_A.append([0] * num_rows)

    for i in range(num_rows):
        for j in range(num_cols):
            trans_A[j][i] = A[i][j]

    return trans_A

def dot_prod(row, column):

    if len(row) != len(column):
        raise ValueError("row and column must be the same length")

    ans = 0
    for i in range(len(row)):
        ans += row[i] * column[i]

def matrix_mult(A, B):

    if len(A[0]) != len(B):
        raise ValueError("columns of A must equal rows of B")

    # build an empty (rows of A) × (cols of B) grid
    C = []
    for i in range(len(A)):
        C.append([0] * len(B[0]))

    for i in range(len(A)):
        for j in range(len(B[0])):
            # collect column j of B
            column_j = []
            for k in range(len(B)):
                column_j.append(B[k][j])

            C[i][j] = dot_prod(A[i], column_j)
    
    return C

def ols(X, y):

    




    return