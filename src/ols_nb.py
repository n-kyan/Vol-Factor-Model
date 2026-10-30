import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt

    return mo, np, plt


@app.cell(hide_code=True)
def _(plt):
    # Chart styling shared by every figure in this notebook
    BLUE = "#2a78d6"     # data points and fitted lines
    ORANGE = "#eb6834"   # misses (residuals) and "wrong" lines
    GRAY = "#8a8985"     # reference lines
    INK = "#52514e"      # axis text


    def new_figure(n_panels=1, width=6.0, height=3.6):
        """Create a figure with n side-by-side panels and quiet axes."""
        fig, axes = plt.subplots(1, n_panels, figsize=(width, height), squeeze=False)
        axes = axes[0]
        for ax in axes:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_color(GRAY)
            ax.spines["bottom"].set_color(GRAY)
            ax.tick_params(colors=INK)
            ax.grid(alpha=0.2)
        return fig, axes

    return BLUE, GRAY, ORANGE, new_figure


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # How the regressions are solved: OLS from scratch 🤖

    *Companion to `main.py`. Code: `src/ols.py`. Used by every regression in the project: the vol shock (`factors_nb.py`) and the industry sensitivities (`estimation_nb.py`).*

    **Bottom line**

    - **OLS (ordinary least squares)** picks the coefficients that make the squared misses as small as possible.
    - The best coefficients solve the **normal equations** X'X b = X'y. The project solves them with its own **Gaussian elimination** (`solve` in `ols.py`), not a library estimator; NumPy is used only for fast array math.
    - Every result is checked against NumPy's `lstsq` answer key, and fitting all 49 industries at once gives exactly the same answer as fitting them one at a time.

    ## 0. How to read this notebook

    **🧠 Predict boxes.** Before some code cells there's a question. Guess the answer *before* you look at the output. Predicting first is what turns reading into understanding.

    ### Words you'll see

    | Word | What it means |
    |---|---|
    | **regression** | Finding the line (or flat surface, with more inputs) that best describes how one thing moves with others. |
    | **regressor** | An input used to explain something. In the project: Mkt-RF and the vol shock. Also called an "explanatory variable". |
    | **coefficient** | The number a regressor gets multiplied by. Slope 1.99 means "y goes up 1.99 for every 1 that x goes up". |
    | **intercept** | The line's height when every regressor is 0. Written $b_0$. |
    | **residual** | A **miss**: the actual value minus the line's prediction. Written $e$ (for "error"). |
    | **vector** | An ordered list of numbers, e.g. one number per day. In code: a 1-D NumPy array. |
    | **matrix** | A table of numbers with rows and columns. Its **shape** is (rows × columns). In code: a 2-D NumPy array. |
    | **design matrix** | The matrix $X$ holding all the regressors: one row per data point (day), one column per regressor. |

    ### Symbols you'll see

    | Symbol | Say it as | What it means | Python |
    |---|---|---|---|
    | $x_i$ | "x sub i" | The $i$-th value of x. The little $i$ is just a position. | `x[i]` |
    | $n$ | "n" | How many data points there are. | `len(x)` |
    | $\sum_i x_i$ | "the sum of x" | Add up $x_i$ for every position $i$. | the loop below |
    | $\hat y_i$ | "y-hat sub i" | The line's **predicted** value for point $i$. The hat means "estimate". | `b0 + b1 * x[i]` |
    | $e_i$ | "e sub i" | The miss for point $i$: $y_i - \hat y_i$. | `y[i] - y_hat[i]` |
    | $\min_{b_0,\,b_1}$ | "minimize over b-zero and b-one" | Choose the $b_0$ and $b_1$ that make what follows as small as possible. | — |
    | $X^\top$ | "X transpose" | $X$ with rows and columns swapped (§3). | `X.T` |
    | $AB$ | "A times B" | Matrix multiplication (§3). | `A @ B` |
    | $A^{-1}$ | "A inverse" | The matrix version of dividing (§5). | — |
    | $\Longrightarrow$ | "which means" | The left side leads to the right side. | — |

    Every $\sum$ is just a loop that keeps a running total:

    ```python
    total = 0
    for i in range(n):
        total += x[i]      # this is  Σ x_i
    ```

    `for i in range(n): total += x[i] * y[i]` is $\sum_i x_i y_i$. Whatever sits to the right of $\sum$ is what goes inside the loop.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. What OLS does

    Here are five points that almost lie on a straight line:

    | $i$ | 0 | 1 | 2 | 3 | 4 |
    |---|---|---|---|---|---|
    | $x_i$ | 1 | 2 | 3 | 4 | 5 |
    | $y_i$ | 2.1 | 3.9 | 6.2 | 7.8 | 10.1 |

    We want the line $\hat y = b_0 + b_1 x$ that comes as close to them as possible. Two numbers pick the line:

    - $b_0$, the **intercept**: how high the line sits.
    - $b_1$, the **slope**: how steep it is.

    **The miss** at each point is the vertical gap between the dot and the line: $e_i = y_i - \hat y_i$. If the dot is above the line, the miss is positive. If it's below, the miss is negative. In the chart below, the orange segments are the misses.

    **Why square the misses?** Just adding them up fails: positive and negative misses cancel out, so a terrible line could total 0. Squaring makes every miss count as positive. There are two reasons to square rather than take absolute values:

    1. Squaring punishes big misses much more than small ones. A miss of 2 counts 4, a miss of 4 counts 16. So the line refuses to badly miss any single point.
    2. Squaring gives a smooth, bowl-shaped total with exactly one lowest point (§4). That lets us solve for the best line with algebra instead of searching for it.

    **OLS** ("ordinary least squares", i.e. least squared misses) picks the $b_0, b_1$ that make the total of the squared misses as small as possible:

    $$
    \min_{b_0,\,b_1} \; \sum_i \big(y_i - b_0 - b_1 x_i\big)^2
    $$

    Read it as: "out of all possible intercepts and slopes, pick the pair whose sum of squared misses is smallest."

    Your `ols_simple` already solves this for one $x$. It gives $b_0 = 0.05$ and $b_1 = 1.99$. Every method later in this notebook must reproduce those two numbers.

    > **🧠 Predict:** the next cell compares the OLS line with a slightly steeper line and a slightly higher line. Which will have the smallest total of squared misses?
    """)
    return


@app.cell
def _():
    x_toy = [1, 2, 3, 4, 5]
    y_toy = [2.1, 3.9, 6.2, 7.8, 10.1]


    def sum_sq_misses(x, y, b0, b1):
        """Total of the squared misses for the line b0 + b1 * x."""
        total = 0
        for i in range(len(x)):
            prediction = b0 + b1 * x[i]
            miss = y[i] - prediction
            total += miss ** 2
        return total


    print("OLS line   b0=0.05, b1=1.99 → total:", round(sum_sq_misses(x_toy, y_toy, 0.05, 1.99), 4))
    print("Steeper    b0=0.05, b1=2.10 → total:", round(sum_sq_misses(x_toy, y_toy, 0.05, 2.10), 4))
    print("Higher     b0=0.30, b1=1.99 → total:", round(sum_sq_misses(x_toy, y_toy, 0.30, 1.99), 4))
    return sum_sq_misses, x_toy, y_toy


@app.cell(hide_code=True)
def _(BLUE, ORANGE, new_figure, np, sum_sq_misses, x_toy, y_toy):
    _lines = [("OLS line", 0.05, 1.99), ("Steeper", 0.05, 2.10), ("Higher", 0.30, 1.99)]
    _fig, _axes = new_figure(n_panels=3, width=11, height=3.6)
    _x = np.array(x_toy)
    _y = np.array(y_toy)
    _x_grid = np.linspace(0.5, 5.5, 50)

    for _ax, (_name, _b0, _b1) in zip(_axes, _lines):
        _prediction = _b0 + _b1 * _x
        _ax.plot(_x_grid, _b0 + _b1 * _x_grid, color=BLUE, linewidth=1.5, label="line")
        _ax.vlines(_x, _prediction, _y, color=ORANGE, linewidth=4, zorder=4, label="miss")
        _ax.scatter(_x, _y, color=BLUE, s=30, zorder=5, label="data")
        _total = sum_sq_misses(x_toy, y_toy, _b0, _b1)
        _ax.set_title(f"{_name}: b0={_b0}, b1={_b1}\ntotal squared miss = {_total:.3f}", fontsize=10)
        _ax.set_xlabel("x")
    _axes[0].set_ylabel("y")
    _axes[0].legend(frameon=False, loc="upper left")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The OLS line has the smallest total. The misses are small next to the y-axis range, so they're hard to see here; §4 zooms in on them. The other two lines miss in one direction: the steeper one ends up too high on the right, and the higher one is too high everywhere.

    Two other lines are only a **demo**, not a proof. Section 4 explains why *no* line can beat the OLS line.

    ## 2. Matrix form: $y = Xb + e$

    The project has two regressors (Mkt-RF and the vol shock), not one. Writing out every sum by hand gets messy fast. So we pack the data into a vector and a matrix, and let a single formula handle any number of regressors.

    **The pieces:**

    - $y$: a vector holding the thing we're explaining, one entry per data point. Shape $(n,)$.
    - $X$: the design matrix. **One row per data point, one column per regressor**, plus a first column that is all 1s (explained below). Shape $(n, k)$, where $k$ is the number of columns.
    - $b$: a vector of coefficients, **one per column of $X$**. Shape $(k,)$.
    - $e$: a vector of misses, one per data point.

    For the toy data, $n = 5$ points and $k = 2$ columns:

    $$
    \underbrace{\begin{bmatrix}2.1\\3.9\\6.2\\7.8\\10.1\end{bmatrix}}_{y} =
    \underbrace{\begin{bmatrix}1&1\\1&2\\1&3\\1&4\\1&5\end{bmatrix}}_{X}
    \underbrace{\begin{bmatrix}b_0\\b_1\end{bmatrix}}_{b} + \underbrace{\begin{bmatrix}e_0\\e_1\\e_2\\e_3\\e_4\end{bmatrix}}_{e}
    $$

    **What $Xb$ means.** Take each row of $X$, multiply it entry by entry with $b$, and add up. Row 0 is $[1, 1]$, so its prediction is $1 \cdot b_0 + 1 \cdot b_1$. With $b = [0.05, 1.99]$ that's $0.05 + 1.99 = 2.04$. The actual $y_0$ is $2.1$, so the miss is $e_0 = 2.1 - 2.04 = 0.06$. $Xb$ does this for every row at once. That's what makes the matrix form compact. The next cell shows the loop and NumPy's `@` giving the same predictions.

    **In the project,** each industry's row for a day will be $[\,1,\ \text{Mkt-RF},\ \text{vol shock}\,]$, and $b = [\alpha, \beta, \gamma]$:

    - $\alpha$ (alpha) is the intercept.
    - $\beta$ (beta) is the market sensitivity.
    - $\gamma$ (gamma) is the vol-shock sensitivity.
    """)
    return


@app.cell
def _(np):
    def add_intercept(x):
        """Put a column of 1s in front of x. The result is the design matrix X."""
        ones = np.ones(len(x))
        return np.column_stack([ones, x])


    def predict_py(X, b):
        """Plain-Python Xb: for each row, multiply entry by entry with b and add up."""
        predictions = []
        for row in X:
            total = 0
            for j in range(len(b)):
                total += row[j] * b[j]
            predictions.append(total)
        return predictions


    _X = add_intercept(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    _b = np.array([0.05, 1.99])
    print("X =\n", _X)
    print("loop predictions:", np.round(predict_py(_X, _b), 4))
    print("X @ b           :", np.round(_X @ _b, 4))
    return (add_intercept,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Why the column of 1s?

    The column of 1s is what lets the line have its own height, $b_0$. Row $i$'s prediction is $1 \cdot b_0 + x_i \cdot b_1$, and the "1 ·" is what keeps $b_0$ in every prediction. Without that column, the formula becomes $\hat y = b_1 x$. A line like that must pass through $(0, 0)$: when $x = 0$, the prediction is 0 no matter what.

    Here's why that matters. Below, $y$ hovers around 5 and has **nothing to do with x**. The right answer is a flat line: slope 0, height 5.

    > **🧠 Predict:** if the line is forced through (0, 0), what will its slope do?
    """)
    return


@app.cell
def _(BLUE, GRAY, ORANGE, add_intercept, new_figure, np, ols):
    x_flat = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_flat = np.array([5.1, 4.9, 5.0, 5.2, 4.8])

    # Fitted with `ols` from §6 (marimo runs cells in dependency order, not page order)
    _height_with, _slope_with = ols(add_intercept(x_flat), y_flat)
    _slope_without = ols(x_flat.reshape(-1, 1), y_flat)[0]

    _fig, _axes = new_figure(n_panels=2, width=9, height=3.6)
    _x_grid = np.linspace(0, 5.5, 50)
    _titles = [f"With column of 1s\nslope = {_slope_with:.2f} (correct: about 0)",
               f"Without (forced through 0, 0)\nslope = {_slope_without:.2f} (invented)"]
    _lines = [_height_with + _slope_with * _x_grid, _slope_without * _x_grid]
    _colors = [BLUE, ORANGE]
    for _ax, _title, _line, _color in zip(_axes, _titles, _lines, _colors):
        _ax.scatter(x_flat, y_flat, color=BLUE, s=40, zorder=3)
        _ax.plot(_x_grid, _line, color=_color, linewidth=2)
        _ax.scatter([0], [0], color=GRAY, s=25, marker="x")
        _ax.set_ylim(-0.5, 8)
        _ax.set_title(_title, fontsize=10)
        _ax.set_xlabel("x")
    _axes[0].set_ylabel("y")
    _fig.tight_layout()
    _fig
    return x_flat, y_flat


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Forced through $(0, 0)$, the only way the line can reach dots that sit at height 5 is to tilt upward. So it reports a slope of 1.36 that isn't real. With the column of 1s, the line can just sit at height 5, and it correctly reports a slope of about 0.

    **Project link.** Each industry has its own average daily return. Without the intercept, $\beta$ and $\gamma$ would get bent to absorb that average, the same way the slope got bent here.

    ## 3. The matrix tools: transpose, dot product, matrix multiply

    OLS needs three operations. Each one is shown in plain Python and then checked against NumPy. NumPy does **exactly the same arithmetic**, just in fast compiled code, which is why we use it in the project.

    ### Transpose: $A^\top$, `A.T`

    Swap rows and columns. Row 0 becomes column 0, row 1 becomes column 1, and so on. A $(3 \times 2)$ matrix becomes $(2 \times 3)$:

    $$
    A = \begin{bmatrix}1&2\\3&4\\5&6\end{bmatrix}
    \qquad
    A^\top = \begin{bmatrix}1&3&5\\2&4&6\end{bmatrix}
    $$

    ### Dot product: $u \cdot v$

    Take two vectors of the same length, multiply them entry by entry, and add up the results:

    $$
    [1, 2, 3] \cdot [4, 5, 6] = 1\cdot4 + 2\cdot5 + 3\cdot6 = 32
    $$

    ### Matrix multiply: $AB$, `A @ B`

    Entry (row $i$, column $j$) of the result is **row $i$ of $A$ dotted with column $j$ of $B$**. That only works if a row of $A$ is as long as a column of $B$, i.e. $A$ has as many columns as $B$ has rows. The result has $A$'s number of rows and $B$'s number of columns.

    For the example below, the top-left entry of $AB$ is row 0 of $A$, $[1, 2]$, dotted with column 0 of $B$, $[1, 0]$: $1\cdot1 + 2\cdot0 = 1$.
    """)
    return


@app.cell
def _():
    def transpose_py(A):
        """Rows become columns."""
        n_rows = len(A)
        n_cols = len(A[0])
        result = []
        for j in range(n_cols):
            new_row = []
            for i in range(n_rows):
                new_row.append(A[i][j])
            result.append(new_row)
        return result


    def dot_py(u, v):
        """Multiply entry by entry, then add up."""
        if len(u) != len(v):
            raise ValueError("u and v must be the same length")
        total = 0
        for i in range(len(u)):
            total += u[i] * v[i]
        return total


    def matmul_py(A, B):
        """Entry (i, j) = row i of A dotted with column j of B."""
        if len(A[0]) != len(B):
            raise ValueError("columns of A must equal rows of B")
        columns_of_B = transpose_py(B)  # the columns of B, each as a list
        result = []
        for row in A:
            new_row = []
            for column in columns_of_B:
                new_row.append(dot_py(row, column))
            result.append(new_row)
        return result

    return dot_py, matmul_py, transpose_py


@app.cell
def _(dot_py, matmul_py, np, transpose_py):
    A_small = [[1, 2],
               [3, 4],
               [5, 6]]
    B_small = [[1, 0, 2],
               [0, 1, 3]]

    print("transpose_py(A):", transpose_py(A_small))
    print("A.T            :", np.array(A_small).T.tolist())
    print()
    print("dot_py([1,2,3], [4,5,6]):", dot_py([1, 2, 3], [4, 5, 6]))
    print("np.dot                  :", np.dot([1, 2, 3], [4, 5, 6]))
    print()
    print("matmul_py(A, B):", matmul_py(A_small, B_small))
    print("A @ B          :", (np.array(A_small) @ np.array(B_small)).tolist())
    print()
    print("All match:",
          np.allclose(transpose_py(A_small), np.array(A_small).T),
          np.allclose(matmul_py(A_small, B_small), np.array(A_small) @ np.array(B_small)))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Dot product = 0 means perpendicular

    This one fact powers all of §4. **Two vectors are perpendicular (at a right angle) exactly when their dot product is 0.**

    In 2 dimensions you can see it. Draw each vector as an arrow from (0, 0):

    - $u = [3, 1]$ and $v = [-1, 3]$: $\;u \cdot v = 3\cdot(-1) + 1\cdot3 = 0$. They form a right angle.
    - $u = [3, 1]$ and $w = [1, 2]$: $\;u \cdot w = 3\cdot1 + 1\cdot2 = 5$. They don't.

    The toy data has 5 points, so its vectors ($y$, $e$, each column of $X$) have 5 entries. That makes them arrows in 5 dimensions, which nobody can draw. So in 5 dimensions, "dot product = 0" isn't a test for perpendicular: it **is** the definition of perpendicular. Same for 9,228 dimensions, one per trading day in the project.
    """)
    return


@app.cell
def _(BLUE, GRAY, ORANGE, new_figure):
    _fig, _axes = new_figure(n_panels=2, width=8, height=3.8)
    _pairs = [("u · v = 0 → perpendicular", [-1, 3], BLUE),
              ("u · w = 5 → not perpendicular", [1, 2], ORANGE)]
    for _ax, (_title, _other, _color) in zip(_axes, _pairs):
        _ax.annotate("", xy=(3, 1), xytext=(0, 0),
                     arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=2))
        _ax.annotate("", xy=tuple(_other), xytext=(0, 0),
                     arrowprops=dict(arrowstyle="-|>", color=_color, lw=2))
        _ax.text(3.05, 0.85, "u = [3, 1]", color=GRAY)
        _ax.text(_other[0] + 0.1, _other[1] + 0.1, f"{_other}", color=_color)
        _ax.set_xlim(-2, 4.5)
        _ax.set_ylim(-0.5, 3.8)
        _ax.set_aspect("equal")
        _ax.set_title(_title, fontsize=10)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. The normal equations: where the OLS formula comes from

    ### Step 1: what the misses look like when a line is *not* the best

    Five points are too few to see a pattern, so this step uses **100 made-up points** that follow a line plus random noise. Both panels show the **same 100 points**; only the line changes:

    - **Left:** the OLS line for these points.
    - **Right:** a line that's too steep. It's tilted through the middle of the data so that it's right on average, just at the wrong angle.

    Each thin vertical segment is one point's miss. **Green** means the point is above the line (positive miss), and **red** means it's below (negative miss).

    > **🧠 Predict:** on the too-steep line, where will most of the green segments be, left or right?
    """)
    return


@app.cell
def _(BLUE, GRAY, ORANGE, new_figure, np, ols):
    GREEN = "#1baf7a"   # point above the line (positive miss)
    RED = "#e34948"     # point below the line (negative miss)

    _rng = np.random.default_rng(7)
    x_many = np.sort(_rng.uniform(0, 10, size=100))
    y_many = 1.0 + 2.0 * x_many + _rng.normal(scale=2.0, size=100)

    _X = np.column_stack([np.ones(100), x_many])
    _b0_ols, _b1_ols = ols(_X, y_many)
    # Too steep, but pivoted through the center of the data (mean x, mean y)
    _b1_steep = _b1_ols + 0.8
    _b0_steep = y_many.mean() - _b1_steep * x_many.mean()

    _cases = [("OLS line", _b0_ols, _b1_ols, BLUE),
              ("Too-steep line", _b0_steep, _b1_steep, ORANGE)]
    _fig, _axes = new_figure(n_panels=2, width=11, height=4.2)
    _x_grid = np.array([0.0, 10.0])
    for _ax, (_name, _b0, _b1, _color) in zip(_axes, _cases):
        _pred = _b0 + _b1 * x_many
        _miss = y_many - _pred
        _above = _miss > 0
        _ax.vlines(x_many[_above], _pred[_above], y_many[_above], color=GREEN, linewidth=1.2, label="above line (+)")
        _ax.vlines(x_many[~_above], _pred[~_above], y_many[~_above], color=RED, linewidth=1.2, label="below line (−)")
        _ax.scatter(x_many, y_many, color=GRAY, s=10, zorder=3)
        _ax.plot(_x_grid, _b0 + _b1 * _x_grid, color=_color, linewidth=2.5, zorder=4)
        _left = _miss[x_many < 5].mean()
        _right = _miss[x_many >= 5].mean()
        _ax.set_title(f"{_name} (slope {_b1:.2f})\naverage miss: left half {_left:+.2f}, right half {_right:+.2f}", fontsize=10)
        _ax.set_xlabel("x")
        _ax.set_ylim(-5, 27)  # same scale in both panels so they compare directly
    _axes[0].set_ylabel("y")
    _axes[0].legend(frameon=False, loc="upper left", fontsize=9)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - **Too-steep line:** on the left the line sits too low, so most points are above it (green). On the right it sits too high, so most are below it (red). The misses go from mostly positive to mostly negative as $x$ grows. That pattern is the giveaway: tilt the line flatter and the misses on both sides shrink.
    - **OLS line:** green and red are mixed evenly all the way across, and the average miss is about 0 on both halves. There's no pattern left, so no tilt or shift can make the misses smaller.

    That's the whole idea behind the best line: **its misses have no pattern that a different line could fix.** The next step turns "no pattern" into equations.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Step 2: turn "no drift" into two equations

    A line is the best possible when two things are both true of its misses:

    1. **They don't lean up or down overall:** $\sum_i e_i = 0$. If the misses added up to something positive, raising the line ($b_0$ up) would shrink them.
    2. **They don't drift with $x$:** $\sum_i x_i e_i = 0$. If big-$x$ points had mostly negative misses, the sum would come out negative, and flattening the line ($b_1$ down) would shrink them.

    > **🧠 Predict:** compute both sums for the OLS line and for the steeper line. Which one gives two zeros?
    """)
    return


@app.cell
def _(x_toy, y_toy):
    def no_drift_sums(x, y, b0, b1):
        """Return (sum of misses, sum of x * miss) for the line b0 + b1 * x."""
        sum_e = 0
        sum_xe = 0
        for i in range(len(x)):
            miss = y[i] - (b0 + b1 * x[i])
            sum_e += miss
            sum_xe += x[i] * miss
        return round(sum_e, 10), round(sum_xe, 10)


    print("OLS line : (Σe, Σxe) =", no_drift_sums(x_toy, y_toy, 0.05, 1.99))
    print("Steeper  : (Σe, Σxe) =", no_drift_sums(x_toy, y_toy, 0.05, 2.10))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Step 3: those two sums are dot products, i.e. $X^\top e$

    Look at the two sums again with §3 in mind:

    - $\sum_i e_i = \sum_i 1 \cdot e_i$ is the **column of 1s dotted with $e$**.
    - $\sum_i x_i e_i$ is the **x column dotted with $e$**.

    So "no drift" means: **every column of $X$, dotted with $e$, gives 0.** By §3, that means $e$ is perpendicular to every column of $X$.

    Matrix multiplication can do all those dot products in one step:

    - $X^\top$ turns each **column** of $X$ into a **row**. Row 0 of $X^\top$ is the 1s, and row 1 is the x values.
    - $X^\top e$ dots each row of $X^\top$ with $e$. That's exactly the list of "column · $e$" sums.

    $$
    X^\top e =
    \begin{bmatrix} 1&1&1&1&1 \\ 1&2&3&4&5 \end{bmatrix}
    \begin{bmatrix} e_0\\e_1\\e_2\\e_3\\e_4 \end{bmatrix}
    =
    \begin{bmatrix} \sum_i e_i \\ \sum_i x_i e_i \end{bmatrix}
    $$

    So the whole best-line condition is one short equation: $\;X^\top e = 0$.

    This works for any number of columns. With $[1, \text{Mkt-RF}, \text{vol shock}]$ it says the misses don't lean overall, don't drift with the market, and don't drift with the vol shock.

    ### Step 4: the algebra, one move at a time

    Start from $X^\top e = 0$. A miss is actual minus predicted, so replace $e$ with $y - Xb$:

    $$
    X^\top (y - Xb) = 0
    $$

    Multiply out the bracket, the same as $a(c - d) = ac - ad$. Keep $X^\top$ on the **left** of each term, because with matrices the order of multiplication matters:

    $$
    X^\top y - X^\top X b = 0
    $$

    Add $X^\top X b$ to both sides to move it across:

    $$
    \boxed{\,X^\top X\, b = X^\top y\,}
    $$

    These are the **normal equations**. "Normal" is an old word for perpendicular, because they say the misses are perpendicular to every column of $X$. Everything in them is known except $b$, so solving them gives the OLS coefficients.

    ### Why this is always the best line, not just better than two others

    Fix $b_0$ and try every slope $b_1$, recording each slope's total of squared misses. The left chart below shows the result: a **smooth bowl** with a single lowest point. At the very bottom the bowl is flat, and that flat spot is exactly where the drift sums from Step 2 are 0. With two coefficients the bowl is a 3-D bowl, but the idea is the same. There is **one** bottom, and the normal equations find it directly.

    The right chart shows what happens with absolute misses instead of squared ones. The curve is a V with sharp corners, and there's no simple equation for "the bottom of a V". That's reason 2 for squaring from §1.
    """)
    return


@app.cell
def _(BLUE, GRAY, ORANGE, new_figure, np, sum_sq_misses, x_toy, y_toy):
    def sum_abs_misses(x, y, b0, b1):
        """Total of the absolute misses for the line b0 + b1 * x."""
        total = 0
        for i in range(len(x)):
            total += abs(y[i] - (b0 + b1 * x[i]))
        return total


    _slopes = np.linspace(1.5, 2.5, 201)
    _sq = []
    _ab = []
    for _s in _slopes:
        _sq.append(sum_sq_misses(x_toy, y_toy, 0.05, _s))
        _ab.append(sum_abs_misses(x_toy, y_toy, 0.05, _s))

    _fig, _axes = new_figure(n_panels=2, width=9, height=3.4)
    _axes[0].plot(_slopes, _sq, color=BLUE, linewidth=2)
    _axes[0].axvline(1.99, color=GRAY, linestyle="--", linewidth=1)
    _axes[0].set_title("Squared misses: smooth bowl\nbottom at b1 = 1.99", fontsize=10)
    _axes[0].set_ylabel("total")
    _axes[1].plot(_slopes, _ab, color=ORANGE, linewidth=2)
    _axes[1].set_title("Absolute misses: V with sharp corners", fontsize=10)
    for _ax in _axes:
        _ax.set_xlabel("slope b1 (with b0 fixed at 0.05)")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Step 5: $X^\top X$ is just a small table of sums

    Every entry of $X^\top X$ is "column $j$ of $X$ dotted with column $k$ of $X$". For the toy data:

    - 1s · 1s $= 1+1+1+1+1 = n = 5$
    - 1s · x $= \sum x = 15$
    - x · x $= \sum x^2 = 1+4+9+16+25 = 55$

    $X^\top y$ works the same way: each column of $X$ dotted with $y$.

    $$
    X^\top X = \begin{bmatrix} n & \sum x \\ \sum x & \sum x^2 \end{bmatrix}
    = \begin{bmatrix} 5 & 15 \\ 15 & 55 \end{bmatrix},
    \qquad
    X^\top y = \begin{bmatrix} \sum y \\ \sum xy \end{bmatrix}
    = \begin{bmatrix} 30.1 \\ 110.2 \end{bmatrix}
    $$

    $X^\top X$ is (columns × columns), so it's tiny no matter how much data there is. In the project it's 3 × 3, even with 9,228 days. All the heavy lifting happens while building it. After that, solving is cheap.

    ### Step 6: solve it by hand

    Write the normal equations out as two ordinary equations:

    - (i) $\;5 b_0 + 15 b_1 = 30.1$
    - (ii) $\;15 b_0 + 55 b_1 = 110.2$

    Take (ii) minus 3 × (i). The $b_0$ terms cancel ($15 - 3\cdot5 = 0$), leaving $10 b_1 = 19.9$, so $b_1 = 1.99$. Put that back into (i): $5 b_0 = 30.1 - 15 \cdot 1.99 = 0.25$, so $b_0 = 0.05$. Same answer as `ols_simple`.

    That two-move recipe, **(1) subtract a multiple of one equation to wipe out an unknown, (2) plug the answer back in**, is all of Gaussian elimination, the next section. The cell below checks every number in Steps 5 and 6 with NumPy and with the plain-Python functions.
    """)
    return


@app.cell
def _(add_intercept, matmul_py, np, transpose_py, x_toy, y_toy):
    X_toy = add_intercept(np.array(x_toy, dtype=float))
    y_toy_arr = np.array(y_toy)

    print("X'X =\n", X_toy.T @ X_toy)
    print("X'y =", X_toy.T @ y_toy_arr)

    _XtX_py = matmul_py(transpose_py(X_toy.tolist()), X_toy.tolist())
    print("plain-Python X'X matches NumPy:", np.allclose(_XtX_py, X_toy.T @ X_toy))

    _e_ols = y_toy_arr - X_toy @ np.array([0.05, 1.99])
    _e_steep = y_toy_arr - X_toy @ np.array([0.05, 2.10])
    print("X'e, OLS line    :", np.round(X_toy.T @ _e_ols, 10), "← both 0: perpendicular")
    print("X'e, steeper line:", np.round(X_toy.T @ _e_steep, 10), "← not 0: can be improved")
    return X_toy, y_toy_arr


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. `solve(A, c)`: Gaussian elimination

    We need $b$ from $A b = c$, where $A = X^\top X$ and $c = X^\top y$.

    ### What "inverse" means, and why we skip it

    With plain numbers, $5b = 10$ is solved by dividing: $b = 10 / 5 = 2$. Dividing by 5 is the same as multiplying by $\tfrac15$, the number that "undoes" 5, because $\tfrac15 \cdot 5 = 1$.

    Matrices have no division, but they have the same idea:

    - The **identity matrix** $I$ has 1s down the diagonal and 0s everywhere else. Multiplying by $I$ changes nothing, just like multiplying by 1.
    - The **inverse** $A^{-1}$ is the matrix that undoes $A$: $A^{-1} A = I$, just like $\tfrac15 \cdot 5 = 1$.
    - So $Ab = c$ gives $b = A^{-1} c$, just like $b = \tfrac15 \cdot 10$.

    Textbooks therefore write OLS as $b = (X^\top X)^{-1} X^\top y$. That's where your interviewer's "invert to minimize" comes from: minimizing the squared misses comes down to undoing $X^\top X$.

    **In code we never build the inverse.** For $5b = 10$ you'd just compute $10/5$ directly; you wouldn't first work out $\tfrac15 = 0.2$ and then multiply. For matrices, building $A^{-1}$ first takes more arithmetic and adds more rounding error than solving the equations directly. Gaussian elimination solves them directly.

    ### The two phases

    **Phase 1, forward elimination.** Work through the columns from left to right. For each column:

    - The equation on the diagonal is the **pivot row**, and its entry in this column is the **pivot**.
    - Subtract a multiple of the pivot row from every row *below* it, chosen so that the entry in this column becomes 0. This is the same move as "(ii) minus 3 × (i)" in §4.

    When every column is done, $A$ has only zeros below the diagonal. That shape is called **upper triangular**:

    $$
    \begin{bmatrix} \ast&\ast&\ast \\ 0&\ast&\ast \\ 0&0&\ast \end{bmatrix}
    $$

    Now the last equation has just one unknown. The one above it has two, but one of them is about to be known.

    **Phase 2, back substitution.** Solve the last equation (one division). Plug that value into the equation above it and solve for the next unknown. Repeat upward until every unknown is solved.

    > **🧠 Predict:** the cell below solves a 3 × 3 system whose answer is $b = [1, 2, 3]$. The first column is $[2, 4, -2]$. Which row will `solve` choose as the first pivot row? (Hint: read "partial pivoting" just below.)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Partial pivoting: why we swap rows first

    Before clearing a column, `solve` swaps up the row whose entry in that column is **largest in size** (ignoring the sign). There are two reasons:

    1. **A pivot of exactly 0 crashes:** you can't divide by 0.
    2. **A tiny pivot quietly gives wrong answers.** Computers store about 16 significant digits. Dividing by a tiny pivot creates a huge multiplier, and adding a huge number to a normal one wipes out the normal one's digits. The 2 × 2 example below shows it: the true answer is about $[1, 1]$, but without pivoting the computer returns $[0, 1]$.

    **The NumPy swap syntax.** `A[[0, 2]]` means "rows 0 and 2 of A, in that order". So `A[[0, 2]] = A[[2, 0]]` writes rows 2 and 0 back into slots 0 and 2. In other words, it swaps them. The next cell runs it on a small matrix.
    """)
    return


@app.cell
def _(np):
    _M = np.array([[1, 1],
                   [2, 2],
                   [3, 3]])
    print("before:\n", _M)
    _M[[0, 2]] = _M[[2, 0]]
    print("after A[[0, 2]] = A[[2, 0]]:\n", _M)
    return


@app.cell
def _(np):
    def solve_no_pivot(A, c):
        """Gaussian elimination WITHOUT row swaps. For the demo only. Don't use this."""
        A = np.array(A, dtype=float)
        c = np.array(c, dtype=float)
        k = A.shape[0]
        for col in range(k):
            for row in range(col + 1, k):
                factor = A[row, col] / A[col, col]
                A[row] = A[row] - factor * A[col]
                c[row] = c[row] - factor * c[col]
        b = np.zeros_like(c)
        for row in range(k - 1, -1, -1):
            already_solved = A[row, row + 1:] @ b[row + 1:]
            b[row] = (c[row] - already_solved) / A[row, row]
        return b

    return (solve_no_pivot,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### `solve`, line by line

    - **Copy the inputs.** `np.array(A, dtype=float)` makes a copy, so the swaps and subtractions don't change the caller's arrays.
    - **`scale`** is the biggest number in $A$ (by size). It's used in the "is this pivot basically zero?" test.
    - **The zero test: `abs(pivot) < 1e-12 * scale`.**
      - Why not `pivot == 0`? Rounding means a pivot that "should" be 0 usually comes out as something like `3e-17`, not exactly 0.
      - Why compare against `scale`? Data comes in different units: daily returns are around 0.01, VIX changes around 1, and squared sums can be in the thousands. So "tiny" has to mean tiny *relative to the other numbers in A*.
      - Why `1e-12`? Rounding errors are around `1e-16` relative to the numbers involved. Any genuine pivot in a real regression is far bigger than `1e-12` relative. So the threshold has a big safety margin on both sides.
    - **When the test fails**, there's no single answer. For OLS that means one column of $X$ can be built from the others, e.g. the same regressor entered twice. That's called **linear dependence**, the opposite of the "linear independence" your interviewer mentioned. With dependent columns, infinitely many $b$'s fit equally well, so we raise an error instead of returning garbage.
    - **Many industries in one call.** $c$ can be a single vector (one industry) or a matrix with one column per industry. Every step does the same thing to each column of $c$, so the same code covers both.
    - **`verbose=True`** prints each step so you can follow along.
    """)
    return


@app.cell
def _(np):
    def solve(A, c, verbose=False):
        """Solve A b = c for b by Gaussian elimination with partial pivoting.

        A is square (k x k). c is a vector (k,) or a matrix (k, m); b has the same shape as c.
        """
        A = np.array(A, dtype=float)  # copies, so the caller's arrays aren't changed
        c = np.array(c, dtype=float)
        k = A.shape[0]
        scale = np.abs(A).max()

        # Phase 1, forward elimination: zeros below the diagonal
        for col in range(k):
            # Partial pivoting: find the biggest entry (by size) at or below the diagonal
            pivot_row = col + np.argmax(np.abs(A[col:, col]))
            if abs(A[pivot_row, col]) < 1e-12 * scale:
                raise ValueError("No unique solution: the columns of X are linearly dependent")
            if pivot_row != col:
                A[[col, pivot_row]] = A[[pivot_row, col]]
                c[[col, pivot_row]] = c[[pivot_row, col]]
                if verbose:
                    print(f"swap rows {col} and {pivot_row} (row {pivot_row} has the biggest entry in column {col})")

            for row in range(col + 1, k):
                factor = A[row, col] / A[col, col]
                A[row] = A[row] - factor * A[col]
                c[row] = c[row] - factor * c[col]
                if verbose:
                    print(f"  row {row} minus {factor:g} × row {col} → column {col} entry becomes 0")

            if verbose and col < k - 1:
                print(f"after clearing column {col}:\nA =\n{A}\nc = {c}\n")

        # Phase 2, back substitution: last unknown first, then work upward
        b = np.zeros_like(c)
        for row in range(k - 1, -1, -1):
            already_solved = A[row, row + 1:] @ b[row + 1:]  # contribution of unknowns solved so far
            b[row] = (c[row] - already_solved) / A[row, row]
            if verbose:
                print(f"back-substitute row {row}: b[{row}] = {b[row]:g}")
        return b

    return (solve,)


@app.cell
def _(np, solve):
    # A 3x3 system built to have the answer b = [1, 2, 3]
    A_3 = np.array([[2.0, 1.0, 1.0],
                    [4.0, -6.0, 0.0],
                    [-2.0, 7.0, 2.0]])
    b_true_3 = np.array([1.0, 2.0, 3.0])
    c_3 = A_3 @ b_true_3

    b_3 = solve(A_3, c_3, verbose=True)
    print("\nsolved b:", b_3, "| matches the answer [1, 2, 3]:", np.allclose(b_3, b_true_3))
    return


@app.cell
def _(np, solve, solve_no_pivot):
    # Tiny-pivot demo: the true answer is extremely close to [1, 1]
    _A_tiny = np.array([[1e-20, 1.0],
                        [1.0, 1.0]])
    _c_tiny = np.array([1.0, 2.0])
    print("without pivoting:", solve_no_pivot(_A_tiny, _c_tiny), "← wrong")
    print("with pivoting   :", solve(_A_tiny, _c_tiny), "← right")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Why the version without pivoting fails: its first pivot is $10^{-20}$, so the multiplier is $10^{20}$. Row 1 becomes $1 - 10^{20}$, and the computer can't hold that many digits, so it stores just $-10^{20}$. The "1" is lost. From there the answer for $b_0$ comes out as 0 instead of 1. Swapping rows first makes the pivot 1 instead, and nothing gets lost.

    > **🧠 Predict:** in the next cell, the third column of $X$ is exactly 2 × the second column. What will `solve` do?
    """)
    return


@app.cell
def _(np, solve):
    # Linear dependence: column 2 = 2 × column 1, so no unique answer exists
    _X_bad = np.array([[1.0, 1.0, 2.0],
                       [1.0, 2.0, 4.0],
                       [1.0, 3.0, 6.0],
                       [1.0, 4.0, 8.0]])
    try:
        solve(_X_bad.T @ _X_bad, _X_bad.T @ np.ones(4))
    except ValueError as err:
        print("ValueError:", err)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    It refuses. If the third column is just 2 × the second, then "1 more of column 2" and "0.5 more of column 3" change the predictions identically. No amount of data can tell them apart.

    ## 6. `ols(X, y)` and `residuals(X, y, b)`

    With `solve` built, OLS is two steps: build the normal equations, then solve them.
    """)
    return


@app.cell
def _(solve):
    def ols(X, y):
        """OLS coefficients: solve the normal equations X'X b = X'y."""
        XtX = X.T @ X
        Xty = X.T @ y
        return solve(XtX, Xty)


    def residuals(X, y, b):
        """Misses: actual minus predicted."""
        return y - X @ b

    return ols, residuals


@app.cell
def _(X_toy, add_intercept, np, ols, x_flat, y_flat, y_toy_arr):
    # Check 1: the toy data
    b_toy = ols(X_toy, y_toy_arr)
    print("toy data b:", np.round(b_toy, 6), "| matches ols_simple (0.05, 1.99):", np.allclose(b_toy, [0.05, 1.99]))

    # Check 2: the intercept demo from §2
    _b_with = ols(add_intercept(x_flat), y_flat)
    _b_without = ols(x_flat.reshape(-1, 1), y_flat)  # X with only the x column
    print(f"flat data, with column of 1s: height {_b_with[0]:.2f}, slope {_b_with[1]:.3f}")
    print(f"flat data, without          : slope {_b_without[0]:.3f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Check 3: invent a truth, then see if `ols` finds it

    With real data you never know the true coefficients, so you can't tell whether a method is right. The trick is to **make up the data ourselves**: pick true coefficients, generate $y$ from them, and see whether `ols` recovers them.

    - $x_1$ and $x_2$ are random numbers drawn from a **normal distribution**, the familiar bell curve. With NumPy's defaults the values center on 0, and about two-thirds of them land between −1 and +1.
    - $y = 0.5 + 2.0\,x_1 - 1.0\,x_2 + \text{noise}$. So the true $b$ is $[0.5,\ 2.0,\ -1.0]$.
    - The **noise** is more bell-curve random numbers, scaled down by `scale=0.5` so about two-thirds land between −0.5 and +0.5. It stands in for everything the regressors don't explain, like an industry's own company news.

    **Why `ols` won't land exactly on the truth:** the noise. With only a few hundred points, the random noise happens to line up a little with $x_1$ or $x_2$, and `ols` can't tell that apart from a real effect. **More data means the noise averages out, so the estimates get closer to the truth.** The cell after this one shows that directly.

    We also check $X^\top e = 0$ (perpendicular misses) and compare against NumPy's built-in `lstsq` as an answer key. `lstsq` appears only here, as an answer key. The project uses our own `ols`.
    """)
    return


@app.cell
def _(add_intercept, np, ols, residuals):
    _rng = np.random.default_rng(0)  # seeded so the "random" numbers are the same every run
    _n = 500
    _x1 = _rng.normal(size=_n)
    _x2 = _rng.normal(size=_n)
    X_sim = add_intercept(np.column_stack([_x1, _x2]))
    _b_true = np.array([0.5, 2.0, -1.0])
    _noise = _rng.normal(scale=0.5, size=_n)
    _y = X_sim @ _b_true + _noise

    _b = ols(X_sim, _y)
    _e = residuals(X_sim, _y, _b)
    _b_answer_key = np.linalg.lstsq(X_sim, _y, rcond=None)[0]

    print("true b :", _b_true)
    print("ols b  :", np.round(_b, 3), "← close, not exact, because of the noise")
    print("X'e    :", np.round(X_sim.T @ _e, 10), "← misses perpendicular to every column")
    print("matches lstsq answer key:", np.allclose(_b, _b_answer_key))
    return (X_sim,)


@app.cell
def _(add_intercept, np, ols):
    # More data → noise averages out → estimates closer to the truth [0.5, 2.0, -1.0]
    _rng = np.random.default_rng(42)
    _b_true = np.array([0.5, 2.0, -1.0])
    for _n in [50, 500, 5000, 50000]:
        _X = add_intercept(np.column_stack([_rng.normal(size=_n), _rng.normal(size=_n)]))
        _y = _X @ _b_true + _rng.normal(scale=0.5, size=_n)
        _b = ols(_X, _y)
        _worst_gap = np.abs(_b - _b_true).max()
        print(f"n = {_n:>6}: ols b = {np.round(_b, 3)}   biggest gap from truth = {_worst_gap:.4f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Check 4: many industries at once

    In the project, $X$ (the same days of Mkt-RF and vol shock) is shared by all 49 industries. Only $y$ changes. So $y$ becomes a matrix with **one column per industry**, and `ols` returns a matrix with **one column of coefficients per industry**. Row 0 holds the intercepts, row 1 the slopes on $x_1$ (think β), and row 2 the slopes on $x_2$ (think γ).

    No new code is needed. $X^\top y$ just gains one column per industry, and `solve` handles every column the same way. The check confirms that one big call matches fitting an industry on its own.
    """)
    return


@app.cell
def _(X_sim, np, ols):
    _rng = np.random.default_rng(1)
    _B_true = np.array([[0.1, 0.0, -0.2],    # row 0: intercepts for industries A, B, C
                        [1.2, 0.8, 1.0],     # row 1: slopes on x1 (think beta)
                        [-0.5, 0.3, 0.0]])   # row 2: slopes on x2 (think gamma)
    _Y = X_sim @ _B_true + _rng.normal(scale=0.5, size=(X_sim.shape[0], 3))

    _B = ols(X_sim, _Y)
    print("true B (one column per industry):\n", _B_true)
    print("ols B:\n", np.round(_B, 2))
    print("column 0 matches fitting industry A alone:", np.allclose(_B[:, 0], ols(X_sim, _Y[:, 0])))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Where this is used in the project

    The functions above live in `src/ols.py` with the same names and inputs:

    - `add_intercept(x)`: puts the column of 1s in front of the regressors.
    - `solve(A, c)`: Gaussian elimination with partial pivoting. Raises an error if the columns of X are linearly dependent (no unique answer).
    - `ols(X, y)`: solves X'X b = X'y. `y` can be one series, shape (days,), or all industries at once, shape (days, 49); `b` has one row per column of X.
    - `residuals(X, y, b)`: the misses, actual minus predicted.

    **Where they're called:**

    - **Vol shock** (`factors.vol_shock`): `ols` of ΔVIX on [1, Mkt-RF]; the residuals are the vol shock.
    - **Industry sensitivities** (`estimation.fit_industries`): `ols` of all 49 industries on [1, Mkt-RF, vol shock] in one call, giving α, β and γ.
    - **Bootstrap and Monte Carlo:** the same `ols`, run thousands of times on resampled or simulated data.
    """)
    return


if __name__ == "__main__":
    app.run()
