import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from regression import ols_simple, transpose

    return ols_simple, transpose


@app.cell
def _(ols_simple):
    # sample
    x = [1, 2, 3, 4, 5]
    y = [2.1, 3.9, 6.2, 7.8, 10.1]
    print(ols_simple(x, y))  # ≈ (0.05, 1.99)
    return


@app.cell
def _(transpose):
    X = [[1, 1, 1],
         [2, 2, 2],
         [3, 3, 3]]

    trans_X = transpose(X)

    trans_X

    X = transpose(trans_X)
    X

    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
