import numpy as np
import matplotlib.pyplot as plt


def get_simple_linear_regression_coefficients(x, y):
    x = np.array(x)
    y = np.array(y)
    assert len(x.shape) == 1
    assert len(y.shape) == 1
    m = len(x)
    assert len(y) == m
    
    assert m >= 1

    if m == 1:
        return y[0], 0.0
    
    mx = np.mean(x)
    my = np.mean(y)
    sxx = np.sum((x - mx)**2)
    sxy = np.sum((x - mx) * (y - my))

    beta = sxy / sxx
    alpha = my - beta * mx
    return alpha, beta


def get_min_sum_squared_residuals(x, y):
    # x = np.array(x)
    # y = np.array(y)
    assert len(x.shape) == 1
    assert len(y.shape) == 1
    m = len(x)
    assert len(y) == m
    
    assert m >= 1

    if m == 1:
        return 0

    mx = np.mean(x)
    my = np.mean(y)
    sxx = np.sum((x - mx)**2)
    syy = np.sum((y - my)**2)
    sxy = np.sum((x - mx) * (y - my))
    return syy - sxy**2 / sxx


def get_loss(x, y):
    x = np.array(x)
    y = np.array(y)
    assert len(x.shape) == 1
    assert len(y.shape) == 1
    m = len(x)
    assert len(y) == m
    
    assert m >= 1

    if m == 1:
        return 0

    mx = np.mean(x)
    my = np.mean(y)
    sxx = np.sum((x - mx)**2)
    sxy = np.sum((x - mx) * (y - my))

    return -sxy**2 / sxx - sxx + 2 * sxy - m * (mx - my)**2


def get_indices_for_paint(x, y):
    x = np.array(x)
    y = np.array(y)
    assert len(x.shape) == 1
    assert len(y.shape) == 1
    m = len(x)
    assert len(y) == m

    min_L = np.inf
    best_indices = None

    for size in range(1, m + 1):
        for min_i in range(m - size + 1):
            x_in = x[min_i : min_i + size]
            y_in = y[min_i : min_i + size]

            # x_out = np.concatenate((x[:min_i], x[min_i + size:]))
            # y_out = np.concatenate((y[:min_i], y[min_i + size:]))
            # L_in = get_min_sum_squared_residuals(x_in, y_in)
            # L_out = np.sum((x_out - y_out)**2)
            # L = L_in + L_out

            L = get_loss(x_in, y_in)

            if L < min_L:
                best_indices = np.arange(min_i, min_i + size)
                min_L = L

    return best_indices, min_L

def main():
    # x = np.array([1.0, 2, 3, 7, 12])
    # y = np.array([4.0, 5, 10, 3, 13])

    # print(x)

    # for _ in range(3):

    #     best_indices, L = get_indices_for_paint(x, y)

    #     alpha, beta = get_simple_linear_regression_coefficients(x[best_indices], y[best_indices])
    #     x[best_indices] = alpha + beta * x[best_indices]

    #     print(best_indices, L)
    #     print(x)


    # best_indices, L = get_indices_for_paint(x, y)
    # print(best_indices)

    # print(get_loss(x[:1], y[:1]))
    # print(get_loss(x[1:], y[1:]))


    m = 100
    x = np.random.randn(m)
    y = np.random.randn(m)

    for _ in range(10):
        best_indices, L = get_indices_for_paint(x, y)
        print(best_indices)
        print(np.sum((x - y)**2), L)

        alpha, beta = get_simple_linear_regression_coefficients(x[best_indices], y[best_indices])
        x[best_indices] = alpha + beta * x[best_indices]

        # print(x)

    losses = np.zeros((m, m))
    for size in range(1, m + 1):
        for min_i in range(m - size + 1):
            x_in = x[min_i : min_i + size]
            y_in = y[min_i : min_i + size]

            L = get_loss(x_in, y_in)

            losses[min_i, min_i + size - 1] = L
    plt.matshow(losses)
    plt.show()

    plt.scatter(x, y)
    plt.show()

    # print(get_loss(x[:100], y[:100]))
    # print(get_loss(x[100:], y[100:]))

    print(get_loss(x[:50], y[:50]) + get_loss(x[50:], y[50:]))
    print(get_loss(x, y))



if __name__ == '__main__':
    main()




'''

-sxy**2 / sxx - sxx + 2 * sxy - m * (mx - my)**2

sxy**2 / sxx - 2 * sxy + sxx + m * (mx - my)**2


sxy**2 / sxx - 2 * sxy + sxx + m * (mx - my)**2


'''
