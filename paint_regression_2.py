import numpy as np
from PIL import Image
from functools import partial
import cProfile, pstats, io
from time import time
from tqdm import tqdm, trange
import os


def get_max_beta_for_zero_sxy_and_sxx(mx, my, n):
    max_beta = 1.0

    for j in range(n):
        for a, b in [(mx[j], my[j]), (1-mx[j], 1-my[j])]:
            if a != 0:
                if b == 0:
                    return 0.0
                else:
                    m = b / a
                    if m < max_beta:
                        max_beta = m

    return max_beta


def get_least_squares_paint_regression_coefs(x, y):
    '''Returns the least squares regression estimates for the model
    y = alpha + beta * x, where x, y, and alpha are n-dimensional vectors
    and beta is a scalar.
    
    Parameters:
        x: array of shape (m, n): list of example x
        y: array of shape (m, n): list of example y
    Returns: a tuple (alpha, beta)'''

    x = np.array(x)
    y = np.array(y)
    assert len(x.shape) == len(y.shape) == 2
    assert x.shape == y.shape
    m, n = x.shape
    
    assert m >= 1

    if m == 1:
        return y[0], 0.0
    
    mx = x.mean(axis=0)
    my = y.mean(axis=0)

    sxy = np.mean(x * y) - np.mean(mx * my)
    sxx = np.mean(x**2) - np.mean(mx**2)

    if np.isclose(sxy, 0, rtol = 0.0, atol=1e-8):
        sxy = np.mean((x - mx) * (y - my))

    # beta * sxx == sxy

    if np.isclose(sxy, 0, rtol = 0.0, atol=1e-8):
        if sxx == 0:
            # sxy == 0, sxx == 0
            # beta can be anything
            max_beta = get_max_beta_for_zero_sxy_and_sxx(mx, my, n)
            beta = max_beta / 2
        else:
            # sxy == 0, sxx != 0
            # beta must be 0
            beta = 0.0
    elif sxx == 0:
        # sxy != 0, sxx == 0
        raise RuntimeError(f"no solution to regression {sxx} {sxy}")
    else:
        # sxy != 0, sxx != 0
        beta = sxy / sxx

    alpha = my - beta * mx

    return alpha, beta


def test():
    # x = np.array([
    #     [0.3, 0.5, 0.7],
    #     [0.6, 0.2, 0.1]
    # ])

    x = np.array([
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0]
    ])

    y = np.array([
        [0.7, 0.4, 0.2],
        [0.3, 0.9, 0.5]
    ])

    # y = np.array([
    #     [0.7, 0.8, 0.9],
    #     [0.8, 0.5, 0.3]
    # ])

    # y = 0.6 * np.array([0.5, 0.7, 0.2]) + 0.4 * x

    alpha, beta = get_least_squares_paint_regression_coefs(x, y)

    print(alpha, beta)

    o = 1 - beta
    c = alpha / o

    print(c, o)

    
    im = Image.open('image-52.jpg')

    arr = np.asarray(im) / 255
    print(arr.shape)

    # arr_reshaped = arr.reshape(-1, 3)
    # print(arr_reshaped.shape)

    arr2 = alpha + beta * arr

    im2_arr = (arr2 * 255).astype(np.uint8)
    im2 = Image.fromarray(im2_arr)
    im2.save('new_im.png')


def get_circle_brush_boundary(row0, col0, brush_size, m, n):
    '''Returns the boundary of a circular brush, in the form of a tuple
    (min_row, max_row, row_specs)
    where row_specs is a list and row_specs[i] is of the form (min_col, max_col)
    where row_specs[i] corresponds to the i'th row of the brush and corresponds
    to row min_row + i.
    
    Parameters:
        row0: row of center of brush
        col0: column of center of brush
        brush_size: radius of brush
        m: number of rows in image
        n: number of columns in image'''
    min_row = max(row0 - brush_size, 0)
    max_row = min(row0 + brush_size, m - 1)

    row_specs = []
    for row in range(min_row, max_row + 1):
        this_row_brush_radius = np.sqrt(brush_size**2 - (row - row0)**2)
        min_col = int(round(col0 - this_row_brush_radius))
        min_col = max(0, min_col)
        max_col = int(round(col0 + this_row_brush_radius))
        max_col = min(n - 1, max_col)

        row_specs.append((min_col, max_col))

    return min_row, max_row, row_specs


def evaluate_loss_circle_brush(x, arr, painting, curr_loss=None, enforce_brush_regression=False):
    row0, col0, brush_size = x

    m, n, c = arr.shape

    min_row, max_row, row_specs = get_circle_brush_boundary(row0, col0, brush_size, m, n)
    rows = range(min_row, max_row + 1)

    painting_brush_rows = []
    arr_brush_rows = []
    for row, (min_col, max_col) in zip(rows, row_specs):
        painting_brush_rows.append(painting[row, min_col : max_col + 1])
        arr_brush_rows.append(arr[row, min_col : max_col + 1])

    painting_brush_pixels = np.concatenate(painting_brush_rows)
    arr_brush_pixels = np.concatenate(arr_brush_rows)

    alpha, beta = get_least_squares_paint_regression_coefs(painting_brush_pixels, arr_brush_pixels)

    if enforce_brush_regression:
        o = 1 - beta
        if not (0 < o <= 1):
            return np.inf, (alpha, beta)
        c = alpha / o
        if not np.all(np.logical_and(0 <= c, c <= 1)):
            return np.inf, (alpha, beta)

    if curr_loss is None:
        curr_loss = np.sum((painting - arr)**2)
    
    curr_loss -= np.sum((painting_brush_pixels - arr_brush_pixels)**2)
    new_painting_brush_pixels = alpha + beta * painting_brush_pixels
    curr_loss += np.sum((new_painting_brush_pixels - arr_brush_pixels)**2)

    return curr_loss, (alpha, beta)


def apply_circle_brush(painting, row, col, brush_size, alpha, beta):
    m, n, c = painting.shape
    min_row, max_row, row_specs = get_circle_brush_boundary(row, col, brush_size, m, n)
    rows = range(min_row, max_row + 1)

    for row, (min_col, max_col) in zip(rows, row_specs):
        painting[row, min_col : max_col + 1] = alpha + beta * painting[row, min_col : max_col + 1]


def random_search(loss_func, random_candidate_func, n_samples, init_best_loss=np.inf):
    best_loss = init_best_loss
    best_x = None
    best_info = None
    for j in range(n_samples):
        x = random_candidate_func()
        loss, info = loss_func(x)
        if loss < best_loss:
            best_loss = loss
            best_x = x
            best_info = info
    
    return best_x, best_loss, best_info


def random_restart_stochastic_hill_climbing(loss_func, random_candidate_func, neighbor_func, n_samples, n_opt_iter, n_neighbors, stop_if_no_improvement, init_best_loss=np.inf):
    best_loss = init_best_loss
    best_x = None
    best_info = None

    for j in range(n_samples):
        best_x_iter = random_candidate_func()
        best_loss_iter, best_info_iter = loss_func(best_x_iter)
        for k in range(n_opt_iter):
            best_neighbor = neighbor_func(best_x_iter)
            best_neighbor_loss, best_neighbor_info = loss_func(best_neighbor)
            for _ in range(n_neighbors - 1):
                neighbor = neighbor_func(best_x_iter)
                neighbor_loss, neighbor_info = loss_func(neighbor)
                if neighbor_loss < best_neighbor_loss:
                    best_neighbor = neighbor
                    best_neighbor_loss, best_neighbor_info = neighbor_loss, neighbor_info

            if best_neighbor_loss < best_loss_iter:
                best_loss_iter = best_neighbor_loss
                best_x_iter = best_neighbor
                best_info_iter = best_neighbor_info
            elif stop_if_no_improvement:
                # print(k)
                break

        if best_loss_iter < best_loss:
            best_loss = best_loss_iter
            best_x = best_x_iter
            best_info = best_info_iter

    return best_x, best_loss, best_info


def make_painting(arr, n_iter, min_brush_size, max_brush_size, method='random search', save_every=None, folder_name=None, **opt_kwargs):
    '''Creates a painting of the image arr.

    Parameters:
        arr: numpy array of shape (M, N, C)
            where (M, N) are the dimensions of the image
            and C is the number of channels.
            All values in arr should be between 0 and 1.
        n_iter: the number of iterations to run to make the painting
        n_iter_opt: number of iterations to run the optimization for each stroke
            of the painting
        min_brush_size: minimum brush size allowed
        max_brush_size: maximum brush size allowed
    Returns: an image of the same shape as arr, which is a painting of arr'''

    assert len(arr.shape) == 3
    m, n, c = arr.shape

    painting = np.ones(arr.shape)

    def generate_random_candidate():
        row = np.random.randint(0, m)
        col = np.random.randint(0, n)
        # brush_size = np.random.randint(min_brush_size, max_brush_size+1)
        log_brush_size = np.random.uniform(np.log(min_brush_size), np.log(max_brush_size))
        brush_size = int(round(np.exp(log_brush_size)))
        return row, col, brush_size
    
    if method == 'random search':
        assert 'n_samples' in opt_kwargs
    elif method == 'hill climbing':
        n_samples = opt_kwargs['n_samples']
        n_opt_iter = opt_kwargs['n_opt_iter']
        n_neighbors = opt_kwargs['n_neighbors']
        stop_if_no_improvement = opt_kwargs['stop_if_no_improvement']
        brush_position_delta = opt_kwargs['brush_position_delta']
        brush_size_change_factor = opt_kwargs['brush_size_change_factor']

        def get_neighbor(x):
            r, c, bs = x

            r_new = r + np.random.randint(-brush_position_delta, brush_position_delta + 1)
            while not (0 <= r_new < m):
                r_new = r + np.random.randint(-brush_position_delta, brush_position_delta + 1)

            c_new = c + np.random.randint(-brush_position_delta, brush_position_delta + 1)
            while not (0 <= c_new < n):
                c_new = c + np.random.randint(-brush_position_delta, brush_position_delta + 1)
            
            log_min_bs, log_max_bs = np.log(bs / brush_size_change_factor), np.log(bs * brush_size_change_factor)
            
            bs_new = int(round(np.exp(np.random.uniform(log_min_bs, log_max_bs))))
            while not (min_brush_size <= bs_new <= max_brush_size):
                bs_new = int(round(np.exp(np.random.uniform(log_min_bs, log_max_bs))))

            return r_new, c_new, bs_new

    curr_loss = np.sum((painting - arr)**2)

    print(f"Initial loss: {curr_loss}")

    if save_every is not None:
        assert folder_name is not None
        if not os.path.isdir(folder_name):
            os.mkdir(folder_name)


    pbar = tqdm(total=n_iter)

    for i in range(n_iter):
        loss_func = partial(evaluate_loss_circle_brush, arr=arr, painting=painting, curr_loss=curr_loss, enforce_brush_regression=True)

        if method == 'random search':
            best_candidate, best_loss, best_params = random_search(
                loss_func, generate_random_candidate, **opt_kwargs, init_best_loss=curr_loss)
        elif method == 'hill climbing':
            best_candidate, best_loss, best_params = random_restart_stochastic_hill_climbing(
                loss_func, generate_random_candidate, get_neighbor, n_samples=n_samples, n_opt_iter=n_opt_iter, n_neighbors=n_neighbors, stop_if_no_improvement=stop_if_no_improvement, init_best_loss=curr_loss)
        else:
            raise ValueError('parameter "method" must be either "random search" or "hill climbing"')

        if best_candidate is not None:
            curr_loss = best_loss
            best_row, best_col, best_brush_size = best_candidate
            best_alpha, best_beta = best_params
            apply_circle_brush(painting, best_row, best_col, best_brush_size, best_alpha, best_beta)
            desc = f"Iteration {i+1}: Loss {curr_loss:.5g}, row {best_row}, col {best_col}, brush size {best_brush_size}"
        else:
            desc = f"Iteration {i+1}: Loss {curr_loss:.5g}, could not improve"

        if (i+1) % save_every == 0:
            fname = os.path.join(folder_name, f'{i+1}.png')
            im_arr = (painting * 255).astype(np.uint8)
            im = Image.fromarray(im_arr)
            im.save(fname)
        
        pbar.set_description(desc)
        pbar.update()

    pbar.close()
    
    return painting, curr_loss


if __name__ == '__main__':
    im = Image.open('image-52.jpg')
    arr = np.asarray(im) / 255

    # params = {
    #     'n_iter': 100_000,
    #     'min_brush_size': 1,
    #     'max_brush_size': 150,
    #     'method': 'random search',
    #     'n_samples': 20
    # }

    params = {
        'n_iter': 1000,
        'min_brush_size': 1,
        'max_brush_size': 30,
        'method': 'hill climbing',
        'n_samples': 5,
        'n_opt_iter': 10,
        'n_neighbors': 3,
        'stop_if_no_improvement': True,
        'brush_position_delta': 60,
        'brush_size_change_factor': 1.3
    }

    # pr = cProfile.Profile()
    # pr.enable()

    painting_info_string = '_'.join(f'{k}={v}' for k, v in params.items())

    t0 = time()
    painting, loss = make_painting(arr, save_every=50, folder_name=painting_info_string, **params)
    dt = time() - t0

    print(f"Time taken: {dt:.6f}s")

    # pr.disable()
    # s = io.StringIO()
    # ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    # ps.print_stats(20)
    # print(s.getvalue())


    im2_arr = (painting * 255).astype(np.uint8)
    im2 = Image.fromarray(im2_arr)

    fname = 'painting_' + painting_info_string + f'_loss={loss:.1f}.png'
    im2.save(fname)
