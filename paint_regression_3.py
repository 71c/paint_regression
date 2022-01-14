import numpy as np
from PIL import Image
from functools import partial
import cProfile, pstats, io
from time import time
from tqdm import tqdm, trange
import os
import datetime
import json


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


class CircleBrush:
    def __init__(self, row, col, radius):
        self._row0 = row
        self._col0 = col
        self._radius = radius
    
    def get_boundary_rows(self, m, n):
        '''Returns the boundary of the circular brush, in the form of a tuple
        (min_row, max_row, row_specs)
        where row_specs is a list and row_specs[i] is of the form (min_col, max_col)
        where row_specs[i] corresponds to the i'th row of the brush and corresponds
        to row min_row + i.
        
        Parameters:
            m: number of rows in image
            n: number of columns in image'''
        row0, col0, radius = self._row0, self._col0, self._radius
        min_row = max(row0 - radius, 0)
        max_row = min(row0 + radius, m - 1)

        row_specs = []
        for row in range(min_row, max_row + 1):
            row_diff = (row - row0) / radius * (radius - 0.5)

            this_row_brush_radius = np.sqrt(radius**2 - row_diff**2)

            min_col = int(round(col0 - this_row_brush_radius))
            min_col = max(0, min_col)
            max_col = int(round(col0 + this_row_brush_radius))
            max_col = min(n - 1, max_col)

            row_specs.append((min_col, max_col))

        return min_row, max_row, row_specs
    
    @classmethod
    def generate_random_brush(cls, m, n, min_radius, max_radius):
        row = np.random.randint(0, m)
        col = np.random.randint(0, n)
        # brush_size = np.random.randint(min_radius, max_radius+1)
        log_brush_size = np.random.uniform(np.log(min_radius), np.log(max_radius))
        brush_size = int(round(np.exp(log_brush_size)))
        return cls(row, col, brush_size)
    
    @classmethod
    def get_neighbor(cls, x, m, n, brush_position_delta, radius_change_factor, min_radius, max_radius):
        r, c, bs = x._row0, x._col0, x._radius

        r_new = r + np.random.randint(-brush_position_delta, brush_position_delta + 1)
        while not (0 <= r_new < m):
            r_new = r + np.random.randint(-brush_position_delta, brush_position_delta + 1)

        c_new = c + np.random.randint(-brush_position_delta, brush_position_delta + 1)
        while not (0 <= c_new < n):
            c_new = c + np.random.randint(-brush_position_delta, brush_position_delta + 1)
        
        log_min_bs, log_max_bs = np.log(bs / radius_change_factor), np.log(bs * radius_change_factor)
        
        bs_new = int(round(np.exp(np.random.uniform(log_min_bs, log_max_bs))))
        while not (min_radius <= bs_new <= max_radius):
            bs_new = int(round(np.exp(np.random.uniform(log_min_bs, log_max_bs))))

        return cls(r_new, c_new, bs_new)
    
    def __str__(self):
        return f"CircleBrush(row={self._row0}, col={self._col0}, radius={self._radius})"


class RectangleBrush:
    def __init__(self, row, col, width, length, angle):
        assert -np.pi/2 <= angle <= np.pi/2
        self._row0 = row
        self._col0 = col
        self._width = width
        self._length = length
        self._angle = angle
    
    def get_boundary_rows(self, m, n):
        row0, col0, width, length, theta = self._row0, self._col0, self._width, self._length, self._angle

        assert width >= 1
        assert length >= 1

        if theta == np.pi/2 or theta == -np.pi/2:
            width, length = length, width
            theta = 0
        
        if theta == 0:
            min_row = int(max(row0 - width/2, 0))
            max_row = int(min(row0 + width/2 - 1, m-1))
            min_col = int(max(col0 - length/2, 0))
            max_col = int(min(col0 + length/2 - 1, n-1))
            row_specs = [(min_col, max_col) for row in range(min_row, max_row + 1)]
            return min_row, max_row, row_specs
        
        if width == 1:
            if length == 1:
                return row0, row0, [(col0, col0)]
            
            if length <= 5 and abs(theta) <= 0.3:
                # theta is very small, effectively 0
                left_col = int(round(col0 - length/2))
                right_col = int(round(left_col + length - 1))
                left_col = max(left_col, 0)
                right_col = min(right_col, n-1)
                return row0, row0, [(left_col, right_col)]

            b = length / 2
            cos, sin = np.cos(theta), np.sin(theta)
            
            x2, y2 = b * cos, b * sin
            x1, y1 = -x2, -y2

            min_row = int(round(row0 + min(y1, y2)))
            max_row = int(round(row0 + max(y1, y2))) - 1
            if max_row == min_row - 1:
                max_row = min_row
            min_row = max(min_row, 0)
            max_row = min(max_row, m - 1)

            row_specs = []
            for row in range(min_row, max_row + 1):
                y = row - row0
                x = (x2 - x1) / (y2 - y1) * (y - y1) + x1
                col = int(round(x)) + col0
                if 0 <= col < n-1:
                    row_specs.append((col, col))

            return min_row, max_row, row_specs

        if theta < 0:
            width, length = length, width
            theta = np.pi/2 - (-theta)
        
        assert 0 < theta < np.pi/2

        a, b = width/2, length/2
        cos, sin = np.cos(theta), np.sin(theta)

        x1 = -b * cos - a * sin
        y1 = -b * sin + a * cos
        x2 = b * cos - a * sin
        y2 = b * sin + a * cos
        x3 = -x2
        y3 = -y2

        def L(y):
            if y1 <= y <= y2:
                return (x2 - x1) / (y2 - y1) * (y - y1) + x1
            return (x3 - x1) / (y3 - y1) * (y - y1) + x1

        min_row = max(int(np.ceil(row0 + y3)), 0)
        max_row = min(int(np.floor(row0 + y2)) - 1, m-1)

        row_specs = []
        for row in range(min_row, max_row + 1):
            y = row - row0 + 0.001
            x_left = int(round(L(y)))
            min_col = max(col0 + x_left, 0)
            x_right = int(round(-L(-y)))
            max_col = min(col0 + x_right - 1, n-1)
            if max_col >= 0:
                if min_col <= max_col:
                    row_specs.append((min_col, max_col))

        return min_row, max_row, row_specs

    @classmethod
    def generate_random_brush(cls, m, n, min_width, max_width, max_length):
        assert max_width <= max_length
        row = np.random.randint(0, m)
        col = np.random.randint(0, n)
        log_width = np.random.uniform(np.log(min_width), np.log(max_width))
        width = round(np.exp(log_width))
        log_length = np.random.uniform(np.log(width), np.log(max_length))
        length = round(np.exp(log_length))
        angle = np.random.uniform(-np.pi/2, np.pi/2)
        return cls(row, col, width, length, angle)
    
    @classmethod
    def get_neighbor(cls, x, m, n, brush_position_delta, angle_delta, width_change_factor, length_change_factor, min_width, max_width, max_length):
        row0, col0, width, length, angle = x._row0, x._col0, x._width, x._length, x._angle

        r_new = row0 + np.random.randint(-brush_position_delta, brush_position_delta + 1)
        while not (0 <= r_new < m):
            r_new = row0 + np.random.randint(-brush_position_delta, brush_position_delta + 1)

        c_new = col0 + np.random.randint(-brush_position_delta, brush_position_delta + 1)
        while not (0 <= c_new < n):
            c_new = col0 + np.random.randint(-brush_position_delta, brush_position_delta + 1)

        log_min_width, log_max_width = np.log(width / width_change_factor), np.log(width * width_change_factor)
        width_new = int(round(np.exp(np.random.uniform(log_min_width, log_max_width))))
        while not (min_width <= width_new <= max_width):
            width_new = int(round(np.exp(np.random.uniform(log_min_width, log_max_width))))
        
        log_min_length, log_max_length = np.log(length / length_change_factor), np.log(length * length_change_factor)
        length_new = int(round(np.exp(np.random.uniform(log_min_length, log_max_length))))
        while not (width_new <= length_new <= max_length):
            length_new = int(round(np.exp(np.random.uniform(log_min_length, log_max_length))))

        angle_new = angle + np.random.uniform(-angle_delta, angle_delta)
        angle_new = (angle_new - np.pi/2) % np.pi - np.pi/2

        return cls(r_new, c_new, width_new, length_new, angle_new)

    def __str__(self):
        return f"RectangleBrush(row={self._row0}, col={self._col0}, width={self._width}, length={self._length}, angle={self._angle:.4f})"


class CircleOrRectangleBrush:
    def __init__(self, brush):
        assert type(brush) in (CircleBrush, RectangleBrush)
        self._brush = brush
    
    def get_boundary_rows(self, m, n):
        return self._brush.get_boundary_rows(m, n)

    @classmethod
    def generate_random_brush(cls, m, n, min_width, max_width, max_length, min_radius, max_radius):
        if np.random.random() < 0.5:
            brush = CircleBrush.generate_random_brush(m, n, min_radius, max_radius)
        else:
            brush = RectangleBrush.generate_random_brush(m, n, min_width, max_width, max_length)
        return cls(brush)
    
    @classmethod
    def get_neighbor(cls, x, m, n, **kwargs):
        if type(x._brush) == CircleBrush:
            brush = CircleBrush.get_neighbor(
                x._brush, m, n, brush_position_delta=kwargs['brush_position_delta'], radius_change_factor=kwargs['radius_change_factor'], min_radius=kwargs['min_radius'], max_radius=kwargs['max_radius'])
        elif type(x._brush) == RectangleBrush:
            brush = RectangleBrush.get_neighbor(
                x._brush, m, n, brush_position_delta=kwargs['brush_position_delta'], angle_delta=kwargs['angle_delta'], width_change_factor=kwargs['width_change_factor'], length_change_factor=kwargs['length_change_factor'], min_width=kwargs['min_width'], max_width=kwargs['max_width'], max_length=kwargs['max_length'])
        return cls(brush)
        # return cls(type(x._brush).get_neighbor(x._brush, m, n, **kwargs))
    
    def __str__(self) -> str:
        return f"CircleOrRectangleBrush({self._brush})"


def evaluate_brush_loss(brush, arr, painting, curr_loss=None, enforce_brush_regression=False):
    m, n, c = arr.shape


    min_row, max_row, row_specs = brush.get_boundary_rows(m, n)
    rows = range(min_row, max_row + 1)

    painting_brush_rows = []
    arr_brush_rows = []
    
    for row, (min_col, max_col) in zip(rows, row_specs):

        if not (0 <= min_col < n):
            print(row_specs)
            print(brush)
            raise RuntimeError(f"min col is not in range: {min_col}, {max_col}")
        
        if not (0 <= max_col < n):
            print(row_specs)
            print(brush)
            raise RuntimeError(f"max col is not in range: {min_col}, {max_col}")

        if not (min_col <= max_col):
            print(row_specs)
            print(brush)
            raise RuntimeError(f"min col is not less than or equal to max col: {min_col}, {max_col}")

        try:
            painting_brush_rows.append(painting[row, min_col : max_col + 1])
        except IndexError as e:
            print(brush)
            print(row_specs)
            raise e
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


def apply_brush(painting, brush, alpha, beta):
    m, n, c = painting.shape
    min_row, max_row, row_specs = brush.get_boundary_rows(m, n)
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


def random_restart_stochastic_hill_climbing(loss_func, random_candidate_func, neighbor_func, n_samples, best_of_per_restart, n_opt_iter, n_neighbors, stop_if_no_improvement, init_best_loss=np.inf):
    best_loss = init_best_loss
    best_x = None
    best_info = None

    for j in range(n_samples):
        if best_of_per_restart == 1:
            best_x_iter = random_candidate_func()
            best_loss_iter, best_info_iter = loss_func(best_x_iter)
        else:
            best_x_iter, best_loss_iter, best_info_iter = random_search(loss_func, random_candidate_func, n_samples=best_of_per_restart, init_best_loss=np.inf)
        
        # print(f'initial: {best_loss_iter}')
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
                # print(k, f'final: {best_loss_iter}')
                break

        if best_loss_iter < best_loss:
            best_loss = best_loss_iter
            best_x = best_x_iter
            best_info = best_info_iter

    return best_x, best_loss, best_info


def make_painting(arr, n_iter, method='random search', brush='circle', save_every=None, folder_name=None, **kwargs):
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

    if brush not in ('circle', 'rectangle', 'circle-rectangle', 'circle-rectangle-best'):
        raise ValueError("brush must be one of 'circle', 'rectangle', 'circle-rectangle', or 'circle-rectangle-best'")

    # get settings for generating random brushes
    if brush == 'circle' or brush == 'circle-rectangle' or brush == 'circle-rectangle-best':
        min_radius = kwargs['min_radius']
        max_radius = kwargs['max_radius']
        if brush == 'circle':
            generate_random_candidate = partial(CircleBrush.generate_random_brush, m=m, n=n, min_radius=min_radius, max_radius=max_radius)
        elif brush == 'circle-rectangle-best':
            generate_random_candidate_circle = partial(CircleBrush.generate_random_brush, m=m, n=n, min_radius=min_radius, max_radius=max_radius)
    if brush == 'rectangle' or brush == 'circle-rectangle' or brush == 'circle-rectangle-best':
        min_width = kwargs['min_width']
        max_width = kwargs['max_width']
        max_length = kwargs['max_length']
        if brush == 'rectangle':
            generate_random_candidate = partial(RectangleBrush.generate_random_brush, m=m, n=n, min_width=min_width, max_width=max_width, max_length=max_length)
        elif brush == 'circle-rectangle-best':
            generate_random_candidate_rectangle = partial(RectangleBrush.generate_random_brush, m=m, n=n, min_width=min_width, max_width=max_width, max_length=max_length)
    if brush == 'circle-rectangle':
        generate_random_candidate = partial(CircleOrRectangleBrush.generate_random_brush, m=m, n=n, min_width=min_width, max_width=max_width, max_length=max_length, min_radius=min_radius, max_radius=max_radius)

    # get settings for optimization
    if method == 'random search':
        n_samples = kwargs['n_samples']
    elif method == 'hill climbing':
        n_samples = kwargs['n_samples']
        best_of_per_restart = kwargs['best_of_per_restart']
        n_opt_iter = kwargs['n_opt_iter']
        n_neighbors = kwargs['n_neighbors']
        stop_if_no_improvement = kwargs['stop_if_no_improvement']
        if brush == 'circle' or brush == 'circle-rectangle' or brush == 'circle-rectangle-best':
            brush_position_delta = kwargs['brush_position_delta']
            radius_change_factor = kwargs['radius_change_factor']
            if brush == 'cirlce':
                get_neighbor = partial(CircleBrush.get_neighbor, m=m, n=n, brush_position_delta=brush_position_delta, radius_change_factor=radius_change_factor, min_radius=min_radius, max_radius=max_radius)
            elif brush == 'circle-rectangle-best':
                get_neighbor_circle = partial(CircleBrush.get_neighbor, m=m, n=n, brush_position_delta=brush_position_delta, radius_change_factor=radius_change_factor, min_radius=min_radius, max_radius=max_radius)
        if brush == 'rectangle' or brush == 'circle-rectangle' or brush == 'circle-rectangle-best':
            brush_position_delta = kwargs['brush_position_delta']
            angle_delta = kwargs['angle_delta']
            width_change_factor = kwargs['width_change_factor']
            length_change_factor = kwargs['length_change_factor']
            if brush == 'rectangle':
                get_neighbor = partial(RectangleBrush.get_neighbor, m=m, n=n, brush_position_delta=brush_position_delta, angle_delta=angle_delta, width_change_factor=width_change_factor, length_change_factor=length_change_factor, min_width=min_width, max_width=max_width, max_length=max_length)
            elif brush == 'circle-rectangle-best':
                get_neighbor_rectangle = partial(RectangleBrush.get_neighbor, m=m, n=n, brush_position_delta=brush_position_delta, angle_delta=angle_delta, width_change_factor=width_change_factor, length_change_factor=length_change_factor, min_width=min_width, max_width=max_width, max_length=max_length)
        if brush == 'circle-rectangle':
            get_neighbor = partial(CircleOrRectangleBrush.get_neighbor,
                m=m, n=n, brush_position_delta=brush_position_delta, radius_change_factor=radius_change_factor, min_radius=min_radius, max_radius=max_radius,
                angle_delta=angle_delta, width_change_factor=width_change_factor, length_change_factor=length_change_factor, min_width=min_width, max_width=max_width, max_length=max_length)
    else:
        raise ValueError("parameter 'method' must be one of 'random search' or 'hill climbing'")

    curr_loss = np.sum((painting - arr)**2)

    print(f"Initial loss: {curr_loss}")

    if save_every is not None:
        assert folder_name is not None
        if not os.path.isdir(folder_name):
            os.mkdir(folder_name)

    pbar = tqdm(total=n_iter)
    loss_lines = []

    for i in range(n_iter):
        loss_func = partial(evaluate_brush_loss, arr=arr, painting=painting, curr_loss=curr_loss, enforce_brush_regression=True)

        if method == 'random search':
            best_candidate, best_loss, best_params = random_search(
                loss_func, generate_random_candidate, n_samples=n_samples, init_best_loss=curr_loss)
        elif method == 'hill climbing':
            if brush == 'circle-rectangle-best':
                best_candidate_circle, best_loss_circle, best_params_circle = random_restart_stochastic_hill_climbing(
                    loss_func, generate_random_candidate_circle, get_neighbor_circle, n_samples=n_samples, best_of_per_restart=best_of_per_restart, n_opt_iter=n_opt_iter, n_neighbors=n_neighbors, stop_if_no_improvement=stop_if_no_improvement, init_best_loss=curr_loss)
                best_candidate_rectangle, best_loss_rectangle, best_params_rectangle = random_restart_stochastic_hill_climbing(
                    loss_func, generate_random_candidate_rectangle, get_neighbor_rectangle, n_samples=n_samples, best_of_per_restart=best_of_per_restart, n_opt_iter=n_opt_iter, n_neighbors=n_neighbors, stop_if_no_improvement=stop_if_no_improvement, init_best_loss=curr_loss)
                if best_loss_circle < best_loss_rectangle:
                    best_candidate, best_loss, best_params = best_candidate_circle, best_loss_circle, best_params_circle
                else:
                    best_candidate, best_loss, best_params = best_candidate_rectangle, best_loss_rectangle, best_params_rectangle
            else:
                best_candidate, best_loss, best_params = random_restart_stochastic_hill_climbing(
                    loss_func, generate_random_candidate, get_neighbor, n_samples=n_samples, best_of_per_restart=best_of_per_restart, n_opt_iter=n_opt_iter, n_neighbors=n_neighbors, stop_if_no_improvement=stop_if_no_improvement, init_best_loss=curr_loss)

        if best_candidate is not None:
            curr_loss = best_loss
            best_alpha, best_beta = best_params
            apply_brush(painting, best_candidate, best_alpha, best_beta)
            desc = f"Iteration {i+1}: Loss {curr_loss:.5g}, brush: {best_candidate}"
        else:
            desc = f"Iteration {i+1}: Loss {curr_loss:.5g}, could not improve"
        
        loss_lines.append(f"{i+1}\t{curr_loss:.2f}\n")

        if (i+1) % save_every == 0:
            fname = os.path.join(folder_name, f'{i+1}.png')
            im_arr = (painting * 255).astype(np.uint8)
            im = Image.fromarray(im_arr)
            im.save(fname)

            with open(os.path.join(folder_name, 'losses.txt'), 'w+') as f:
                f.writelines(loss_lines)
        
        pbar.set_description(desc)
        pbar.update()

    pbar.close()
    
    return painting, curr_loss


def brush_test():
    im = Image.open('image-52.jpg')
    arr = np.asarray(im) / 255

    # brush = CircleBrush(row=390, col=870, radius=40)
    # brush = RectangleBrush(row=390, col=870, width=1, length=120, angle=-np.pi/2)
    
    # brush = RectangleBrush(row=392, col=533, width=1.0, length=2.0, angle=1.5458750617574797)
    # brush = RectangleBrush(row=392, col=533, width=1, length=2, angle=np.pi/2)
    brush = RectangleBrush(row=972, col=530, width=1.0, length=3.0, angle=0.1320)


    opacity = 0.7
    color = np.array([1.0, 0.0, 0.3])
    alpha, beta = opacity * color, 1 - opacity
    apply_brush(arr, brush, alpha, beta)

    im2_arr = (arr * 255).astype(np.uint8)
    im2 = Image.fromarray(im2_arr)
    im2.save('test.png')


def main():
    # image_name = 'image-52.jpg'
    image_name = 'Screenshot.png'

    im = Image.open(image_name)
    arr = np.asarray(im)
    assert len(arr.shape) == 3
    if arr.shape[2] == 4:
        arr = arr[:, :, :3] # remove alpha channel if there is one
    arr = arr / 255 # normalize to be between 0 and 1

    # params = {
    #     'n_iter': 100_000,
    #     'min_brush_size': 1,
    #     'max_brush_size': 150,
    #     'method': 'random search',
    #     'n_samples': 20
    # }

    # params = {
    #     'n_iter': 1000,
    #     'min_brush_size': 1,
    #     'max_brush_size': 30,
    #     'method': 'hill climbing',
    #     'n_samples': 5,
    #     'n_opt_iter': 10,
    #     'n_neighbors': 3,
    #     'stop_if_no_improvement': True,
    #     'brush_position_delta': 60,
    #     'radius_change_factor': 1.3
    # }


    # Random search rectangles
    # params = {
    #     'n_iter': 1000,

    #     'brush': 'rectangle',
    #     'min_width': 2,
    #     'max_width': 100,
    #     'max_length': 300,

    #     'method': 'random search',
    #     'n_samples': 300
    # }

    # hill climbing rectangles
    # params = {
    #     'n_iter': 1000,

    #     'brush': 'rectangle',
    #     'min_width': 2,
    #     'max_width': 100,
    #     'max_length': 300,

    #     'method': 'hill climbing',
    #     'n_samples': 50,
    #     'n_opt_iter': 10,
    #     'n_neighbors': 3,
    #     'stop_if_no_improvement': True,
    #     'brush_position_delta': 60,
    #     'angle_delta': np.pi/6,
    #     'width_change_factor': 1.3,
    #     'length_change_factor': 1.3
    # }

    # Random search + hillclimbing rectangles
    # params = {
    #     'n_iter': 1000,

    #     'brush': 'rectangle',
    #     'min_width': 2,
    #     'max_width': 100,
    #     'max_length': 300,

    #     'method': 'hybrid',
    #     'n_samples': 300,
    #     'best_of_per_restart': 1,
    #     'n_opt_iter': 20,
    #     'n_neighbors': 10,
    #     'stop_if_no_improvement': True,
    #     'brush_position_delta': 30,
    #     'angle_delta': np.pi/8,
    #     'width_change_factor': 1.1,
    #     'length_change_factor': 1.1
    # }

    # Random search + hillclimbing rectangles
    # params = {
    #     'n_iter': 1000,

    #     'brush': 'rectangle',
    #     'min_width': 2,
    #     'max_width': 100,
    #     'max_length': 300,

    #     'method': 'hill climbing',
    #     'n_samples': 1,
    #     'best_of_per_restart': 300,
    #     'n_opt_iter': 20,
    #     'n_neighbors': 10,
    #     'stop_if_no_improvement': True,
    #     'brush_position_delta': 30,
    #     'angle_delta': np.pi/8,
    #     'width_change_factor': 1.1,
    #     'length_change_factor': 1.1
    # }

    # rectanlge brush hill climbing with multiple rounds of random search. nice settings but slow
    # params = {
    #     'n_iter': 1000,

    #     'brush': 'rectangle',
    #     'min_width': 2,
    #     'max_width': 100,
    #     'max_length': 300,

    #     'method': 'hill climbing',
    #     'n_samples': 6,
    #     'best_of_per_restart': 50,
    #     'n_opt_iter': 20,
    #     'n_neighbors': 10,
    #     'stop_if_no_improvement': True,
    #     'brush_position_delta': 30,
    #     'angle_delta': np.pi/8,
    #     'width_change_factor': 1.1,
    #     'length_change_factor': 1.1
    # }

    # hill climbing - good params
    params = {
        'n_iter': 100_000,

        'brush': 'circle-rectangle-best',
        'min_width': 2,
        'max_width': 105,
        'max_length': 300,
        'min_radius': 1,
        'max_radius': 100,

        'method': 'hill climbing',
        'n_samples': 6,
        'best_of_per_restart': 50,
        'n_opt_iter': 20,
        'n_neighbors': 10,
        'stop_if_no_improvement': True,

        'brush_position_delta': 20,

        'radius_change_factor': 1.05,

        'angle_delta': np.pi/9,
        'width_change_factor': 1.05,
        'length_change_factor': 1.05
    }


    # brush_position_delta = kwargs['brush_position_delta']
    # angle_delta = kwargs['angle_delta']
    # width_change_factor = kwargs['width_change_factor']
    # length_change_factor = kwargs['length_change_factor']



    # pr = cProfile.Profile()
    # pr.enable()

    painting_info_string = '_'.join(
        f'{k}={v:.3f}' if type(v) is float else f'{k}={v}'
        for k, v in params.items())

    tm = datetime.datetime.now()
    folder_name = f'painting_{tm.year}-{tm.month}-{tm.day}T{tm.hour}_{tm.minute}_{tm.second}'

    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)
    
    with open(os.path.join(folder_name, 'params.json'), 'w+') as f:
        json.dump(params, f, indent=4)


    t0 = time()
    painting, loss = make_painting(arr, save_every=50, folder_name=folder_name, **params)
    dt = time() - t0

    print(f"Time taken: {dt:.6f}s")

    # pr.disable()
    # s = io.StringIO()
    # ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    # ps.print_stats(20)
    # print(s.getvalue())


    im2_arr = (painting * 255).astype(np.uint8)
    im2 = Image.fromarray(im2_arr)

    # fname = 'painting_' + painting_info_string + f'_loss={loss:.1f}.png'
    # im2.save(fname)



if __name__ == '__main__':
    main()
