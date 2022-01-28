import numpy as np
from PIL import Image
from functools import partial
import cProfile, pstats, io
from time import time
from tqdm import tqdm, trange
import os
import datetime
import json
from queue import PriorityQueue
from dataclasses import dataclass, field
from typing import Any
from copy import deepcopy
import random
from numba import jit


@jit(nopython=True)
def get_max_beta_for_zero_sxy_and_sxx(mx, my):
    n = len(mx)
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


@jit(nopython=True)
def get_least_squares_paint_regression_coefs_from_stats(mx, my, sxx, sxy):
    # Solves beta * sxx == sxy

    if sxx == 0:
        # Assume that sxy == 0
        max_beta = get_max_beta_for_zero_sxy_and_sxx(mx, my)
        beta = max_beta / 2
    else:
        beta = sxy / sxx

    alpha = my - beta * mx

    return alpha, beta


def get_least_squares_paint_regression_coefs(x, y):
    '''Returns the least squares regression estimates for the model
    y = alpha + beta * x, where x, y, and alpha are n-dimensional vectors
    and beta is a scalar.
    
    Parameters:
        x: array of shape (m, n): list of example x
        y: array of shape (m, n): list of example y
    Returns: a tuple (mx, my, sxx, sxy, alpha, beta)'''

    assert len(x.shape) == len(y.shape) == 2
    assert x.shape == y.shape
    m, n = x.shape
    
    assert m >= 1

    mx = x.mean(axis=0)
    my = y.mean(axis=0)

    sxy = np.sum(x * y) - m * np.sum(mx * my)
    sxx = np.sum(x**2) - m * np.sum(mx**2)

    if m == 1:
        alpha, beta = y[0], 0.0
    else:
        alpha, beta = get_least_squares_paint_regression_coefs_from_stats(mx, my, sxx, sxy)
    
    return mx, my, sxx, sxy, alpha, beta


class Brush:
    def intersects(self, other):
        x1, x2, y1, y2 = self.get_min_and_max_coords()
        x1_O, x2_O, y1_O, y2_O = other.get_min_and_max_coords()

        return not (x2 < x1_O or x2_O < x1 or y2 < y1_O or y2_O < y1)


class CircleBrush(Brush):
    def __init__(self, row, col, radius):
        self._row0 = row
        self._col0 = col
        self._radius = radius
        self._n_pixels = None

    def get_boundary_rows(self, m, n):
        '''Returns the boundary of the circular brush, in the form of a tuple
        (min_row, max_row, row_specs)
        where row_specs is a list and row_specs[i] is of the form (min_col, max_col)
        where row_specs[i] corresponds to the i'th row of the brush and corresponds
        to row min_row + i.
        
        Parameters:
            m: number of rows in image
            n: number of columns in image'''
        min_row, max_row, row_specs = self._get_boundary_rows_helper(m, n)
        if self._n_pixels is None:
            n_pixels = 0
            for min_col, max_col in row_specs:
                n_pixels += max_col - min_col + 1
            self._n_pixels = n_pixels
        return min_row, max_row, row_specs

    def _get_boundary_rows_helper(self, m, n):
        row0, col0, radius = self._row0, self._col0, self._radius
        min_row = max(row0 - radius, 0)
        max_row = min(row0 + radius, m - 1)

        rows = np.arange(min_row, max_row + 1)
        row_diffs = (rows - row0) / radius * (radius - 0.5)
        brush_radii = np.sqrt(radius**2 - row_diffs**2)
        min_cols = np.fmax(0, np.around(col0 - brush_radii).astype(int))
        max_cols = np.fmin(n-1, np.around(col0 + brush_radii).astype(int))
        row_specs = [(min_cols[i], max_cols[i]) for i in range(max_row - min_row + 1)]

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
    
    def get_min_and_max_coords(self):
        min_x = self._col0 - self._radius
        max_x = self._col0 + self._radius
        min_y = self._row0 - self._radius
        max_y = self._row0 + self._radius
        return min_x, max_x, min_y, max_y
    
    def __str__(self):
        return f"CircleBrush(row={self._row0}, col={self._col0}, radius={self._radius})"


@jit(nopython=True)
def func(m, n, row0, col0, width, length, theta):
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
        true_min_row, true_max_row = 2*m, -2*m
        for row in range(min_row, max_row + 1):
            y = row - row0
            x = (x2 - x1) / (y2 - y1) * (y - y1) + x1
            col = int(round(x)) + col0
            if 0 <= col < n-1:
                row_specs.append((col, col))
                if row < true_min_row:
                    true_min_row = row
                if row > true_max_row:
                    true_max_row = row

        return true_min_row, true_max_row, row_specs

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

    min_row = max(int(np.ceil(row0 + y3)), 0)
    max_row = min(int(np.floor(row0 + y2)) - 1, m-1)

    row_specs = []
    true_min_row, true_max_row = 2*m, -2*m
    for row in range(min_row, max_row + 1):
        y = row - row0 + 0.001

        if y1 <= y <= y2:
            x_left = int(round((x2 - x1) / (y2 - y1) * (y - y1) + x1))
        else:
            x_left = int(round((x3 - x1) / (y3 - y1) * (y - y1) + x1))
        
        if y1 <= -y <= y2:
            x_right = int(round(-((x2 - x1) / (y2 - y1) * (-y - y1) + x1)))
        else:
            x_right = int(round(-((x3 - x1) / (y3 - y1) * (-y - y1) + x1)))

        min_col = max(col0 + x_left, 0)
        max_col = min(col0 + x_right - 1, n-1)
        if max_col >= 0:
            if min_col <= max_col:
                row_specs.append((min_col, max_col))
                if row < true_min_row:
                    true_min_row = row
                if row > true_max_row:
                    true_max_row = row

    return true_min_row, true_max_row, row_specs


class RectangleBrush(Brush):
    def __init__(self, row, col, width, length, angle):
        assert -np.pi/2 <= angle <= np.pi/2
        self._row0 = row
        self._col0 = col
        self._width = width
        self._length = length
        self._angle = angle
        self._n_pixels = None

    def get_boundary_rows(self, m, n):
        min_row, max_row, row_specs = self._get_boundary_rows_helper(m, n)
        if self._n_pixels is None:
            n_pixels = 0
            for min_col, max_col in row_specs:
                n_pixels += max_col - min_col + 1
            self._n_pixels = n_pixels
        return min_row, max_row, row_specs

    
    def _get_boundary_rows_helper(self, m, n):
        row0, col0, width, length, theta = self._row0, self._col0, self._width, self._length, self._angle
        return func(m, n, row0, col0, width, length, theta)

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

    def get_min_and_max_coords(self):
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
            min_x = min_col
            max_x = max_col
            min_y = min_row
            max_y = max_row
            return min_x, max_x, min_y, max_y
        
        if width == 1:
            if length == 1:
                return row0, row0, [(col0, col0)]
            
            if length <= 5 and abs(theta) <= 0.3:
                # theta is very small, effectively 0
                left_col = int(round(col0 - length/2))
                right_col = int(round(left_col + length - 1))

                min_x = left_col
                max_x = right_col
                min_y = row0
                max_y = row0
                return min_x, max_x, min_y, max_y

            b = length / 2
            cos, sin = np.cos(theta), np.sin(theta)
            
            x2, y2 = b * cos, b * sin
            x1, y1 = -x2, -y2

            min_row = int(round(row0 + min(y1, y2)))
            max_row = int(round(row0 + max(y1, y2))) - 1
            if max_row == min_row - 1:
                max_row = min_row
            
            min_col = int(round(col0 + min(x1, x2)))
            max_col = int(round(col0 + max(x1, x2)))

            min_x = min_col
            max_x = max_col
            min_y = min_row
            max_y = max_row
            return min_x, max_x, min_y, max_y

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
        y3 = -y2
        x4 = -x1

        min_row = int(np.ceil(row0 + y3))
        max_row = int(np.floor(row0 + y2)) - 1

        min_col = int(np.ceil(col0 + x1))
        max_col = int(np.floor(col0 + x4)) - 1

        min_x = min_col
        max_x = max_col
        min_y = min_row
        max_y = max_row
        return min_x, max_x, min_y, max_y

    def __str__(self):
        return f"RectangleBrush(row={self._row0}, col={self._col0}, width={self._width}, length={self._length}, angle={self._angle:.4f})"


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


def hill_climbing(loss_func, random_candidate_func, neighbor_func, n_samples, best_of_per_restart, n_opt_iter, n_neighbors, stop_if_no_improvement, init_best_loss=np.inf):
    best_loss = init_best_loss
    best_x = None
    best_info = None

    for j in range(n_samples):
        if best_of_per_restart == 1:
            best_x_iter = random_candidate_func()
            best_loss_iter, best_info_iter = loss_func(best_x_iter)
        else:
            # Occasionally, random search gets a loss of infinity for ALL of its
            # random samples which causes best_x_iter to be None.
            best_x_iter = None
            while best_x_iter is None:
                best_x_iter, best_loss_iter, best_info_iter = random_search(loss_func, random_candidate_func, n_samples=best_of_per_restart, init_best_loss=np.inf)

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
                break

        if best_loss_iter < best_loss:
            best_loss = best_loss_iter
            best_x = best_x_iter
            best_info = best_info_iter

    return best_x, best_loss, best_info


class Painter:
    def __init__(self, brushes, hillclimbing_params, reuse_samples_start_iter, src_image, init_painting, n_queue, max_n_pixels_regression=None, x_res=1, y_res=1):
        '''Initialize the painter.
        Parameters:
            brushes: a list of dictionaries, with each such dictionary having
                the following format: 
                {
                    'class': [the the class of a brush, e.g., CircleBrush],
                    'random_sample_params': [dictionary of parameters to be
                        passed to the brush's generate_random_brush method,
                        except for the size bounds parameters],
                    'neighbor_params': [dictionary of parameters to be passed to
                        the brush's get_neighbor method, except for the size
                        bounds parameters]
                    'size_bounds': [dictionary of parameters specifying the min
                        and max size parameters of the brush, to be passed to
                        both the brush's generate_random_brush method and the
                        brush's get_neighbor method]
                }
            hillclimbing_params: a dictionary (with keys as strings) specifying
                the parameters to be used for the hill climbing optimization
                algorithm. These parameters are:
                    n_samples: the number of restarts to do when
                        iter < reuse_samples_start_iter
                    best_of_per_restart: When generating initial sample brush
                        parameters to be further optimized by hillclimbing, get
                        this many random samples of brush parameters and pick
                        the one that gives the lowest loss
                    n_opt_iter: the maximum number of iterations to be done by
                        hillclimbing to reduce the loss of a sample
                    n_neighbors: the number of neighbors to generate from a
                        candidate in the hillclimbing algorithm
                    stop_if_no_improvement: whether to stop optimizing a
                        candidate in hillclimbing if no neigbor gives a lower
                        loss than the candidate's loss
            reuse_samples_start_iter: iteration to start reusing the best
                samples from previous iterations of painting when getting new
                samples for the brush stroke to speed up the process.
                None or inf for never
            src_image: the image to try to recreate by painting. NumPy array of
                shape (m, n, c) where m is the number of rows in the image, n is
                the number of columns, and c is the number of channels
            init_painting: the initial painting. NumPy array of shape (m, n, c).
            max_n_pixels_regression: When finding the optimal color and opacity
                of a brush, if the are of the brush in pixels exceeds this,
                then take a random sample of this many pixels to do the
                regression. None or inf to disable this feature.
            x_res: Resolution of brushes in x direction. Divides image into
                squares of this width for increased performance.
            y_res: Resolution of brushes in x direction. Divides image into
                squares of this height for increased performance.
        '''
        assert len(src_image.shape) == 3
        assert init_painting.shape == src_image.shape
        m, n, c = src_image.shape
        self._m = m
        self._n = n
        self._c = c
        self._src_image = src_image
        self._painting = init_painting

        assert x_res >= 1 and y_res >= 1
        assert m % y_res == 0
        assert n % x_res == 0
        self._x_res = x_res
        self._y_res = y_res

        m_squares = m // y_res
        n_squares = n // x_res
        self._m_squares = m_squares
        self._n_squares = n_squares

        self._use_brush_approximation = not (x_res == 1 and y_res == 1)
        # self._use_brush_approximation = True

        if self._use_brush_approximation:
            # Initialize these for use by approximation method
            print('Initializing squares values')
            t0 = time()
            X_sections = np.empty((m_squares, n_squares, y_res, x_res, c))
            Y_sections = np.empty((m_squares, n_squares, y_res, x_res, c))
            row = 0
            for i in range(m_squares):
                col = 0
                for j in range(n_squares):
                    X_sections[i, j] = init_painting[row:row+y_res, col:col+x_res]
                    Y_sections[i, j] = src_image[row:row+y_res, col:col+x_res]
                    col += x_res
                row += y_res

            xbars = X_sections.mean(axis=(2, 3))
            ybars = Y_sections.mean(axis=(2, 3))

            xdiffs = X_sections - xbars[:, :, None, None, :]
            ydiffs = Y_sections - ybars[:, :, None, None, :]
            self._sxxs = np.sum(xdiffs**2, axis=(2, 3, 4))
            self._sxys = np.sum(xdiffs * ydiffs, axis=(2, 3, 4))
            print(f'Initializing squares values took {time() - t0} seconds')

            self._xbars = xbars
            self._ybars = ybars

        self._curr_loss = np.sum((init_painting - src_image)**2)

        self._n_iters = 0

        self._brush_specs = []
        self._brush_type_to_random_brush_func = {}
        self._brush_type_to_neighbor_func = {}
        self._brush_types = []
        for brush_spec in brushes:
            brush_type = brush_spec['class']
            random_sample_params = brush_spec['random_sample_params']
            neighbor_params = brush_spec['neighbor_params']
            size_bounds = brush_spec['size_bounds']

            random_brush_func = partial(brush_type.generate_random_brush,
                m=m_squares, n=n_squares,
                **random_sample_params, **size_bounds)

            neighbor_func = partial(brush_type.get_neighbor,
                m=m_squares, n=n_squares,
                **neighbor_params, **size_bounds)

            self._brush_specs.append((random_brush_func, neighbor_func))
            self._brush_type_to_random_brush_func[brush_type] = random_brush_func
            self._brush_type_to_neighbor_func[brush_type] = neighbor_func
            self._brush_types.append(brush_type)

        assert reuse_samples_start_iter is None or reuse_samples_start_iter == np.inf or type(reuse_samples_start_iter) is int
        self._reuse_samples_start_iter = np.inf if reuse_samples_start_iter is None else reuse_samples_start_iter
        reuse_samples = self._reuse_samples_start_iter != np.inf
        if reuse_samples:
            self._items = []
            self._queue_initialized = False

            self._n_samples = hillclimbing_params['n_samples']
            self._best_of_per_restart = hillclimbing_params['best_of_per_restart']
            self._n_opt_iter = hillclimbing_params['n_opt_iter']
            self._n_neighbors = hillclimbing_params['n_neighbors']
            self._stop_if_no_improvement = hillclimbing_params['stop_if_no_improvement']

        self._hillclimbing_params = hillclimbing_params
        self._n_queue = n_queue

        self._max_n_pixels_regression = np.inf if max_n_pixels_regression is None else max_n_pixels_regression

    def _evaluate_brush_loss_no_approximation(self, brush, random_sample=True):
        m, n = self._m, self._n
        painting = self._painting
        src = self._src_image

        min_row, max_row, row_specs = brush.get_boundary_rows(m, n)
        assert len(row_specs) == max_row - min_row + 1
        rows = range(min_row, max_row + 1)

        X_rows = []
        Y_rows = []

        for row, (min_col, max_col) in zip(rows, row_specs):
            X_rows.append(painting[row, min_col : max_col + 1])
            Y_rows.append(src[row, min_col : max_col + 1])

        original_n_pixels = brush._n_pixels

        if random_sample and original_n_pixels > self._max_n_pixels_regression:
            num_rows = max_row - min_row + 1
            if num_rows <= 5:
                X = np.concatenate(X_rows)
                Y = np.concatenate(Y_rows)
                indices = np.random.choice(original_n_pixels, size=self._max_n_pixels_regression, replace=False)
                X = X[indices]
                Y = Y[indices]
            else:
                prop = self._max_n_pixels_regression / original_n_pixels
                n_rows_to_pick = int(round(prop * num_rows))
                indices = np.random.choice(num_rows, size=n_rows_to_pick, replace=False)
                X_rows = [X_rows[i] for i in indices]
                Y_rows = [Y_rows[i] for i in indices]
                X = np.concatenate(X_rows)
                Y = np.concatenate(Y_rows)
        else:
            X = np.concatenate(X_rows)
            Y = np.concatenate(Y_rows)

        xbar, ybar, sxx, sxy, alpha, beta = get_least_squares_paint_regression_coefs(X, Y)

        # print(sxx, sxy, xbar, ybar, 'original')

        # enforce that the brush opacity and color are in range
        o = 1 - beta
        if not (0 < o <= 1):
            return np.inf, (alpha, beta)
        c = alpha / o
        if not np.all(np.logical_and(0 <= c, c <= 1)):
            return np.inf, (alpha, beta)

        N = X.shape[0]

        # Calculate the loss. Equivalent to calculating the following:
        #    np.sum((alpha + beta * X - Y)**2) - np.sum((X - Y)**2)
        loss = -(beta - 1)**2 * sxx - N * np.sum((xbar - ybar)**2)

        if random_sample and original_n_pixels > self._max_n_pixels_regression:
            loss = loss / N * original_n_pixels

        return loss, (alpha, beta)
    
    def _evaluate_brush_loss_approximation(self, brush):
        m, n = self._m_squares, self._n_squares
        sxxs, sxys = self._sxxs, self._sxys
        xbars, ybars = self._xbars, self._ybars

        min_row, max_row, row_specs = brush.get_boundary_rows(m, n)
        assert len(row_specs) == max_row - min_row + 1
        n_squares = brush._n_pixels

        brush_xbars_rows, brush_ybars_rows = [], []
        brush_sxxs_rows, brush_sxys_rows = [], []
        for row, (min_col, max_col) in zip(range(min_row, max_row + 1), row_specs):
            brush_xbars_rows.append(xbars[row, min_col : max_col+1])
            brush_ybars_rows.append(ybars[row, min_col : max_col+1])
            brush_sxxs_rows.append(sxxs[row, min_col:max_col+1])
            brush_sxys_rows.append(sxys[row, min_col:max_col+1])
        brush_xbars = np.vstack(brush_xbars_rows)
        brush_ybars = np.vstack(brush_ybars_rows)
        brush_sxxs = np.concatenate(brush_sxxs_rows)
        brush_sxys = np.concatenate(brush_sxys_rows)

        # Compute xbar, ybar
        xbar, ybar = brush_xbars.mean(axis=0), brush_ybars.mean(axis=0)

        # Compute sxx, sxy
        Nk = self._x_res * self._y_res
        xdiffs = brush_xbars - xbar
        ydiffs = brush_ybars - ybar
        sxx = np.sum(brush_sxxs) + Nk * np.sum(xdiffs**2)
        sxy = np.sum(brush_sxys) + Nk * np.sum(xdiffs * ydiffs)

        # print(sxx, sxy, xbar, ybar, 'new')

        alpha, beta = get_least_squares_paint_regression_coefs_from_stats(xbar, ybar, sxx, sxy)

        # enforce that the brush opacity and color are in range
        o = 1 - beta
        if not (0 < o <= 1):
            return np.inf, (alpha, beta)
        c = alpha / o
        if not np.all(np.logical_and(0 <= c, c <= 1)):
            return np.inf, (alpha, beta)

        N = Nk * n_squares
        loss = -(beta - 1)**2 * sxx - N * np.sum((xbar - ybar)**2)

        return loss, (alpha, beta)

    def _evaluate_brush_loss(self, brush, random_sample=True):
        if self._use_brush_approximation:
            return self._evaluate_brush_loss_approximation(brush)
        else:
            return self._evaluate_brush_loss_no_approximation(brush, random_sample)

        # loss1, (alpha1, beta1) = self._evaluate_brush_loss_approximation(brush)
        # loss2, (alpha2, beta2) = self._evaluate_brush_loss_no_approximation(brush, random_sample)

        # print(f"loss: {loss1}, {loss2}, alpha: {alpha1}, {alpha2}, beta: {beta1}, {beta2}")
        # print()

        # return loss1, (alpha1, beta1)

    def _apply_brush(self, brush, alpha, beta):
        painting = self._painting
        m, n = self._m_squares, self._n_squares
        min_row, max_row, row_specs = brush.get_boundary_rows(m, n)
        rows = np.arange(min_row, max_row + 1)

        if self._use_brush_approximation:
            x_res, y_res = self._x_res, self._y_res
            sxxs, sxys = self._sxxs, self._sxys
            xbars = self._xbars
            
            for row, (min_col, max_col) in zip(rows, row_specs):
                y_min, y_max1 = row * y_res, row * y_res + y_res
                x_min, x_max1 = min_col * x_res, (max_col + 1) * x_res
                painting[y_min:y_max1, x_min:x_max1] = alpha + beta * painting[y_min:y_max1, x_min:x_max1]

                sxxs[row, min_col : max_col+1] *= beta**2
                sxys[row, min_col : max_col+1] *= beta
                xbars[row, min_col : max_col+1] = alpha + beta * xbars[row, min_col : max_col+1]
        else:
            for row, (min_col, max_col) in zip(rows, row_specs):
                painting[row, min_col : max_col + 1] = alpha + beta * painting[row, min_col : max_col + 1]

    @dataclass(order=True)
    class PrioritizedBrush:
        loss: float
        brush: Brush=field(compare=False)
        params: Any=field(compare=False)

    def _add_item(self, item):
        self._items.append(item)

    def _pop_item(self):
        min_item = min(self._items)
        self._items.remove(min_item)
        return min_item

    def _get_new_item(self, brush_type, random_brush_func=None):
        if random_brush_func is None:
            random_brush_func = self._brush_type_to_random_brush_func[brush_type]
        neighbor_func = self._brush_type_to_neighbor_func[brush_type]

        # Sometimes, hill_climbing returns None
        # so we keep doing it until we get a legit brush
        brush = None
        while brush is None:
            brush, loss, params = hill_climbing(
                self._evaluate_brush_loss, random_brush_func, neighbor_func,
                n_samples=1,
                best_of_per_restart=self._best_of_per_restart,
                n_opt_iter=self._n_opt_iter,
                n_neighbors=self._n_neighbors,
                stop_if_no_improvement=self._stop_if_no_improvement)
        
        if brush._n_pixels > self._max_n_pixels_regression:
            # re-evaluate the brush loss, without random sample
            loss, params = self._evaluate_brush_loss(brush, random_sample=False)
            if loss == np.inf:
                return self._get_new_item(brush_type, random_brush_func)

        return self.PrioritizedBrush(loss=loss, brush=brush, params=params)

    def _add_random_brush(self):
        brush_type = random.choice(self._brush_types)
        item = self._get_new_item(brush_type)
        self._add_item(item)

    def paint_stroke(self):
        reuse_samples = self._n_iters + 1 >= self._reuse_samples_start_iter

        if reuse_samples:
            ############# Get the brush to use from the queue #############
            if self._queue_initialized:
                best_item = self._pop_item()
            else:
                # If queue not initialized yet, we need to do that
                # initialize the queue and keep track of the best brush params
                print("Initializing queue...")
                best_loss = np.inf
                best_params = None
                for _ in trange(self._n_queue):
                    for brush_type in self._brush_types:
                        item = self._get_new_item(brush_type)
                        self._add_item(item)
                        loss, params = item.loss, item.params
                        if loss < best_loss:
                            best_loss = loss
                            best_params = params

                self._queue_initialized = True

                # get the best brush and params
                best_item = self._pop_item()

            ############ Apply the brush ###########
            best_brush, best_loss, best_params = best_item.brush, best_item.loss, best_item.params

            self._curr_loss += best_loss
            best_alpha, best_beta = best_params
            self._apply_brush(best_brush, best_alpha, best_beta)

            ###########  Remove brushes that intersect the added brush, and add new brushes back
            queue_size = len(self._items)
            self._items = [x for x in self._items if not best_brush.intersects(x.brush)]
            n_to_add = queue_size - len(self._items)
            # rg = trange(n_to_add)
            # print(n_to_add)
            rg = range(n_to_add)
            for _ in rg:
                self._add_random_brush()

            ########## Put a new brush back in queue ###########
            self._add_random_brush()

            self._n_iters += 1

            return best_brush
        else:
            best_brush = None
            best_loss = np.inf
            best_params = None
            for random_brush_func, neighbor_func in self._brush_specs:
                brush, loss, params = hill_climbing(
                    self._evaluate_brush_loss, random_brush_func, neighbor_func, **self._hillclimbing_params)
                if loss < best_loss:
                    best_brush = brush
                    best_loss = loss
                    best_params = params

            if best_brush is not None:
                if best_brush._n_pixels > self._max_n_pixels_regression:
                    # re-evaluate the brush loss, without random sample
                    best_loss, best_params = self._evaluate_brush_loss(best_brush, random_sample=False)

                    if best_loss == np.inf:
                        return self.paint_stroke()

                self._curr_loss += best_loss
                best_alpha, best_beta = best_params
                self._apply_brush(best_brush, best_alpha, best_beta)

                self._n_iters += 1

            return best_brush


def make_painting(brushes, hillclimbing_params, reuse_samples_start_iter, src_image, n_iter, n_queue, max_n_pixels_regression, x_res=1, y_res=1, save_every=None, folder_name=None):
    '''Creates a painting of the image src_image.'''

    painter = Painter(brushes, hillclimbing_params, reuse_samples_start_iter, src_image, init_painting=np.ones(src_image.shape), n_queue=n_queue, max_n_pixels_regression=max_n_pixels_regression, x_res=x_res, y_res=y_res)

    print(f"Initial loss: {painter._curr_loss}")

    if save_every is not None:
        assert folder_name is not None
        if not os.path.isdir(folder_name):
            os.mkdir(folder_name)

    pbar = tqdm(total=n_iter)
    loss_lines = []

    for i in range(n_iter):
        best_brush = painter.paint_stroke()

        if best_brush is not None:
            desc = f"Iteration {i+1}: Loss {painter._curr_loss:.5g}, brush: {best_brush}"
        else:
            desc = f"Iteration {i+1}: Loss {painter._curr_loss:.5g}, could not improve"

        loss_lines.append(f"{i+1}\t{painter._curr_loss:.2f}\n")

        if (i+1) % save_every == 0:
            fname = os.path.join(folder_name, f'{i+1}.png')
            im_arr = (painter._painting * 255).astype(np.uint8)
            im = Image.fromarray(im_arr)
            im.save(fname)

            with open(os.path.join(folder_name, 'losses.txt'), 'w+') as f:
                f.writelines(loss_lines)
        
        pbar.set_description(desc)
        pbar.update()

    pbar.close()

    return painter._painting, painter._curr_loss


_f = lambda x, n: np.mean(np.split(x, n, axis=1), axis=2)
def pixelate(matrix, size=2):  
  n_a = matrix.shape[1] // size
  n_b = matrix.shape[0] // size
  return _f(_f(matrix, n_a), n_b)


def main():
    image_name = 'image-52.jpg'

    im = Image.open(image_name)
    arr = np.asarray(im)
    assert len(arr.shape) == 3
    if arr.shape[2] == 4:
        arr = arr[:, :, :3] # remove alpha channel if there is one
    arr = arr / 255 # normalize to be between 0 and 1

    # arr = pixelate(arr, size=4)

    # params = {
    #     'n_iter': 50_000,
    #     'brushes': [
    #         {
    #             'class': CircleBrush,
    #             'random_sample_params': {},
    #             'neighbor_params': {
    #                 'brush_position_delta': 20,
    #                 'radius_change_factor': 1.05
    #             },
    #             'size_bounds': {
    #                 'min_radius': 1,
    #                 'max_radius': 100
    #             }
    #         },
    #         {
    #             'class': RectangleBrush,
    #             'random_sample_params': {},
    #             'neighbor_params': {
    #                 'brush_position_delta': 20,
    #                 'angle_delta': np.pi/9,
    #                 'width_change_factor': 1.05,
    #                 'length_change_factor': 1.05
    #             },
    #             'size_bounds': {
    #                 'min_width': 2,
    #                 'max_width': 105,
    #                 'max_length': 300
    #             }
    #         }
    #     ],
    #     'hillclimbing_params': {
    #         'n_samples': 20,
    #         'best_of_per_restart': 20,
    #         'n_opt_iter': 20,
    #         'n_neighbors': 15,
    #         'stop_if_no_improvement': True
    #     },
    #     'reuse_samples_start_iter': 100,
    #     'n_queue': 500,
    #     'max_n_pixels_regression': 5_000
    # }

    # parameters for testing
    # params = {
    #     'n_iter': 100,
    #     'brushes': [
    #         {
    #             'class': CircleBrush,
    #             'random_sample_params': {},
    #             'neighbor_params': {
    #                 'brush_position_delta': 20,
    #                 'radius_change_factor': 1.05
    #             },
    #             'size_bounds': {
    #                 'min_radius': 1,
    #                 'max_radius': 100
    #             }
    #         },
    #         {
    #             'class': RectangleBrush,
    #             'random_sample_params': {},
    #             'neighbor_params': {
    #                 'brush_position_delta': 20,
    #                 'angle_delta': np.pi/9,
    #                 'width_change_factor': 1.05,
    #                 'length_change_factor': 1.05
    #             },
    #             'size_bounds': {
    #                 'min_width': 2,
    #                 'max_width': 105,
    #                 'max_length': 300
    #             }
    #         }
    #     ],
    #     'hillclimbing_params': {
    #         'n_samples': 10,
    #         'best_of_per_restart': 20,
    #         'n_opt_iter': 10,
    #         'n_neighbors': 5,
    #         'stop_if_no_improvement': True
    #     },
    #     'reuse_samples_start_iter': 50,
    #     'n_queue': 50,
    #     'max_n_pixels_regression': None,
    #     'x_res': 1,
    #     'y_res': 1
    # }

    params = {
        'n_iter': 300,
        'brushes': [
            {
                'class': CircleBrush,
                'random_sample_params': {},
                'neighbor_params': {
                    'brush_position_delta': 5,
                    'radius_change_factor': 1.05
                },
                'size_bounds': {
                    'min_radius': 1,
                    'max_radius': 25
                }
            },
            {
                'class': RectangleBrush,
                'random_sample_params': {},
                'neighbor_params': {
                    'brush_position_delta': 5,
                    'angle_delta': np.pi/9,
                    'width_change_factor': 1.05,
                    'length_change_factor': 1.05
                },
                'size_bounds': {
                    'min_width': 2,
                    'max_width': 26,
                    'max_length': 75
                }
            }
        ],
        'hillclimbing_params': {
            'n_samples': 10,
            'best_of_per_restart': 20,
            'n_opt_iter': 10,
            'n_neighbors': 5,
            'stop_if_no_improvement': True
        },
        'reuse_samples_start_iter': 50,
        'n_queue': 50,
        'max_n_pixels_regression': None,
        'x_res': 4,
        'y_res': 4
    }

    # pr = cProfile.Profile()
    # pr.enable()

    tm = datetime.datetime.now()
    folder_name = f'painting_{tm.year}-{tm.month:02}-{tm.day:02}T{tm.hour:02}_{tm.minute:02}_{tm.second:02}'

    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)
    
    with open(os.path.join(folder_name, 'params.json'), 'w+') as f:
        # To avoid the error "Object of type type is not JSON serializable"
        params_copy = deepcopy(params)
        for d in params_copy['brushes']:
            d['class'] = d['class'].__name__
        json.dump(params_copy, f, indent=4)

    save_every = 50
    t0 = time()
    painting, loss = make_painting(
        params['brushes'], params['hillclimbing_params'],
        params['reuse_samples_start_iter'], src_image=arr,
        n_iter=params['n_iter'], n_queue=params['n_queue'],
        max_n_pixels_regression=params['max_n_pixels_regression'],
        x_res=params['x_res'], y_res=params['y_res'],
        save_every=save_every, folder_name=folder_name)
    dt = time() - t0

    time_taken_string = f"Time taken: {dt:.6f}s"
    print(time_taken_string)

    # pr.disable()
    # s = io.StringIO()
    # ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    # ps.print_stats(50)
    # print(s.getvalue())

    with open(os.path.join(folder_name, 'time_taken.txt'), 'w+') as f:
        f.write(time_taken_string + '\n')


if __name__ == '__main__':
    main()
