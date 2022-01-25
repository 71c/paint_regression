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
import threading
import concurrent.futures


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


def func2(m, n, row0, col0, radius):
    min_row = max(row0 - radius, 0)
    max_row = min(row0 + radius, m - 1)

    rows = np.arange(min_row, max_row + 1)
    row_diffs = (rows - row0) / radius * (radius - 0.5)
    brush_radii = np.sqrt(radius**2 - row_diffs**2)
    min_cols = np.fmax(0, np.around(col0 - brush_radii).astype(int))
    max_cols = np.fmin(n-1, np.around(col0 + brush_radii).astype(int))
    row_specs = [(min_cols[i], max_cols[i]) for i in range(max_row - min_row + 1)]

    return min_row, max_row, row_specs


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
        return func2(m, n, row0, col0, radius)
    
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

    # def L(y):
    #     if y1 <= y <= y2:
    #         return (x2 - x1) / (y2 - y1) * (y - y1) + x1
    #     return (x3 - x1) / (y3 - y1) * (y - y1) + x1

    min_row = max(int(np.ceil(row0 + y3)), 0)
    max_row = min(int(np.floor(row0 + y2)) - 1, m-1)

    row_specs = []
    true_min_row, true_max_row = 2*m, -2*m
    for row in range(min_row, max_row + 1):
        y = row - row0 + 0.001

        # x_left = int(round(L(y)))
        # x_right = int(round(-L(-y)))

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


class Painter:
    def __init__(self, brushes, hillclimbing_params, reuse_samples_start_iter, src_image, init_painting, n_queue, max_n_pixels_regression=None):
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
        '''
        assert len(src_image.shape) == 3
        assert init_painting.shape == src_image.shape
        m, n, c = src_image.shape
        self._m = m
        self._n = n
        self._src_image = src_image
        self._painting = init_painting

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
                m=m, n=n, **random_sample_params, **size_bounds)

            neighbor_func = partial(brush_type.get_neighbor, m=m, n=n,
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

        # self._executor = concurrent.futures.ThreadPoolExecutor(4)

    def _evaluate_brush_loss(self, brush, random_sample=True):
        m = self._m
        n = self._n
        painting = self._painting
        arr = self._src_image

        min_row, max_row, row_specs = brush.get_boundary_rows(m, n)
        assert len(row_specs) == max_row - min_row + 1
        rows = range(min_row, max_row + 1)

        painting_brush_rows = []
        arr_brush_rows = []

        for row, (min_col, max_col) in zip(rows, row_specs):
            painting_brush_rows.append(painting[row, min_col : max_col + 1])
            arr_brush_rows.append(arr[row, min_col : max_col + 1])

        original_n_pixels = brush._n_pixels

        # painting_brush_pixels = np.concatenate(painting_brush_rows)
        # arr_brush_pixels = np.concatenate(arr_brush_rows)

        # if random_sample and original_n_pixels > self._max_n_pixels_regression:
        #     indices = np.random.choice(original_n_pixels, size=self._max_n_pixels_regression, replace=False)
        #     painting_brush_pixels = painting_brush_pixels[indices]
        #     arr_brush_pixels = arr_brush_pixels[indices]


        if random_sample and original_n_pixels > self._max_n_pixels_regression:
            num_rows = max_row - min_row + 1
            if num_rows <= 5:
                painting_brush_pixels = np.concatenate(painting_brush_rows)
                arr_brush_pixels = np.concatenate(arr_brush_rows)
                indices = np.random.choice(original_n_pixels, size=self._max_n_pixels_regression, replace=False)
                painting_brush_pixels = painting_brush_pixels[indices]
                arr_brush_pixels = arr_brush_pixels[indices]
            else:
                prop = self._max_n_pixels_regression / original_n_pixels
                n_rows_to_pick = int(round(prop * num_rows))
                indices = np.random.choice(num_rows, size=n_rows_to_pick, replace=False)
                painting_brush_rows = [painting_brush_rows[i] for i in indices]
                arr_brush_rows = [arr_brush_rows[i] for i in indices]
                painting_brush_pixels = np.concatenate(painting_brush_rows)
                arr_brush_pixels = np.concatenate(arr_brush_rows)
        else:
            painting_brush_pixels = np.concatenate(painting_brush_rows)
            arr_brush_pixels = np.concatenate(arr_brush_rows)

        mx, my, sxx, sxy, alpha, beta = get_least_squares_paint_regression_coefs(painting_brush_pixels, arr_brush_pixels)

        # enforce that the brush opacity and color are in range
        o = 1 - beta
        if not (0 < o <= 1):
            return np.inf, (alpha, beta)
        c = alpha / o
        if not np.all(np.logical_and(0 <= c, c <= 1)):
            return np.inf, (alpha, beta)
        
        # original_err = np.sum((painting_brush_pixels - arr_brush_pixels)**2)
        # new_painting_brush_pixels = alpha + beta * painting_brush_pixels
        # new_err = np.sum((new_painting_brush_pixels - arr_brush_pixels)**2)
        # loss = new_err - original_err

        N = painting_brush_pixels.shape[0]
        original_err_2 = sxx - 2 * sxy + N * np.sum((mx - my)**2)
        new_err_2 = beta**2 * sxx - 2 * beta * sxy
        loss = new_err_2 - original_err_2

        if random_sample and original_n_pixels > self._max_n_pixels_regression:
            loss = loss / painting_brush_pixels.shape[0] * original_n_pixels

        return loss, (alpha, beta)

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

    def get_intersection_avoiding_random_brush_func(self, random_brush_func):
        self._items.sort()
        # curr_best_brushes = self._items[:self._n_queue // 5]
        curr_best_brushes = self._items[:3]
        def f():
            brush = random_brush_func()
            n_tries = 0
            while any([b.brush.intersects(brush) for b in curr_best_brushes]) and n_tries < 10:
                brush = random_brush_func()
                n_tries += 1
            return brush
        return f

    def _add_random_brush(self):
        brush_type = random.choice(self._brush_types)

        item = self._get_new_item(brush_type)

        # random_brush_func = self.get_intersection_avoiding_random_brush_func(self._brush_type_to_random_brush_func[brush_type])
        # item = self._get_new_item(brush_type, random_brush_func)

        self._add_item(item)
    
    def _add_n_brushes(self, n):
        for _ in range(n):
            self._add_random_brush()

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
                t0 = time()
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
                print(f"Took {time() - t0} seconds")
                
                # t0 = time()
                # brush_types = [brush_type for _ in range(self._n_queue) for brush_type in self._brush_types]
                # result = self._executor.map(self._get_new_item, brush_types)
                # best_loss = np.inf
                # best_params = None
                # for item in result:
                #     self._add_item(item)
                #     loss, params = item.loss, item.params
                #     if loss < best_loss:
                #         best_loss = loss
                #         best_params = params
                # print(f"Took {time() - t0} seconds")

                self._queue_initialized = True

                # get the best brush and params
                best_item = self._pop_item()

            ############ Apply the brush ###########
            best_brush, best_loss, best_params = best_item.brush, best_item.loss, best_item.params

            if best_brush._n_pixels > self._max_n_pixels_regression:
                # re-evaluate the brush loss, without random sample
                best_loss, best_params = self._evaluate_brush_loss(best_brush, random_sample=False)

            if best_loss == np.inf:
                print('A?')

            self._curr_loss += best_loss
            best_alpha, best_beta = best_params
            apply_brush(self._painting, best_brush, best_alpha, best_beta)

            ###########  Remove brushes that intersect the added brush, and add new brushes back
            queue_size = len(self._items)
            self._items = [x for x in self._items if not best_brush.intersects(x.brush)]
            n_to_add = queue_size - len(self._items)
            # rg = trange(n_to_add)
            # print(n_to_add)
            rg = range(n_to_add)

            for _ in rg:
                self._add_random_brush()

            # threads = []
            # for _ in rg:
            #     t = threading.Thread(group=None, target=self._add_random_brush)
            #     t.start()
            #     threads.append(t)
            # for t in threads:
            #     t.join()

            # per_thread = 20
            # n_threads = n_to_add // per_thread
            # threads = []
            # for _ in range(n_threads):
            #     threads.append(threading.Thread(group=None, target=self._add_n_brushes, args=(per_thread,)))
            # n_extra = n_to_add - n_threads * per_thread
            # if n_extra != 0:
            #     threads.append(threading.Thread(group=None, target=self._add_n_brushes, args=(n_extra,)))
            # for t in threads:
            #     t.start()
            # for t in threads:
            #     t.join()

            # res = self._executor.map(self._add_n_brushes, [1 for _ in range(n_to_add)])
            # concurrent.futures.wait(res)
            # print(len(self._items))
            


            # futures = []
            # for _ in range(n_to_add):
            #     future = self._executor.submit(self._add_random_brush)
            #     futures.append(future)
            # concurrent.futures.wait(futures)
            # print(len(self._items))

            ##### should be good...
            # per_thread = 20
            # n_threads = n_to_add // per_thread
            # ns = []
            # for _ in range(n_threads):
            #     ns.append(per_thread)
            # n_extra = n_to_add - n_threads * per_thread
            # if n_extra != 0:
            #     ns.append(n_extra)
            # futures = []
            # for num in ns:
            #     future = self._executor.submit(self._add_n_brushes, num)
            #     futures.append(future)
            # concurrent.futures.wait(futures)
            # print(len(self._items))


            # per_thread = 20
            # n_threads = n_to_add // per_thread
            # threads = []
            # for _ in range(n_threads):
            #     threads.append(per_thread)
            # n_extra = n_to_add - n_threads * per_thread
            # if n_extra != 0:
            #     threads.append(n_extra)
            # self._executor.map(self._add_n_brushes, threads)


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
                        return None

                self._curr_loss += best_loss
                best_alpha, best_beta = best_params
                apply_brush(self._painting, best_brush, best_alpha, best_beta)

                self._n_iters += 1

            return best_brush


def make_painting(brushes, hillclimbing_params, reuse_samples_start_iter, src_image, n_iter, n_queue, max_n_pixels_regression, save_every=None, folder_name=None):
    '''Creates a painting of the image src_image.'''

    painter = Painter(brushes, hillclimbing_params, reuse_samples_start_iter, src_image, init_painting=np.ones(src_image.shape), n_queue=n_queue, max_n_pixels_regression=max_n_pixels_regression)

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


def main():
    image_name = 'image-52.jpg'

    im = Image.open(image_name)
    arr = np.asarray(im)
    assert len(arr.shape) == 3
    if arr.shape[2] == 4:
        arr = arr[:, :, :3] # remove alpha channel if there is one
    arr = arr / 255 # normalize to be between 0 and 1

    params = {
        'n_iter': 10_000,
        'brushes': [
            {
                'class': CircleBrush,
                'random_sample_params': {},
                'neighbor_params': {
                    'brush_position_delta': 20,
                    'radius_change_factor': 1.05
                },
                'size_bounds': {
                    'min_radius': 1,
                    'max_radius': 100
                }
            },
            {
                'class': RectangleBrush,
                'random_sample_params': {},
                'neighbor_params': {
                    'brush_position_delta': 20,
                    'angle_delta': np.pi/9,
                    'width_change_factor': 1.05,
                    'length_change_factor': 1.05
                },
                'size_bounds': {
                    'min_width': 2,
                    'max_width': 105,
                    'max_length': 300
                }
            }
        ],
        # 'hillclimbing_params': {
        #     'n_samples': 1,
        #     'best_of_per_restart': 20,
        #     'n_opt_iter': 6,
        #     'n_neighbors': 3,
        #     'stop_if_no_improvement': False
        # },
        'hillclimbing_params': {
            'n_samples': 10,
            'best_of_per_restart': 20,
            'n_opt_iter': 20,
            'n_neighbors': 15,
            'stop_if_no_improvement': True
        },
        'reuse_samples_start_iter': 100,
        'n_queue': 500,
        'max_n_pixels_regression': 7_000
    }

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
    #         'best_of_per_restart': 4,
    #         'n_opt_iter': 10,
    #         'n_neighbors': 4,
    #         'stop_if_no_improvement': True
    #     },
    #     'reuse_samples_start_iter': 200,
    #     'n_queue': 300,
    #     'max_n_pixels_regression': None
    # }

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

    t0 = time()
    painting, loss = make_painting(params['brushes'], params['hillclimbing_params'], params['reuse_samples_start_iter'], src_image=arr, n_iter=params['n_iter'], n_queue=params['n_queue'], max_n_pixels_regression=params['max_n_pixels_regression'], save_every=50, folder_name=folder_name)
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
