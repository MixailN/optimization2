import numpy as np
import sympy as sp

LAMBDA_MAX = 1e-2
LAMBDA_MIN = 0


def golden_ratio(function, epsilon):
    iteration = 0
    left_border = LAMBDA_MIN
    right_border = LAMBDA_MAX
    x1 = 0
    while right_border - left_border > epsilon:
        iteration += 1
        x1 = left_border + 0.381966011 * (right_border - left_border)
        x2 = right_border - 0.381966011 * (right_border - left_border)
        fx1 = function(x1)
        fx2 = function(x2)

        if fx1 > fx2:
            left_border = x1

        if fx1 < fx2:
            right_border = x2

        if fx1 == fx2:
            left_border = x1
            right_border = x2

    return x1


def function1(x1, x2):
    return 100*(x2 - x1**2)**2 + (1 - x1)**2


def function2(x1, x2):
    return (x2 - x1**2)**2 + (1 - x1)**2


def function3(x1, x2):
    return (1.5 - x1*(1 - x2))**2 + (2.25 - x1 * (1 - x2*2))**2 + (2.625 - x1*(1 - x2**3))**2


def function4(x1, x2, x3, x4):
    return (x1 + x2)**2 + 5*(x3 - x4)**2 + (x2 - 2*x3)**4 + 10*(x1-x4)**4


def dict_for_grad(arguments, point):
    return {arguments[i]: point[i] for i in range(len(arguments))}


def dict_for_func(arguments, point, grad_value_for_index, index):
    dict = {}
    for i in range(len(arguments)):
        if index == i:
            dict[arguments[i]] = grad_value_for_index
        else:
            dict[arguments[i]] = point[i]
    return dict


def fast_descent_methods(func, arguments):
    argue_count = len(arguments)
    point = [0 for i in range(argue_count)]
    for k in range(1000):
        for i in range(argue_count):
            grad = func.diff(arguments[i])
            dict_grad = dict_for_grad(arguments, point)
            grad_value = grad.evalf(subs=dict_grad)
            lambda_min = \
                golden_ratio(lambda l:
                             func.evalf(subs=dict_for_func(arguments, point, point[i] - l * grad_value, i)), 0.0001)
            point[i] -= lambda_min * grad_value
    return point


def coord_descent(function, arguments):
    eps = 0.01
    argue_count = len(arguments)
    point = [2 for i in range(argue_count)]

    max_iter = 1000
    step = 0.001

    for k in range(max_iter):
        copy_point = point.copy()
        for i in range(argue_count):
            cur_func = function.copy()

            for arg in range(argue_count):
                if arg == i:
                    continue
                cur_func = cur_func.subs(arguments[arg], point[arg])

            div = cur_func.diff(arguments[i])

            div_val = div.evalf(subs={arguments[i]: point[i]})

            point[i] = point[i] - step * div_val

            if np.abs(point[i] - copy_point[i]) < eps:
                return point

    return point


x1, x2, x3, x4 = sp.symbols('x1 x2 x3 x4')
func = function1(x1, x2)
# f = func.diff(x1)
# y = sp.lambdify((x1, x2), f)
# a = y(5, 4)
# b = f.evalf(subs={x1: 5, x2: 4})
point = fast_descent_methods(func, [x1, x2])
print(point)
point = coord_descent(func, [x1, x2])
print(point)
print(func)


