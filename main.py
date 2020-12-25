import numpy as np
import sympy as sp
import math

LAMBDA_MAX = 1e-1
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


def dict_for_func_arguments(arguements, point):
    return {arguements[i]: point[i] for i in range(len(arguements))}


def dict_for_changed_point(arguements, point, grad_value_for_index, index):
    dict = {}
    for i in range(len(arguements)):
        if index == i:
            dict[arguements[i]] = grad_value_for_index
        else:
            dict[arguements[i]] = point[i]
    return dict


def fast_descent_methods(func, arguments, epsilon):
    arque_count = len(arguments)
    point = [0 for i in range(arque_count)]
    func_value_prev = func.evalf(subs=dict_for_func_arguments(arguments, point))
    func_value_next = func_value_prev + 2 * epsilon
    while math.fabs(func_value_next - func_value_prev) >= epsilon:
        func_value_prev = func.evalf(subs=dict_for_func_arguments(arguments, point))
        for i in range(arque_count):
            grad = func.diff(arguments[i])
            dict_grad = dict_for_func_arguments(arguments, point)
            grad_value = grad.evalf(subs=dict_grad)
            lambda_min = \
                golden_ratio(lambda l:
                                func.evalf(subs=dict_for_changed_point(arguments, point, point[i] - l * grad_value, i)), 1e-5)
            point[i] -= lambda_min * grad_value
        func_value_next = func.evalf(subs=dict_for_func_arguments(arguments, point))
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


def gradient(arguments, point):
    return np.array([func.diff(arg).evalf(subs=dict_for_func_arguments(arguments, point)) for arg in arguments]).astype('float32')


def ravine_gradient_method(func, arguments, point1, point2, epsilon):
    l = 2e-3
    h = 1e-4
    c = 5e-1
    point1 = np.array(point1).astype('float32')
    point2 = np.array(point2).astype('float32')
    point1_grad = point1 - l * gradient(arguments, point1)
    point2_grad = point2 - l * gradient(arguments, point2)
    while True:
        func_point1_grad = float(func.evalf(subs=dict_for_func_arguments(arguments, point1_grad)))
        func_point2_grad = float(func.evalf(subs=dict_for_func_arguments(arguments, point2_grad)))
        point3 = point2_grad + h * np.sign(func_point1_grad - func_point2_grad) * \
              (point2_grad - point1_grad) / np.linalg.norm(point2_grad - point1_grad)
        point3_grad = point3 - l * gradient(arguments, point3)
        cos1 = np.dot((point2 - point1_grad), (point2_grad - point1_grad)) / \
            (np.linalg.norm(point2 - point1_grad) * np.linalg.norm(point2_grad - point1_grad))
        cos2 = np.dot((point3 - point2_grad), (point3_grad - point2_grad)) / \
            (np.linalg.norm(point3 - point2_grad) * np.linalg.norm(point3_grad - point2_grad))

        h = h * c ** (cos2 - cos1)

        point1, point1_grad = point2, point2_grad
        point2, point2_grad = point3, point3_grad

        func_point1 = float(func.evalf(subs=dict_for_func_arguments(arguments, point1)))
        if func_point1 < 0.2:
            a = 1
        func_point2 = float(func.evalf(subs=dict_for_func_arguments(arguments, point2)))
        if np.linalg.norm(point2 - point1) < epsilon or \
                abs(func_point1 - func_point2) < epsilon:
            break
    return point2


x1, x2, x3, x4 = sp.symbols('x1 x2 x3 x4')
func = function2(x1, x2)
# f = func.diff(x1)
# y = sp.lambdify((x1, x2), f)
# a = y(5, 4)
# b = f.evalf(subs={x1: 5, x2: 4})

point = ravine_gradient_method(func, [x1, x2], [-0.5, 0.3], [0, -0.5], 1e-6)
print(point)
point = fast_descent_methods(func, [x1, x2], 1e-4)
print(point)
point = coord_descent(func, [x1, x2])
print(point)
print(func)


