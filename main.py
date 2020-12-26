import numpy as np
import sympy as sp
import math
import matplotlib.pyplot as plt


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


def f1_brent(x):
    return -5 * x ** 5 + 4 * x ** 4 - 12 * x ** 3 + 11 * x ** 2 - 2 * x + 1


def function1(x1, x2):
    return 100*(x2 - x1**2)**2 + (1 - x1)**2


def function2(x1, x2):
    return (x2 - x1**2)**2 + (1 - x1)**2


def function3(x1, x2):
    return (1.5 - x1*(1 - x2))**2 + (2.25 - x1 * (1 - x2*2))**2 + (2.625 - x1*(1 - x2**3))**2


def function4(x1, x2, x3, x4):
    return (x1 + x2)**2 + 5*(x3 - x4)**2 + (x2 - 2*x3)**4 + 10*(x1-x4)**4


def get_minimum(x, fx, y, fy):
    if abs(y - x) <= 1e-7:
        return 0

    a = (fy - fx) / (y - x)
    b = fx - a * x

    return - (b / a)


def brent(func, a, b, arguments, eps):
    iters = 0
    max_iters = 1000

    argue_count = len(arguments)
    eps = 1e-3
    print(a, b)
    x = w = v = [0] * argue_count
    for i in range(argue_count):
        x[i] = w[i] = v[i] = (a[i] + b[i]) / 2

    fx = fw = fv = [0] * argue_count
    fdx = fdw = fdv = [0] * argue_count

    for i in range(argue_count):
        cur_func = func.copy()
        for arg in range(argue_count):
            if arg == i:
                continue
            cur_func = cur_func.subs(arguments[arg], x[arg])

        div = cur_func.diff(arguments[i])

        fx[i] = fw[i] = fv[i] = cur_func.evalf(subs={arguments[i]: x[i]})

        fdx[i] = fdw[i] = fdv[i] = div.evalf(subs={arguments[i]: x[i]})

    d = e = [(b[i] - a[i]) for i in range(argue_count)]
    prev_u = [None] * argue_count
    g = [None] * argue_count
    while iters < max_iters:

        iters += 1

        for i in range(argue_count):
            g[i] = e[i]
        for i in range(argue_count):
            e[i] = d[i]

        u = [None] * argue_count
        for i in range(argue_count):

            if x[i] != w[i] and fdx[i] != fdw[i]:
                u[i] = get_minimum(x[i], fx[i], w[i], fw[i])
                if not ((a[i] + eps) <= u[i] <= (b[i] - eps) and abs(u[i] - x[i]) < g[i] / 2):
                    u[i] = None

        u2 = [None] * argue_count
        for i in range(argue_count):
            if x[i] != v[i] and fdx[i] != fdv[i]:
                u2[i] = get_minimum(x[i], fx[i], v[i], fv[i])
                if (a[i] + eps) <= u2[i] <= (b[i] - eps) and abs(u2[i] - x[i]) < g[i] / 2:
                    if u[i] is not None and abs(u2[i] - x[i]) < abs(u[i] - x[i]):
                        u[i] = u2[i]

            if u[i] is None:
                if fdx[i] > 0:
                    u[i] = (a[i] + x[i]) / 2
                else:
                    u[i] = (x[i] + b[i]) / 2

            if abs(u[i] - x[i]) < eps:
                u[i] = x[i] + np.sign(u[i] - x[i]) * eps

        fu = [None] * argue_count
        fdu = [None] * argue_count

        flag = False
        for i in range(argue_count):
            new_func = func.copy()
            for arg in range(argue_count):
                if arg == i:
                    continue
                new_func = new_func.subs(arguments[arg], u[arg])

            d[i] = abs(x[i] - u[i])
            fu[i] = new_func.evalf(subs={arguments[i]: u[i]})
            div = new_func.diff(arguments[i])
            fdu[i] = div.evalf(subs={arguments[i]: u[i]})

            if fu[i] <= fx[i]:
                if u[i] >= x[i]:
                    a[i] = x[i]
                else:
                    b[i] = x[i]

                v[i] = w[i]
                w[i] = x[i]
                x[i] = u[i]

                fv[i] = fw[i]
                fw[i] = fx[i]
                fx[i] = fu[i]

                fdv[i] = fdw[i]
                fdw[i] = fdx[i]
                fdx[i] = fdu[i]
            else:
                if u[i] >= x[i]:
                    b[i] = u[i]
                else:
                    a[i] = u[i]

                if fu[i] <= fw[i] or w[i] == x[i]:
                    v[i] = w[i]
                    w[i] = u[i]

                    fv[i] = fw[i]
                    fw[i] = fu[i]

                    fdv[i] = fdw[i]
                    fdw[i] = fdu[i]

                elif fu[i] <= fv[i] or v[i] == x[i] or v[i] == w[i]:
                    v[i] = u[i]
                    fv[i] = fu[i]
                    fdv[i] = fdu[i]

            if prev_u[i] is not None and abs(prev_u[i] - u[i]) < eps:
                flag = True
                break
        if flag:
            break
        prev_u = u
    return x, iters, func.evalf(subs=dict_for_func_arguments(arguments, x))


def proect_grad(func, arguments, epsilon):
    arque_count = len(arguments)
    point = [0 for i in range(arque_count)]
    func_value_prev = func.evalf(subs=dict_for_func_arguments(arguments, point))
    func_value_next = func_value_prev + 2 * epsilon
    iterations = 1
    while math.fabs(func_value_next - func_value_prev) >= epsilon and iterations < 5000:
        iterations += 1
        func_value_prev = func.evalf(subs=dict_for_func_arguments(arguments, point))
        for i in range(arque_count):
            grad = func.diff(arguments[i])
            dict_grad = dict_for_func_arguments(arguments, point)
            grad_value = grad.evalf(subs=dict_grad)
            lambda_min = \
                golden_ratio(lambda l:
                             func.evalf(
                                 subs=dict_for_changed_point(arguments, point, point[i] - l * grad_value, i)), 1e-5)
            point[i] -= lambda_min * grad_value
        func_value_next = func.evalf(subs=dict_for_func_arguments(arguments, point))
    return point, iterations, func.evalf(subs=dict_for_func_arguments(arguments, point))


def dict_for_grad(arguments, point):
    return {arguments[i]: point[i] for i in range(len(arguments))}


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
    iterations = 1
    while math.fabs(func_value_next - func_value_prev) >= epsilon and iterations < 5000:
        iterations += 1
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
    return point, iterations, func.evalf(subs=dict_for_func_arguments(arguments, point))


def coord_descent(func, arguments, eps):
    argue_count = len(arguments)
    point = [2 for i in range(argue_count)]

    max_iter = 1000
    step = 0.001
    iterations = 1
    for k in range(max_iter):
        iterations += 1
        copy_point = point.copy()
        for i in range(argue_count):
            cur_func = func.copy()

            for arg in range(argue_count):
                if arg == i:
                    continue
                cur_func = cur_func.subs(arguments[arg], point[arg])

            div = cur_func.diff(arguments[i])

            div_val = div.evalf(subs={arguments[i]: point[i]})

            point[i] = point[i] - step * div_val

            if np.abs(point[i] - copy_point[i]) < eps:
                return point

    return point, iterations, func.evalf(subs=dict_for_func_arguments(arguments, point))


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
    iterations = 1
    while True:
        iterations += 1
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
    return point2, iterations, func.evalf(subs=dict_for_func_arguments(arguments, point2))


x1, x2, x3, x4 = sp.symbols('x1 x2 x3 x4')
x = sp.symbols('x')
func = function1(x1, x2)
# f = func.diff(x1)
# y = sp.lambdify((x1, x2), f)
# a = y(5, 4)
# b = f.evalf(subs={x1: 5, x2: 4})
# point = fast_descent_methods(func, [x1, x2])
# print(point)
# point = coord_descent(func, [x1, x2])
# print(point)

point, iterations, func_value = brent(func, [-2, -2], [2, 2], [x1, x2], 1e-6)
print('Brent method:')
print('point: ', point, '\n', 'iterations: ', iterations, '\n', 'func_value: ', func_value)

# point, iterations, func_value = ravine_gradient_method(func, [x1, x2], [-0.5, 0.3], [0, -0.5], 1e-6)
# print('Ravine gradient method:')
# print('point: ', point, '\n', 'iterations: ', iterations, '\n', 'func_value: ', func_value)
# point, iterations, func_value = fast_descent_methods(func, [x1, x2], 1e-5)
# print('Fast descent method:')
# print('point: ', point, '\n', 'iterations: ', iterations, '\n', 'func_value: ', func_value)
# point, iterations, func_value = coord_descent(func, [x1, x2], 1e-6)
# print('Coord descent method:')
# print('point: ', point, '\n', 'iterations: ', iterations, '\n', 'func_value: ', func_value)



