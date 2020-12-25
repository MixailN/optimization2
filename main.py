import numpy as np
import sympy as sp


def function1(x):
    return 100*(x[1] - x[0]**2)**2 + (1 - x[0])**2


def function2(x):
    return (x[1] - x[0]**2)**2 + (1 - x[0])**2


def function3(x):
    return (1.5 - x[0]*(1 - x[1]))**2 + (2.25 - x[0] * (1 - x[1]*2))**2 + (2.625 - x[0]*(1 - x[1]**3))**2


def function4(x):
    return (x[0] + x[1])**2 + 5*(x[2] - x[3])**2 + (x[1] - 2*x[2])**4 + 10*(x[0]-x[3])**4


