import pandas as pd
import numpy as np
from sympy import Symbol
import sympy
import sys
import math
import matplotlib.pyplot as plt

xx = Symbol('x')


def do_research(x_arr, y_arr):
    sko_min = 1000000000
    eq_min = 0
    label_min = 0
    for func in define_funcs():
        try:
            sko, eq, label = func(x_arr, y_arr)
        except ValueError:
            print("Выход из ОДЗ!")
            continue
        build_graph(x_arr, y_arr, eq, label)
        if sko_min > sko:
            sko_min = sko
            eq_min = eq
            label_min = label

    print("Полученная функция -", label_min, ":", eq_min)
    print("СКО:", sko_min)
    # реализовать методы


def build_graph(x_arr, y_arr, eq, label):
    plt.grid(True, which='both')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.plot(x_arr, y_arr, "ro")
    min_x = min(x_arr)
    max_x = max(x_arr)
    xm = np.linspace(min_x, max_x, len(x_arr) * 10)
    plt.plot(xm, [eq.subs(xx, x) for x in xm], label=label)

    plt.legend(loc="best", fontsize='x-small')

    plt.show()


def define_funcs():
    return np.array([linear_func, pol_sec_func, pol_third_func, pow_func, log_func, exp_func])


def print_error():
    print("Ошибка ввода данных, попробуйте еще раз")


def print_file_error():
    print("Ошибка ввода данных из файла, попробуйте еще раз")


def file_all_input(file):
    output = input_int(1, 2, file)
    if output is None:
        return None
    elif output == 1:
        output = sys.stdout
    else:
        output = open("output.txt", "w+", encoding="utf-8")

    n = input_int(2, 12, file)
    if n is None:
        return None
    x_arr, y_arr = file_arr_input(n, file)
    if x_arr is None or y_arr is None:
        return None

    do_research(x_arr, y_arr)

    if output != sys.stdout:
        output.close()
    return True


def file_arr_input(n, file_s):
    x_arr = list()
    y_arr = list()
    for i in range(n):
        try:
            x_arr.append(input_float(file_s))
            y_arr.append(input_float(file_s))
        except ValueError:
            return None, None
    return x_arr, y_arr


def keyboard_all_input():
    if keyboard_choose_output_type() == 1:
        output = sys.stdout
    else:
        output = open("output.txt", "w+", encoding="utf-8")
    n = keyboard_n_input(2, 12)
    x_arr, y_arr = keyboard_arr_input(n)

    do_research(x_arr, y_arr)

    if output != sys.stdout:
        output.close()


def keyboard_arr_input(n):
    print("Введите", n, "пар точек в формате x1 y1 x2 y2 ... xn yn, в каждой строчке по символу!")
    while True:
        x_arr = list()
        y_arr = list()
        for i in range(n):
            append_to_arr_from_input(x_arr)
            append_to_arr_from_input(y_arr)
        return x_arr, y_arr


def append_to_arr_from_input(arr):
    while True:
        try:
            arr.append(input_float(sys.stdin))
            break
        except ValueError:
            print_error()


def keyboard_n_input(min_n, max_n):
    print("Введите число точек от", min_n, "до", max_n)
    return keyboard_int_input(min_n, max_n)


def input_float(stream):
    return float(stream.readline())


def input_int(min_n, max_n, stream):
    try:
        num = int(stream.readline())
        if min_n <= num <= max_n:
            return num
        else:
            raise ValueError
    except ValueError:
        return None


def keyboard_int_input(min_n, max_n):
    while True:
        num = input_int(min_n, max_n, sys.stdin)
        if num is None:
            print_error()
        else:
            return num


def keyboard_choose_input_type():
    print("Выберите метод ввода данных: ")
    print("1. С клавиатуры")
    print("2. Из файла (input.txt)")
    return keyboard_int_input(1, 2)


def keyboard_choose_output_type():
    print("Выберите метод вывода данных: ")
    print("1. В консоль")
    print("2. В файл (output.txt)")
    return keyboard_int_input(1, 2)


def approximation(x_arr, y_arr, max_pow):
    coef_arr = list()
    d_arr = list()
    for i in range(max_pow + 1):
        coef_arr.append([sum([x ** (j + i) for x in x_arr]) for j in reversed(range(max_pow + 1))])
        d_arr.append(sum([x ** i * y for x, y in zip(x_arr, y_arr)]))
    # print(coef_arr)
    # print(d_arr)
    return np.linalg.solve(np.array(coef_arr), np.array(d_arr))


def SKO(x_arr, y_arr, eq):
    ans = math.sqrt(sum([(eq.subs(xx, x) - y) ** 2 for x, y in zip(x_arr, y_arr)]) / len(x_arr))
    print("СКО:", ans)
    return ans


def linear_func(x_arr, y_arr):
    coefs = approximation(x_arr, y_arr, 1)
    a = round(coefs[0], 4)
    b = round(coefs[1], 4)
    eq = a * xx + b
    print("linear:")
    print(eq)
    coef_pirson(x_arr, y_arr)
    return SKO(x_arr, y_arr, eq), eq, "linear"


def pol_sec_func(x_arr, y_arr):
    coefs = approximation(x_arr, y_arr, 2)
    a = round(coefs[0], 4)
    b = round(coefs[1], 4)
    c = round(coefs[2], 4)
    eq = a * xx ** 2 + b * xx + c
    print("Pol sec:")
    print(eq)
    return SKO(x_arr, y_arr, eq), eq, "pol 2th"


def pol_third_func(x_arr, y_arr):
    coefs = approximation(x_arr, y_arr, 3)
    a = round(coefs[0], 4)
    b = round(coefs[1], 4)
    c = round(coefs[2], 4)
    d = round(coefs[3], 4)
    eq = a * xx ** 3 + b * xx ** 2 + c * xx + d
    print("Pol third:")
    print(eq)
    return SKO(x_arr, y_arr, eq), eq, "pol 3th"


def exp_func(x_arr, y_arr):
    if min(y_arr) <= 0:
        raise ValueError
    coefs = approximation(x_arr, [np.log(y) for y in y_arr], 1)
    A = coefs[0]
    b = coefs[1]
    eq = round(sympy.exp(b), 4) * sympy.exp(round(A, 4) * xx)
    print("exp:")
    print(eq)
    return SKO(x_arr, y_arr, eq), eq, "exp"


def log_func(x_arr, y_arr):
    if min(x_arr) <= 0:
        raise ValueError
    coefs = approximation([np.log(x) for x in x_arr], y_arr, 1)
    a = round(coefs[0], 4)
    b = round(coefs[1], 4)
    eq = a * sympy.log(xx) + b
    print("log:")
    print(eq)
    return SKO(x_arr, y_arr, eq), eq, "log"


def pow_func(x_arr, y_arr):
    if min(x_arr) <= 0 or min(y_arr) <= 0:
        raise ValueError
    coefs = approximation([np.log(x) for x in x_arr], [np.log(y) for y in y_arr], 1)
    a = round(coefs[0], 4)
    b = round(coefs[1], 4)
    eq = round(sympy.exp(b), 4) * xx ** a
    print("pow:")
    print(eq)
    return SKO(x_arr, y_arr, eq), eq, "pow"


def coef_pirson(x_arr, y_arr):
    x_mid = np.array(x_arr).mean()
    y_mid = np.array(y_arr).mean()
    top = sum((x - x_mid) * (y - y_mid) for x, y in zip(x_arr, y_arr))
    bot = sum((x - x_mid) ** 2 for x in x_arr) * sum((y - y_mid) ** 2 for y in y_arr)
    print("Коэффициент Пирсона", round(top / np.sqrt(bot), 4))
    return float(top / np.sqrt(bot))


if __name__ == "__main__":
    if keyboard_choose_input_type() == 1:
        keyboard_all_input()
    else:
        file_ss = open("input.txt", "r+")
        if file_all_input(file_ss) is None:
            print_file_error()
        file_ss.close()
