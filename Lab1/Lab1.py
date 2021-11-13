import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2, t
import scipy.stats as stats


def check_pearson_criteria(e_matrix, t_matrix, n):
    chi2_value = stats.chi2.ppf(0.95, t_matrix.size - 1)
    chi2 = n * np.sum((e_matrix - t_matrix) ** 2 / t_matrix)
    pass_criteria = chi2 < chi2_value
    if pass_criteria:
        print('Факт. данные не противоречат ожидаемым исходя из критерия Пирсона.')


# теоретическое мат. ожидание
def find_t_expected_value(values, p):
    e = 0
    for i in range(len(values)):
        e += values[i] * p[i]
    return e


# эмпирическое мат. ожидание
def find_e_expected_value(values):
    return np.mean(values)


# теоретическая дисперсия
def find_t_dispersion(values, p, m):
    d = 0
    for i in range(len(values)):
        d += (values[i] ** 2) * p[i]
    d -= m ** 2
    return d


# эмпирическая дисперсия
def find_e_dispersion(values, expected_value):
    d = 0
    for i in range(len(values)):
        d += (values[i] - expected_value) ** 2
    return d / (len(values) - 1)


def find_correlation_coef(x, y, expected_x, expected_y, dispersion_x, dispersion_y):
    c = 0
    for i in range(len(x)):
        c += (x[i] - expected_x) * (y[i] - expected_y)
    return c / (np.sqrt(dispersion_y * dispersion_x * ((len(x) - 1) ** 2)))


def find_intervals_for_expected(expected, dispersion, n, a):
    diff = np.sqrt(dispersion) * t.ppf(1 - a / 2, n - 1) / np.sqrt(n - 1)
    return expected - diff, expected + diff


def find_intervals_for_dispersion(dispersion, n, a):
    x1 = n * dispersion / chi2.isf((1 - a) / 2, n - 1)
    x2 = n * dispersion / chi2.isf((1 + a) / 2, n - 1)
    return x1, x2


def p_x(P_matrix):
    p_x = []
    for i in range(np.shape(P_matrix)[0]):
        p_x.append(sum(P_matrix[i]))
    return p_x


def p_y(P_matrix):
    p_y = []
    for i in range(np.shape(P_matrix)[1]):
        temp = []
        for j in range(np.shape(P_matrix)[0]):
            temp.append(P_matrix[j][i])
        p_y.append(sum(temp))

    return p_y


def find_index(p):
    d = np.zeros(len(p))
    d[0] = p[0]
    for i in range(len(p)):
        d[i] = d[i - 1] + p[i]
    r = random.uniform(0, d[len(d) - 1])
    h = len(d) - 1
    i = 0
    while i < h:
        middle = (i + h) // 2
        if r > d[middle]:
            i = middle + 1
        else:
            h = middle
    return i


def find_empirical_matrix(X, Y, p, N):
    def find_p_y_x(x_index, ):
        p_y_x = []
        for j in range(np.shape(p)[0]):
            p_y_x.append(p[x_index][j] / P_X[x_index])
        return p_y_x

    P_X = p_x(p)
    e_X = []
    e_Y = []
    e_P = np.zeros(np.shape(p))
    for i in range(N):
        x_index = find_index(P_X)
        x = X[x_index]
        e_X.append(x)
        y_index = find_index(find_p_y_x(x_index))
        y_x_k = Y[y_index]
        e_Y.append(y_x_k)
        e_P[x_index][y_index] = e_P[x_index][y_index] + 1 / N
    return e_P, e_X, e_Y


def plot_diagrams(x1, y1, x2, y2, title1, title2):
    fig, ax = plt.subplots(nrows=1, ncols=2)
    ax[0].bar(x1, y1)
    ax[0].set_title(title1)

    ax[1].bar(x2, y2)
    ax[1].set_title(title2)

    plt.show()


if __name__ == '__main__':
    X = np.array([2, 3, 4])
    Y = np.array([2, 3, 4])
    P = np.array([[0.2, 0.15, 0.1],
                  [0.1, 0.1, 0.05],
                  [0.1, 0.15, 0.05]])
    n = 10000
    a = 0.025

    print("Теоретическая матрица:")
    print(P)

    empiric_P, empirical_X, empirical_Y = find_empirical_matrix(p=P, X=X, Y=Y, N=n)

    print("Эмпирическая матрица вероятностей при n = " + str(n) + ": ")
    print(empiric_P)

    theoretical_expected_X = find_t_expected_value(X, p_x(P))
    empiric_expected_X = find_e_expected_value(empirical_X)
    theoretical_expected_Y = find_t_expected_value(Y, p_y(P))
    empiric_expected_Y = find_e_expected_value(empirical_Y)

    print("Теоретическое мат. ожидание X:", theoretical_expected_X)
    print("Эмпирическое мат. ожидание X:", empiric_expected_X)
    print("Теоретическое мат. ожидание Y:", theoretical_expected_Y)
    print("Эмпирическое мат. ожидание Y:", empiric_expected_Y)

    d_X = find_t_dispersion(X, p_x(P), theoretical_expected_X)
    empiric_dispersion_X = find_e_dispersion(empirical_X, empiric_expected_X)
    d_Y = find_t_dispersion(Y, p_y(P), theoretical_expected_Y)
    empiric_dispersion_Y = find_e_dispersion(empirical_Y, empiric_expected_Y)

    print("Теоритическая дисперсия X:", d_X)
    print("Эмпирическая дисперсия X:", empiric_dispersion_X)
    print("Теоритическая дисперсия Y:", d_Y)
    print("Эмпирическая дисперсия Y:", empiric_dispersion_Y)

    plot_diagrams(X, p_x(empiric_P), X, p_x(P), "Эмпирическое значение X", "Теоретическое значение X")
    plot_diagrams(Y, p_y(empiric_P), Y, p_y(P), "Эмпирическое значение Y", "Теоретическое значение Y")

    print("Интервальная оценка мат. ожидания X",
          find_intervals_for_expected(empiric_expected_X, empiric_dispersion_X, len(empirical_X), a))
    print("Интервальная оценка мат. ожидания Y",
          find_intervals_for_expected(empiric_expected_Y, empiric_dispersion_Y, len(empirical_Y), a))
    print("Интервальная оценка дисперсии X",
          find_intervals_for_dispersion(empiric_dispersion_X, len(empirical_X), a))
    print("Интервальная оценка дисперсии Y",
          find_intervals_for_dispersion(empiric_dispersion_Y, len(empirical_Y), a))

    print("Коэффициент корреляции Пирсона: ")
    print(find_correlation_coef(x=empirical_X,
                                expected_x=empiric_expected_X,
                                dispersion_x=empiric_dispersion_X,
                                y=empirical_Y,
                                expected_y=empiric_expected_Y,
                                dispersion_y=empiric_dispersion_Y)
          )

    check_pearson_criteria(P, empiric_P, n)
