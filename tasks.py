# tasks.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import expon, chi2, t, norm, pearsonr
import pandas as pd

plt.rcParams['figure.dpi'] = 120
plt.rcParams['font.size'] = 11
np.set_printoptions(precision=4, suppress=True)

def task1():
    print('Задание 1')

    n = 475
    a = 0.9
    sigma_squared = 12.96
    gamma = 0.95
    alpha = 0.05

    sigma = np.sqrt(sigma_squared)
    U = np.random.uniform(0, 1, (n, 12))
    Z = np.sum(U, axis=1) - 6
    data = a + sigma * Z

    bins = int(np.round(np.log2(n) + 1))  # Формула Стерджеса

    plt.figure(figsize=(8, 5))

    # Строим гистограмму
    plt.hist(data, bins=bins, density=True, edgecolor='black', alpha=0.7, color='skyblue')

    # Оформление
    plt.title('Гистограмма выборки и теоретическая плотность')
    plt.xlabel('Значения')
    plt.ylabel('Плотность вероятности')

    # Теоретическая плотность
    x_vals = np.linspace(a - 4 * sigma, a + 4 * sigma, 1000)
    theoretical = norm.pdf(x_vals, loc=a, scale=sigma)
    plt.plot(x_vals, theoretical, 'r-', linewidth=2, label=f'Теоретическая N({a}, {sigma_squared})')

    # Добавляем легенду
    plt.legend()

    plt.grid(True, linestyle='-', alpha=0.3)

    # Показываем результат
    plt.savefig('task1_hist.png', dpi=300, bbox_inches='tight')
    plt.show()

    hist_counts, hist_edges = np.histogram(data, bins=bins)

    interval_lengths = np.diff(hist_edges)

    relative_freq = hist_counts / n

    heights = hist_counts / (n * interval_lengths)

    midpoints = (hist_edges[:-1] + hist_edges[1:]) / 2

    theoretical_density = norm.pdf(midpoints, loc=a, scale=sigma)

    df_hist = pd.DataFrame({
        'Левая граница': hist_edges[:-1],
        'Правая граница': hist_edges[1:],
        'Частота ni': hist_counts,
        'Отн. частота wi': relative_freq,
        'Высота гистограммы': heights,
        'Теор. плотность': theoretical_density
    })

    df_hist = df_hist.round(4)
    df_hist.to_excel('table_1_1.xlsx', index=False)

    print('\nТаблица 1.1')
    print(df_hist.to_string(index=False))

    print('\nПункт 1.2:')

    sample_mean = np.mean(data)
    sample_variance_biased = np.var(data, ddof=0)

    print(f'Выборочное среднее: {sample_mean:.4f}')
    print(f'Выборочная дисперсия: {sample_variance_biased:.4f}')

    print(f"\nОтклонение среднего: {sample_mean - 0.9:.6f}")
    print(f"Отклонение дисперсии: {sample_variance_biased - 12.96:.6f}")
    print(f"Относительное отклонение дисперсии: {(sample_variance_biased - 12.96) / 12.96 * 100:.2f}%")

    print('\nПункт 1.3:')

    # Оценки максимального правдоподобия
    ML_mean = np.mean(data)  # = sample_mean
    ML_variance = np.var(data, ddof=0)  # = sample_variance_biased

    # Несмещенная оценка дисперсии
    unbiased_variance = np.var(data, ddof=1)

    print(f'Оценка матожидания (МП): {ML_mean:.6f}')
    print(f'Оценка дисперсии (МП, смещенная): {ML_variance:.6f}')
    print(f'Несмещенная оценка дисперсии: {unbiased_variance:.6f}')

    print('\nПункт 1.4:')

    # Параметры
    gamma = 0.95
    n = 475
    alpha = round(1 - gamma, 3)

    # Несмещенное стандартное отклонение (для интервала матожидания)
    std_unbiased = np.sqrt(unbiased_variance)

    # ----- 1. Доверительный интервал для математического ожидания -----
    t_crit = t.ppf((1 + gamma) / 2, df=n - 1)  # квантиль Стьюдента
    margin = t_crit * std_unbiased / np.sqrt(n)
    ci_mean = (ML_mean - margin, ML_mean + margin)

    print(f'Доверительный интервал для матожидания (γ={gamma}):')
    print(f'  [{ci_mean[0]:.6f}; {ci_mean[1]:.6f}]')

    # ----- 2. Доверительный интервал для дисперсии -----
    chi2_low = chi2.ppf((1 - gamma) / 2, df=n - 1)  # левый квантиль
    chi2_high = chi2.ppf((1 + gamma) / 2, df=n - 1)  # правый квантиль

    ci_var_low = (n - 1) * unbiased_variance / chi2_high
    ci_var_high = (n - 1) * unbiased_variance / chi2_low

    print(f'\nДоверительный интервал для дисперсии (γ={gamma}):')
    print(f'  [{ci_var_low:.6f}; {ci_var_high:.6f}]')

    print('\nПункт 1.5:')
    print(f'Гипотеза H0: данные распределены по закону N({a}, {sigma_squared})')
    print(f'Уровень значимости α = {alpha}')

    # 1. Количество интервалов
    k = int(np.round(1 + np.log2(n)))
    print(f'\n1. Количество интервалов (Стерджес): k = {k}')

    # 2. Наблюдаемые частоты
    observed_freq, bin_edges = np.histogram(data, bins=k)
    print(f'\n2. Наблюдаемые частоты: {observed_freq}')

    # 3. Теоретические частоты
    theor_probs = []
    ML_std = np.sqrt(ML_variance)
    for i in range(len(bin_edges) - 1):
        if i == 0:
            # Левый хвост уходит в минус бесконечность
            prob = norm.cdf(bin_edges[i + 1], ML_mean, ML_std)
        elif i == len(bin_edges) - 2:
            # Правый хвост уходит в плюс бесконечность
            prob = 1 - norm.cdf(bin_edges[i], ML_mean, ML_std)
        else:
            # Внутренние интервалы
            prob = norm.cdf(bin_edges[i + 1], ML_mean, ML_std) - \
                   norm.cdf(bin_edges[i], ML_mean, ML_std)
        theor_probs.append(prob)

    theor_freq = np.array(theor_probs) * n
    print(f'\n3. Теоретические частоты: {theor_freq}')
    print(f'Сумма теоретических вероятностей: {sum(theor_probs):.4f}')  # Должна быть ровно 1.0

    # 4. Объединение малых интервалов
    min_freq = 5
    merged_obs = []
    merged_theor = []
    temp_obs = 0
    temp_theor = 0

    for i in range(len(observed_freq)):
        temp_obs += observed_freq[i]
        temp_theor += theor_freq[i]

        if temp_theor >= min_freq or i == len(observed_freq) - 1:
            if temp_theor > 0:
                merged_obs.append(temp_obs)
                merged_theor.append(temp_theor)
            temp_obs = 0
            temp_theor = 0

    print(f'\n4. После объединения (np_i ≥ {min_freq}):')
    print(f'   Наблюдаемые частоты: {[int(x) for x in merged_obs]}')
    print(f'   Теоретические частоты: {[round(float(x), 4) for x in merged_theor]}')

    df_chi2 = pd.DataFrame({
        'Левая граница': bin_edges[:-1],
        'Правая граница': bin_edges[1:],
        'Наблюдаемая частота': observed_freq,
        'Теоретическая частота': theor_freq,
        'Теоретическая вероятность': theor_freq / n
    })

    df_chi2 = df_chi2.round(4)
    df_chi2.to_excel('table_1_5.xlsx', index=False)

    print('\nТаблица критерия Пирсона')
    print(df_chi2.to_string(index=False))

    # 5. Статистика χ²
    chi2_stat = sum((merged_obs[i] - merged_theor[i]) ** 2 / merged_theor[i]
                    for i in range(len(merged_obs)))
    print(f'\n5. Статистика χ² = {chi2_stat:.6f}')

    # 6. Критическое значение
    df = len(merged_obs) - 1 - 2
    chi2_crit = chi2.ppf(1 - alpha, df)
    print(f'\n6. Степени свободы: ν = {df}')
    print(f'   Критическое значение χ²_{alpha}({df}) = {chi2_crit:.6f}')

    # 7. Вывод
    print(f'\n7. Результат проверки:')
    if chi2_stat > chi2_crit:
        print(f'   χ² = {chi2_stat:.4f} > {chi2_crit:.4f} → H0 ОТВЕРГАЕТСЯ')
        print('   Распределение выборки НЕ соответствует нормальному')
    else:
        print(f'   χ² = {chi2_stat:.4f} ≤ {chi2_crit:.4f} → H0 НЕ ОТВЕРГАЕТСЯ')
        print('   Распределение выборки соответствует нормальному')

def task2():
    print('\n\nЗадание 2')

    a_true = 1.5
    alpha_criterion = 0.001
    gamma = 0.99
    n = 400

    math_exp = 1 / a_true

    U = np.random.uniform(0, 1, n)
    data = -np.log(U) / a_true
    bins = int(np.round(np.log2(n) + 1))  # Формула Стерджеса

    plt.figure(figsize=(8, 5))

    # Строим гистограмму
    plt.hist(data, bins=bins, density=True, edgecolor='black', alpha=0.7, color='skyblue')

    # Оформление
    plt.title('Гистограмма выборки и теоретическая плотность (Пункт 2.1)')
    plt.xlabel('Значения')
    plt.ylabel('Плотность вероятности')

    # Теоретическая плотность
    x_vals = np.linspace(-1, np.max(data) * 1.1, 1000)
    theoretical = expon.pdf(x_vals, loc=0, scale=math_exp)
    plt.plot(x_vals, theoretical, 'r-', linewidth=2, label=f'Теоретическая плотность Exp(a={a_true})')

    # Добавляем легенду
    plt.legend()

    plt.grid(True, linestyle='-', alpha=0.3)

    # Показываем результат
    plt.savefig('task2_hist.png', dpi=300, bbox_inches='tight')
    plt.show()

    hist_counts, hist_edges = np.histogram(data, bins=bins)

    interval_lengths = np.diff(hist_edges)

    relative_freq = hist_counts / n

    heights = hist_counts / (n * interval_lengths)

    midpoints = (hist_edges[:-1] + hist_edges[1:]) / 2

    theoretical_density = expon.pdf(midpoints, scale=math_exp)

    df_hist2 = pd.DataFrame({
        'Левая граница': hist_edges[:-1],
        'Правая граница': hist_edges[1:],
        'Частота ni': hist_counts,
        'Отн. частота wi': relative_freq,
        'Высота гистограммы': heights,
        'Теор. плотность': theoretical_density
    })

    df_hist2 = df_hist2.round(4)
    df_hist2.to_excel('table_2_1.xlsx', index=False)

    print('\nТаблица 2.1')
    print(df_hist2.to_string(index=False))

    print('\nПункт 2.2:')

    sample_mean = np.mean(data)
    a_mm = 1 / sample_mean

    print(f'Параметр a, данный в условии задачи: {a_true:4f}')
    print(f'Точечная оценка параметра  a, найденная методом моментов: {a_mm:4f}')

    print('\nПункт 2.3:')

    exp_val = sample_mean
    var = np.var(data, ddof=1)

    print(f'Точечная оценка математического ожидания: {exp_val:4f}')
    print(f'Точечная оценка дисперсии: {var:4f}')

    print('\nПункт 2.4:')

    alpha_conf = 1 - gamma
    sample_sum = np.sum(data)

    chi2_low = chi2.ppf(alpha_conf / 2, df=(n * 2))
    chi2_high = chi2.ppf(1 - alpha_conf / 2, df=(n * 2))

    mu_lower = 2 * sample_sum / chi2_high
    mu_upper = 2 * sample_sum / chi2_low

    var_lower = mu_lower ** 2
    var_upper = mu_upper ** 2

    print(f"Доверительный интервал для матожидания (γ={gamma}):")
    print(f"[{mu_lower:.4f}, {mu_upper:.4f}]")
    print(f"Точечная оценка: {sample_mean:.4f}")
    print(f"Теоретическое значение: {1 / a_true:.4f}")

    print(f"\nДоверительный интервал для дисперсии (γ={gamma}):")
    print(f"[{var_lower:.4f}, {var_upper:.4f}]")
    print(f"Точечная оценка: {var:.4f}")
    print(f"Теоретическое значение: {1 / a_true ** 2:.4f}")

    print('\nПункт 2.5:')

    k = 15

    a_est = a_mm  # оценка параметра

    # Теоретические вероятности (одинаковые для равновероятных интервалов)
    p_i = 1 / k

    # Границы интервалов (k-1 граница)
    boundaries = [0] + [-np.log(1 - j / k) / a_est for j in range(1, k)] + [np.inf]

    observed, _ = np.histogram(data, bins=boundaries)
    expected = np.full(k, n * p_i)  # массив из k одинаковых значений

    chi2_stat = np.sum((observed - expected) ** 2 / expected)

    df = k - 1 - 1
    alpha_criterion = 0.001  # из условия
    chi2_crit = chi2.ppf(1 - alpha_criterion, df)

    print(f"Количество интервалов: {k}")
    print(f"Длина observed: {len(observed)}")
    print(f"Наблюдаемые частоты: {observed}")
    print(f"Ожидаемые частоты: {expected}")

    df_exp_chi2 = pd.DataFrame({
        'Левая граница': boundaries[:-1],
        'Правая граница': boundaries[1:],
        'Наблюдаемая частота': observed,
        'Теоретическая частота': expected,
        'Теоретическая вероятность': expected / n
    })

    df_exp_chi2 = df_exp_chi2.round(4)
    df_exp_chi2.to_excel('table_2_5.xlsx', index=False)

    print('\nТаблица критерия Пирсона')
    print(df_exp_chi2.to_string(index=False))

    print(f"Статистика χ²: {chi2_stat:.6f}")
    print(f"Критическое значение χ²: {chi2_crit:.6f}")
    print(f"Степени свободы: {df}")

    if chi2_stat > chi2_crit:
        print("Гипотеза отвергается: распределение не показательное")
    else:
        print("Нет оснований отвергать гипотезу о показательном распределении")

def task3():
    print('\n\nЗадание 3:')

    n = 475
    alpha = 0.05
    r_xy = -0.43

    a_x = 0.9
    sigma_squared_x = 12.96

    a_y = 3.2
    sigma_squared_y = 4.5

    std_x = np.sqrt(sigma_squared_x)
    std_y = np.sqrt(sigma_squared_y)

    covariance = r_xy * std_x * std_y

    # Диагональ - это дисперсии (var_x, var_y)
    # Вне диагонали - ковариации
    cov_matrix = np.array([
        [sigma_squared_x, covariance],
        [covariance, sigma_squared_y]
    ])

    print("Матрица ковариации:")
    print(cov_matrix)

    x, y = np.random.multivariate_normal(np.array([a_x, a_y]), cov_matrix, n).T

    print('\nПункт 3.1:')

    mean_x = np.mean(x)
    mean_y = np.mean(y)

    var_x = np.var(x, ddof=1)
    var_y = np.var(y, ddof=1)

    corr = np.corrcoef(x, y)[0][1]

    print(f'Полученное математическое ожидание x: {mean_x}')
    print(f'Полученное математическое ожидание y: {mean_y}')
    print(f'Полученная дисперсия x: {var_x}')
    print(f'Полученная дисперсия y: {var_y}')
    print(f'Полученный коэффициент корреляции между x и y: {corr}')

    print(f'Истинное математическое ожидание X: {a_x}')
    print(f'Истинное математическое ожидание Y: {a_y}')
    print(f'Истинная дисперсия X: {sigma_squared_x}')
    print(f'Истинная дисперсия Y: {sigma_squared_y}')
    print(f'Истинный коэффициент корреляции: {r_xy}')

    df_params = pd.DataFrame({
        'Параметр': [
            'M(X)', 'M(Y)',
            'D(X)', 'D(Y)',
            'r(X,Y)'
        ],
        'Истинное значение': [
            a_x, a_y,
            sigma_squared_x, sigma_squared_y,
            r_xy
        ],
        'Выборочная оценка': [
            mean_x, mean_y,
            var_x, var_y,
            corr
        ]
    })

    df_params = df_params.round(4)
    df_params.to_excel('table_3_1.xlsx', index=False)

    print('\nТаблица оценок параметров')
    print(df_params.to_string(index=False))

    print('\nПункт 3.2:')

    # В методичке требуется использовать критерий хи-квадрат и построить двумерную таблицу.
    K_bins = int(np.round(1 + np.log2(n)))
    L_bins = K_bins

    # Получаем двумерную гистограмму (частоты m_ij)
    hist_2d, x_edges, y_edges = np.histogram2d(x, y, bins=[K_bins, L_bins])

    m_i = np.sum(hist_2d, axis=1)  # Маргинальные частоты по X
    m_j = np.sum(hist_2d, axis=0)  # Маргинальные частоты по Y

    chi2_stat_3 = 0.0

    # Вычисляем статистику хи-квадрат
    for i in range(K_bins):
        for j in range(L_bins):
            if m_i[i] > 0 and m_j[j] > 0:
                expected_ij = (m_i[i] * m_j[j]) / n
                if expected_ij > 0:
                    chi2_stat_3 += ((hist_2d[i, j] - expected_ij) ** 2) / expected_ij

    # Степени свободы
    df_3 = (K_bins - 1) * (L_bins - 1)
    chi2_crit_3 = chi2.ppf(1 - alpha, df_3)

    print(f"Количество интервалов по X: {K_bins}, по Y: {L_bins}")
    print(f"Статистика χ² для проверки независимости: {chi2_stat_3:.6f}")
    print(f"Критическое значение χ²({df_3}): {chi2_crit_3:.6f}")

    if chi2_stat_3 > chi2_crit_3:
        print("Гипотеза о независимости ОТВЕРГАЕТСЯ (X и Y зависимы)")
    else:
        print("Гипотеза о независимости ПРИНИМАЕТСЯ (X и Y независимы)")

    # Справочно оставим r Пирсона, так как он полезен для самопроверки
    r_observed, p_value = pearsonr(x, y)
    print(f"\nСправочно (r Пирсона): r = {r_observed:.4f}, p-value = {p_value:.4f}")

    # Преобразование в DataFrame для сохранения двумерной таблицы частот
    df_2d_hist = pd.DataFrame(hist_2d,
                              index=[f"X: {x_edges[i]:.2f}-{x_edges[i + 1]:.2f}" for i in range(K_bins)],
                              columns=[f"Y: {y_edges[j]:.2f}-{y_edges[j + 1]:.2f}" for j in range(L_bins)])
    df_2d_hist['Сумма (m_i)'] = m_i
    df_2d_hist.loc['Сумма (m_j)'] = np.append(m_j, n)

    df_2d_hist.to_excel('table_3_2_contingency.xlsx')
    print("\nДвумерная таблица частот сохранена в 'table_3_2_contingency.xlsx'")

    print('\nПункт 3.3:')

    s_x = np.sqrt(var_x)
    s_y = np.sqrt(var_y)

    b1 = corr * (s_y / s_x)
    b0 = mean_y - b1 * mean_x

    c1 = corr * (s_x / s_y)
    c0 = mean_x - c1 * mean_y

    print(f"Уравнение регрессии X на Y: X = {c0:.4f} + ({c1:.4f}) * Y")
    print(f"Уравнение регрессии Y на X: Y = {b0:.4f} + ({b1:.4f}) * X")

    # Создаем сетку точек для красивых и ровных линий
    x_lines = np.linspace(np.min(x), np.max(x), 100)

    # Считаем Y для обеих прямых по уравнениям
    y_from_reg_Y_X = b0 + b1 * x_lines
    y_from_reg_X_Y = (x_lines - c0) / c1

    # Рисуем график
    plt.figure(figsize=(10, 6))

    # Рисуем точки выборки
    plt.scatter(x, y, color='lightblue', s=10, alpha=0.8, label='Выборочные значения')

    # Рисуем линии регрессии
    plt.plot(x_lines, y_from_reg_Y_X, color='red', linewidth=2, label='Регрессия Y на X')
    plt.plot(x_lines, y_from_reg_X_Y, color='green', linewidth=2, label='Регрессия X на Y')

    # Отметим точку пересечения
    plt.scatter(mean_x, mean_y, color='black', zorder=5, s=50,
                label=r'Центр совместного распределения ($\bar{x}$, $\bar{y}$)')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Эмпирические линии регрессии и выборочные точки')
    plt.legend()
    plt.grid(True)

    plt.savefig('task3_regression.png', dpi=300, bbox_inches='tight')
    plt.show()