# Лабораторная работа №1

Выполнены пункты 1–8 по классификации и регрессии.

## Требования

Python 3.13. Проверка выполнена в Windows 11.
Версии библиотек указаны в requirements.txt.

## Установка

Откройте терминал в папке проекта:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Запуск

```powershell
.venv\Scripts\python.exe src/p01_preprocessing.py
.venv\Scripts\python.exe src/p02_labels.py
.venv\Scripts\python.exe src/p03_logistic.py
.venv\Scripts\python.exe src/p04_naive_bayes.py
.venv\Scripts\python.exe src/p05_confusion.py
.venv\Scripts\python.exe src/p06_income.py
.venv\Scripts\python.exe src/p07_regression.py
.venv\Scripts\python.exe src/p08_housing.py
```

В Linux/macOS используется `.venv/bin/python`.
Результаты выводятся в терминал и сохраняются в `results/`.
Графики сохраняются в PNG, одномерная модель — в `model.pkl`.

## Структура

- `src/p01_preprocessing.py` — бинаризация, стандартизация, MinMax, L1/L2.
- `src/p02_labels.py` — кодирование и восстановление меток.
- `src/p03_logistic.py` — логистическая регрессия при C=1 и C=100.
- `src/p04_naive_bayes.py` — GaussianNB, метрики и кросс-валидация.
- `src/p05_confusion.py` — матрица ошибок и classification report.
- `src/p06_income.py` — SVM и три прогноза дохода.
- `src/p07_regression.py` — одномерная и многомерная регрессия, полином степени 10.
- `src/p08_housing.py` — пять вариантов SVR для трёх объектов.
- `src/common.py` — пути, метрики и графики.
- `data/` — исходные учебные данные.
- `results/` — графики, текстовые результаты и сохранённая модель.

Boston Housing взят из [архива scikit-learn](https://raw.githubusercontent.com/scikit-learn/scikit-learn/1.1.3/sklearn/datasets/data/boston_house_prices.csv).
Остальные данные предоставлены вместе с лабораторной работой.
