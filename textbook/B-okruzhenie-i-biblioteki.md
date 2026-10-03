# Приложение B. Окружение, библиотеки и указатель файлов репозитория

## B.1. Базовое окружение

Репозиторий `ai-for-trading` содержит `pyproject.toml` (Poetry) с современными
версиями:

```toml
python = "^3.12"
pandas = "^2.2.1"
numpy = "^1.26.4"
scipy = "^1.12.0"
statsmodels = "^0.14.1"
patsy = "^0.5.6"
matplotlib = "^3.8.3"
nltk = "^3.8.1"
torch = "^2.2.1"
tqdm = "^4.66.2"
jupyter = "^1.0.0"
```

Этого достаточно для глав 1–2, 6–7, 9 (кроме загрузки данных Barra, которые
распространялись только в среде курса). Для остальных глав нужны дополнительные
пакеты:

| Глава | Пакеты сверх базовых | Зачем |
|---|---|---|
| 3 | `cvxpy` | выпуклая оптимизация портфеля |
| 4 | `scikit-learn` | PCA |
| 5 | `zipline-reloaded`, `alphalens-reloaded`, `cvxpy`, `graphviz` | Pipeline, оценка факторов, оптимизация |
| 6 | `beautifulsoup4`, `lxml`, `requests`, `scikit-learn`, `alphalens-reloaded`, `ratelimit` | парсинг 10-K, BoW/TF-IDF, оценка альф |
| 7 | `torch` (есть в базе), GPU желательно | обучение LSTM на 1,5 млн сообщений |
| 8 | `zipline-reloaded`, `alphalens-reloaded`, `scikit-learn`, `shap`, `graphviz`, `xgboost` | факторы, случайный лес, SHAP, упражнение на оверфиттинг |
| 9 | `patsy`, `statsmodels` (в базе), `scipy` | формулы patsy, OLS, L-BFGS |

Установка (pip):

```bash
pip install cvxpy scikit-learn beautifulsoup4 lxml requests shap xgboost graphviz
pip install zipline-reloaded alphalens-reloaded
```

⚠️ `zipline-reloaded` требует C-компилятор и TA-Lib не нужен; на Windows проще ставить
через conda-forge. Оригинальные `zipline==1.2/1.3` и `alphalens==0.3.2` из
`requirements.txt` проектов на Python ≥ 3.8 не собираются.

## B.2. Данные

Проекты курса использовали закрытые наборы данных Udacity, которых **нет в репозитории**:

* `eod-quotemedia.csv` — дневные OHLCV + дивиденды по акциям S&P 500 (QuoteMedia),
  2013–2017; в проектах 4, 5, 7 — тот же набор, упакованный в Zipline-бандл
  (`data/project_4_eod` и т. п.);
* `project_4_sector/data.npy` — коды секторов;
* `loughran_mcdonald_master_dic_2016.csv` — словарь тональности (публично доступен на
  сайте авторов);
* `project_6_stocktwits/twits.json` — 1,5 млн размеченных сообщений StockTwits;
* `project_8_barra/*.pickle` — факторные экспозиции и ковариации Barra за 2003–2008.

Для самостоятельного воспроизведения: бесплатные дневные данные можно взять через
`yfinance`, `pandas-datareader` или Nasdaq Data Link; для Zipline-бандла пригоден
`csvdir`-ингест (как в проектах: `bundles.csvdir.csvdir_equities(['daily'], name)`).
Полноценной бесплатной замены Barra нет — глава 9 объясняет, как построить
собственную риск-модель из PCA (глава 4) и использовать её вместо Barra.

## B.3. Что менялось при модернизации кода

Ноутбуки написаны под Python 3.6 / pandas 0.18–0.24 / numpy 1.13–1.16 /
scikit-learn 0.19. В книге код приведён к pandas 2.x и актуальным API. Сводка замен:

| Было (в ноутбуках) | Стало (в книге) | Где встречается |
|---|---|---|
| `np.int`, `np.float` | `int`, `float` | пр. 1, 2, 6 |
| `.resample('M')` | `.resample('ME')` (month-end) | пр. 1 |
| `df.append(other)` | `pd.concat([df, other])` | квиз m7 |
| `df.any(1)` | `df.any(axis=1)` | пр. 5 |
| `df.iteritems()` | `df.items()` | пр. 4 |
| `sklearn.metrics.jaccard_similarity_score` | `jaccard_score` (поэлементно по булевым векторам) | пр. 5 |
| `BaggingClassifier(base_estimator=...)` | `BaggingClassifier(estimator=...)` | пр. 7 |
| `pd.Timestamp(..., offset='C')` | `pd.Timestamp(...)` | пр. 4, 7 |
| `alpha.T.values * weights` (cvxpy) | `alpha.values.flatten() @ weights` | пр. 4 |
| `-alpha.T*x` (cvxpy) | `-alpha @ x` | квиз Regularization |
| `pd.DatetimeIndex(...).year` + `to_datetime(format='%Y')` | без изменений, но см. `dt.year` | пр. 5 |
| `tqdm_notebook` | `tqdm.auto.tqdm` | квиз m7 |
| `zipline.utils.calendars.get_calendar` | `zipline.utils.calendar_utils.get_calendar` (zipline-reloaded) | пр. 4, 7 |
| `df.index.levels[0]` для выбора дней | `df.index.get_level_values(0).unique()` (levels может содержать неиспользуемые значения) | пр. 7 |
| `pd.date_range(freq='BM' / 'BQ')` | `'BME'` / `'BQE'` | пр. 7 |
| `VotingClassifier(estimators, voting)` | `VotingClassifier(estimators, voting=voting)` — только по имени | пр. 7 |
| `groupby(['ticker'])` | `groupby('ticker')` (со списком ключ группы — кортеж) | пр. 2 |
| `df.stack()` | `df.stack(future_stack=True).dropna()` (новая реализация не удаляет NaN) | пр. 2 |
| `cvx.quad_form(x, P)` на выборочной ковариации | `cvx.quad_form(x, cvx.psd_wrap(P))` (численная проверка PSD) | пр. 3 |
| `sum(x)` в cvxpy | `cvx.sum(x)` | пр. 3, 4 |
| `scipy.optimize.fmin_l_bfgs_b` | `scipy.optimize.minimize(..., method='L-BFGS-B')` (старый интерфейс работает) | пр. 8 |
| `xgb.train({'silent': 1})` | `{'verbosity': 0}` | квиз m8 |
| `'\w+'` без префикса `r` | `r'\w+'` (Python 3.12: `SyntaxWarning`) | пр. 5, 6 |

Отдельная оговорка про ошибку в оригинале: в проекте 8 строка
`df.loc[df['SpecRisk'] == 0]['SpecRisk'] = median(...)` — это присваивание в копию
(chained assignment), оно **ничего не меняет**. Корректно:
`df.loc[df['SpecRisk'] == 0, 'SpecRisk'] = df['SpecRisk'].median()`. В главе 9 это
исправлено и отмечено.

## B.4. Указатель файлов репозитория по главам

| Глава | Проект | Ключевые квизы и заметки |
|---|---|---|
| 1 | `Projects/1-Trading-with-momentum/project_1_starter.ipynb` | `Quiz/m1_quant_basics/l2_stock_prices/stock_data.ipynb`, `l3_market_mechanics/resample_data.ipynb`, `l5_stock_returns/calculate_returns.ipynb`, `l6_momentum_trading/dtype.ipynb`, `l6_momentum_trading/top_and_bottom_performing.ipynb`; `Notes/1-Trading-with-momentum/stock-market-data.md` |
| 2 | `Projects/2-Breakout-strategy/project_2_starter.ipynb` (+ `README.md`, `-zh.ipynb`) | `Quiz/m2_advanced_quants/l3_regression/test_normality.ipynb`, `l5_volatility/rolling_windows.ipynb`, `l6_pairs_trading_and_mean_reversion/pairs_candidates.ipynb`; `Side-projects/1D-Kalman-filter.ipynb`, `Side-projects/Hypthesis-testing.ipynb`; `Notes/2-Breakout-strategy/*.md` |
| 3 | `Projects/3-Smart-Beta/project_3_starter.ipynb` | `Quiz/m3_funds_etfs_portfolio_optimization/l1_stocks_indices_funds/cumsum_and_cumprod.ipynb`, `l3_portfolio_risk_and_return/m3l3_covariance.ipynb`, `l4_portfolio_optimization/m3l4_cvxpy_basic.ipynb`, `m3l4_cvxpy_advanced.ipynb`; `Notes/3-Porfolio-optimization/*.md` |
| 4 | `Projects/4-Multi-factor-Model/project_4_starter.pdf` (часть «Statistical Risk Model») | `Quiz/m4_multifactor_models/m4l2/*` (historical_variance, factor_model_asset_return, factor_model_portfolio_return, covariance_matrix_assets, portfolio_variance, pca_basics, pca_factor_model), `PCA_Core/`, `PCA Toy Problem/`, `PCA.ipynb`, `PCA_3D.ipynb`; `Notes/4-Alpha-Research-and-Factor-Modeling/*.md` |
| 5 | `Projects/4-Multi-factor-Model/project_4_starter.pdf` (части «Alpha Factors», «Optimal Portfolio») | `Quiz/m4_multifactor_models/m4l1/zipline_coding_exercises.ipynb`, `Zipline-Pipeline/`, `m4l3/*` (sector_neutral, rank, zscore, smoothing, quantiles, factor_returns, sharpe_ratio, turnover, rank_ic, transfer_coefficient, regression_against_time, overnight_returns, clean_forward_returns), `Advanced_Opt/Advanced_Opt.ipynb`, `Regularization.ipynb` |
| 6 | `Projects/5-Intro-NLP/project_5_starter.ipynb` | `Quiz/m5_financial_statements/*` (text_processing, regexes, applying_regexes_10ks, beautifulSoup, requests_library, Readability_Exercises, Bag_of_Word_Exercises, process_tweets, download10k.py); `Notes/5-Intro-NLP/README.md` |
| 7 | `Projects/6-Sentiment-Analysis/project_6_starter.ipynb` | `Quiz/m6/1.Tensors-in-PyTorch.ipynb` (+ PDF 2–8); `Notes/6-Sentiment-analysis-with-neutral-networks/README.md` |
| 8 | `Projects/7-Combining-alphas/project_7_starter.ipynb` | `Quiz/m7/titanic_survival_exploration.ipynb`, `spam_rf.ipynb`, `Learning Curve.ipynb`, `dependent_labels_solution.ipynb`, `m7l3/feature_engineering.ipynb`, `m7l6/*` (sklearn_feature_importance, calculate_shap, tree_shap, rank_features); `Notes/7-Combining-signals-for-enhanced-alpha/*.md` |
| 9 | `Projects/8-Backtesting/project_8_starter.ipynb` | `Quiz/m8/optimization_with_tcosts.ipynb`, `Quiz/m8/overfitting_exercise.ipynb`; `Notes/8-Backtesting/*.md` |

Файлы `project_helper.py`, `project_tests.py`, `helper.py`, `tests.py` в папках
проектов — вспомогательный код курса (графики, юнит-тесты); книга на них не
опирается, но упоминает, когда функция проекта вызывает их.
