# Глава 8. Машинное обучение для комбинирования альфа-факторов

> Чему научитесь: превращать несколько альфа-факторов и признаков рыночного режима в один
> сигнал («AI-альфу») с помощью случайного леса; честно разбивать финансовые данные во
> времени; распознавать и лечить проблему перекрывающихся меток; читать важность признаков
> (Джини и SHAP). Модуль M7 *Combining Signals for Enhanced Alpha*, проект
> 📁 `Projects/7-Combining-alphas/project_7_starter.ipynb`.

## 8.1. Мотивация и гипотеза

В главе 5 мы построили три альфа-фактора — годовой моментум, пятидневный возврат к
среднему и «ночной сентимент» — и оценили каждый по отдельности в Alphalens. Самый простой
способ объединить их — усреднить. Усреднение линейно: оно предполагает, что вклад каждого
фактора одинаков всегда и для всех акций.

Опыт говорит обратное. Эффективность альфы зависит от **режима рынка**: моментум работает
на спокойном трендовом рынке и ломается при разворотах; возврат к среднему, наоборот, любит
высокую волатильность и дисперсию доходностей. Есть и другие нелинейности: фактор может
работать только в части секторов, для ликвидных бумаг, в декабре. Перебирать такие правила
руками — долго и чревато подгонкой.

Гипотеза главы: модель машинного обучения, которой на вход подаются сами альфа-факторы
**плюс** описание контекста (волатильность и ликвидность акции, волатильность и дисперсия
рынка, сектор, календарь), научится *взвешивать факторы в зависимости от режима* и даст
сигнал лучше каждого фактора в отдельности — **AI-альфу** (AI alpha). Предупреждение из
введения здесь особенно актуально: модель с точностью заметно выше 50 % или Шарпом выше 4
на дневных данных скорее всего обманывает нас — утечкой будущего, перекрывающимися метками
или подгонкой.

## 8.2. Теория

### 8.2.1. Деревья решений: энтропия, прирост информации, Джини

Дерево решений (decision tree) последовательно делит выборку пороговыми правилами
«признак $x_j \le t$», пока в листьях не окажутся по возможности однородные группы. Чтобы
выбрать лучший сплит, нужна мера «беспорядка» в узле.

**Энтропия** (entropy) узла с долями классов $p_1,\dots,p_C$:

$$
H = -\sum_{c=1}^{C} p_c \log_2 p_c ,
$$

где $p_c$ — доля наблюдений класса $c$ в узле. Интуиция из конспекта курса: энтропия
измеряет «число способов» расставить объекты, не меняя состава. Узел из одного класса
имеет $H = 0$; узел 50/50 — $H = 1$ бит.

**Прирост информации** (information gain) от сплита узла $P$ на детей $L$ и $R$:

$$
IG = H(P) - \frac{n_L}{n_P} H(L) - \frac{n_R}{n_P} H(R),
$$

где $n_P, n_L, n_R$ — число наблюдений в родителе и детях. Дерево жадно выбирает признак и
порог с максимальным $IG$ (в `scikit-learn` — `criterion='entropy'`).

Альтернатива — **загрязнённость Джини** (Gini impurity): $G = \sum_c p_c (1 - p_c) = 1 - \sum_c p_c^2$.
Она ведёт себя так же (0 для чистого узла, 0,5 при 50/50 для двух классов), но считается
без логарифма; это критерий по умолчанию в `scikit-learn`, он же используется во встроенной
важности признаков (§ 8.4.1).

### 8.2.2. Переобучение дерева и случайный лес

Глубокое дерево способно запомнить обучающую выборку — каждый лист по одному наблюдению —
и плохо работать на новых данных: это **оверфиттинг (переобучение)** (overfitting).
Ограничение глубины (`max_depth`) или минимального размера листа (`min_samples_leaf`)
помогает, но дерево остаётся нестабильным: небольшое изменение данных меняет первый сплит,
а за ним всю структуру.

**Случайный лес** (random forest) лечит нестабильность усреднением. Два источника
случайности:

1. **Бэггинг** (bagging): каждое дерево обучается на бутстреп-выборке — $n$ наблюдений,
   взятых с возвращением из $n$ исходных. В неё попадает в среднем
   $1 - (1 - 1/n)^n \approx 1 - e^{-1} \approx 2/3$ уникальных наблюдений.
2. **Случайные подмножества признаков**: в каждом узле рассматривается случайная часть
   признаков (`max_features`, по умолчанию $\sqrt{M}$). Это декоррелирует деревья: сильный
   признак не захватывает корень во всех деревьях сразу.

Дисперсия ансамбля из $T$ деревьев с попарной корреляцией $\rho$ и дисперсией $\sigma^2$
падает примерно до $\rho\sigma^2 + \frac{1-\rho}{T}\sigma^2$. Отсюда ключевое следствие:
**увеличение числа деревьев бесполезно, если деревья сильно коррелированы** — именно это
происходит при перекрывающихся метках (§ 8.2.5).

**OOB-оценка** (out-of-bag score). Из конспекта `Random_forest.md`: каждое дерево видит
около 2/3 наблюдений, оставшаяся треть для него «вне мешка». Для каждого наблюдения можно
собрать предсказания только тех деревьев, которые его не видели (примерно треть леса), и
сравнить с истинной меткой. Усреднив по выборке, получаем оценку точности вне выборки *без
отдельного валидационного набора* (`oob_score=True`, атрибут `oob_score_`). OOB-оценка
честна **только при независимых наблюдениях** — для финансовых рядов это нарушено.

### 8.2.3. Классификация против регрессии: зачем квантизовать цель

Казалось бы, естественно предсказывать саму 5-дневную доходность регрессией. Но в курсе
цель **квантизируется**: `Returns(window_length=5).quantiles(2)` даёт бинарную метку
«доходность акции выше (1) или ниже (0) медианы кросс-секции в этот день». Причины (квиз
`m7l3/feature_engineering`): метка становится **рыночно-нейтральной** (мы предсказываем не
«вырастет ли акция», а «обгонит ли половину универсума», что и нужно long/short-портфелю);
квантизация **нормирует меняющиеся во времени волатильность и дисперсию** (2 % в спокойный
год и в кризис — разные события, «выше медианы» — одно и то же); метка **устойчива к смене
режимов и выбросам**. Цена — потеря информации о величине доходности. Для диагностики
проект строит ещё `quantiles(25)`.

### 8.2.4. Признаки

Все признаки считаются конвейером Zipline Pipeline (глава 5) на универсуме 500 самых
ликвидных акций (`AverageDollarVolume(window_length=120).top(500)`) за три года до
2016-01-05.

**Альфа-факторы из главы 5** (после `rank().zscore()`): `Momentum_1YR` — доходность за
252 дня, центрированная по сектору; `Mean_Reversion_Sector_Neutral_Smoothed` — минус
5-дневная доходность, центрированная по сектору, сглаженная SMA(20);
`Overnight_Sentiment_Smoothed` — сумма ночных доходностей за 10 дней, сглаженная SMA(10).

**«Универсальные» квант-признаки** описывают саму акцию: `AnnualizedVolatility` за 20 и
120 дней, `AverageDollarVolume` за 20 и 120 дней (всё — `rank().zscore()`), сектор,
развёрнутый в one-hot-столбцы `sector_<имя>`.

**Признаки рыночного режима** (regime features) одинаковы для всех акций в данный день:

* **дисперсия рынка** (`MarketDispersion`) — стандартное отклонение кросс-секции дневных
  доходностей $d_t = \sqrt{\frac{1}{N}\sum_i (r_{i,t} - \bar r_t)^2}$, где $\bar r_t$ —
  средняя доходность универсума; затем SMA за 20 и 120 дней;
* **волатильность рынка** (`MarketVolatility`) — $\sigma_m = \sqrt{260 \cdot \frac{1}{T} \sum_t (r_{m,t} - \bar r_m)^2}$,
  где $r_{m,t} = \frac{1}{N}\sum_i r_{i,t}$ — равновзвешенная доходность «рынка», $T$ — окно
  (20 или 120 дней). В проекте множитель 260, в квизе — 252; для дерева разница не важна:
  сплит делается по порогу, а не по абсолютному значению.

Оба класса наследуют `CustomFactor` с `window_safe = True` — флагом Zipline, разрешающим
подавать выход фактора на вход другому (`SimpleMovingAverage`).

**Календарные признаки**: январь, декабрь, день недели, квартал, первый/последний рабочий
день месяца и квартала — чтобы дерево могло поймать календарные аномалии («эффект января»,
ребалансировки фондов в конце квартала).

### 8.2.5. Метки, их сдвиг и проблема IID

Метка для строки (день $t$, акция $i$) — квантиль 5-дневной доходности, **сдвинутый на 5
дней назад внутри каждой акции**: `groupby(level=1)['return_5d'].shift(-5)`. Признаки дня
$t$ сопоставляются с доходностью за $t \to t+5$. Сдвиг обязателен по группе актива: иначе
`shift` перемешает соседние акции внутри одного дня.

Большинство ML-моделей предполагают, что наблюдения **независимы и одинаково
распределены** (IID). У нас это не так: в один день 500 строк делят одни признаки режима и
один рынок, а метки соседних дней **перекрываются** (overlapping labels) — 5-дневные
доходности за $t \to t+5$ и $t+1 \to t+6$ имеют четыре общих дня. Из `overlap.md`: при
перекрывающихся метках бутстреп-выборки разных деревьев содержат почти одну информацию,
деревья получаются похожими, лес перестаёт снижать дисперсию, а OOB-оценка становится
оптимистичной (наблюдение «вне мешка» присутствует в мешке в виде соседнего дня). Решения
курса: **подвыборка**, **меньшие мешки** (`max_samples`), **ансамбль на непересекающихся
подвыборках**.

### 8.2.6. Разбиение во времени и выбор гиперпараметров

Случайный `train_test_split` для рядов недопустим: соседние дни попадут в разные наборы, и
валидация увидит почти копии обучающих строк. Разбиение делается **строго по датам**:
первые 60 % дней — обучение, следующие 20 % — валидация, последние 20 % — тест; день
целиком принадлежит одному набору. Из `Evaluation.md`: k-fold кросс-валидация для рядов
тоже идёт только «вперёд во времени»; кривые обучения диагностируют недо- и переобучение;
при несбалансированных классах вместо точности смотрят точность/полноту и ошибки I/II рода.

Ключевой гиперпараметр — `min_samples_leaf`. Логика курса: в универсуме ~500 акций, и
чтобы лист давал представительное предсказание, в нём должен быть хотя бы один полный день
— 500 наблюдений; обычно это умножают на 2, 3, 5 или 10. При малых значениях модель на
обучении «слишком хороша», а практическое правило курса: **коэффициент Шарпа выше 4 на
дневных данных — признак переобучения**. Поэтому выбрано $500 \times 10 = 5000$. Второе
правило: менять гиперпараметры **на порядок, а не на единицы** (10, 20, 100, а не 10, 11,
12) — иначе переобучимся уже на валидацию.

## 8.3. Разбор проекта шаг за шагом

Ниже — функции проекта 7 с модернизированным кодом (pandas 2.x, scikit-learn ≥ 1.2;
приложение B §B.3). Вспомогательные `project_helper.*` (движок Pipeline, графики, обёртки
Alphalens) и юнит-тесты `project_tests.*` не разбираются.

### 8.3.1. Конвейер признаков и цели

Импорт календаря в `zipline-reloaded`: `from zipline.utils.calendar_utils import
get_calendar`. Факторы `momentum_1yr`, `mean_reversion_5day_sector_neutral_smoothed`,
`overnight_sentiment_smoothed` — из главы 5 без изменений.

```python
from zipline.pipeline import Pipeline
from zipline.pipeline.factors import (CustomFactor, DailyReturns, Returns, SimpleMovingAverage,
                                      AnnualizedVolatility, AverageDollarVolume)

universe = AverageDollarVolume(window_length=120).top(500)
sector = project_helper.Sector()
pipeline = Pipeline(screen=universe)

pipeline.add(momentum_1yr(252, universe, sector), 'Momentum_1YR')
pipeline.add(mean_reversion_5day_sector_neutral_smoothed(20, universe, sector),
             'Mean_Reversion_Sector_Neutral_Smoothed')
pipeline.add(overnight_sentiment_smoothed(2, 10, universe), 'Overnight_Sentiment_Smoothed')

# «универсальные» признаки акции
pipeline.add(AnnualizedVolatility(window_length=20, mask=universe).rank().zscore(), 'volatility_20d')
pipeline.add(AnnualizedVolatility(window_length=120, mask=universe).rank().zscore(), 'volatility_120d')
pipeline.add(AverageDollarVolume(window_length=20, mask=universe).rank().zscore(), 'adv_20d')
pipeline.add(AverageDollarVolume(window_length=120, mask=universe).rank().zscore(), 'adv_120d')
pipeline.add(sector, 'sector_code')


class MarketDispersion(CustomFactor):
    inputs = [DailyReturns()]
    window_length = 1
    window_safe = True

    def compute(self, today, assets, out, returns):
        # returns: дни в строках, акции в столбцах
        out[:] = np.sqrt(np.nanmean((returns - np.nanmean(returns)) ** 2))


class MarketVolatility(CustomFactor):
    inputs = [DailyReturns()]
    window_length = 1
    window_safe = True

    def compute(self, today, assets, out, returns):
        mkt_returns = np.nanmean(returns, axis=1)      # одна доходность рынка на день
        out[:] = np.sqrt(260. * np.nanmean((mkt_returns - np.nanmean(mkt_returns)) ** 2))


pipeline.add(SimpleMovingAverage(inputs=[MarketDispersion(mask=universe)], window_length=20), 'dispersion_20d')
pipeline.add(SimpleMovingAverage(inputs=[MarketDispersion(mask=universe)], window_length=120), 'dispersion_120d')
pipeline.add(MarketVolatility(window_length=20), 'market_vol_20d')
pipeline.add(MarketVolatility(window_length=120), 'market_vol_120d')

# цель: 5-дневная доходность, квантизованная в 2 и 25 корзин
pipeline.add(Returns(window_length=5, mask=universe).quantiles(2), 'return_5d')
pipeline.add(Returns(window_length=5, mask=universe).quantiles(25), 'return_5d_p')

all_factors = engine.run_pipeline(pipeline, factor_start_date, universe_end_date)
```

`window_length = 1` у `MarketDispersion`: на вход уже подаются доходности, для дисперсии
одного дня достаточно одной строки, а усреднение по 20 и 120 дням делает внешний
`SimpleMovingAverage`. У `MarketVolatility` окно задаётся при создании, потому что
стандартное отклонение ряда рыночных доходностей считается внутри `compute`.

Календарные признаки добавляются к готовому `DataFrame`. Частоты `'BM'` и `'BQ'` в
pandas 2.2 устарели так же, как `'M'` → `'ME'`: теперь `'BME'` и `'BQE'` (`'BMS'`, `'BQS'`
не изменились). Опечатку `is_Janaury` из проекта исправляем.

```python
dates = all_factors.index.get_level_values(0)

all_factors['is_January'] = dates.month == 1
all_factors['is_December'] = dates.month == 12
all_factors['weekday'] = dates.weekday
all_factors['quarter'] = dates.quarter
all_factors['month_end'] = dates.isin(pd.date_range(factor_start_date, universe_end_date, freq='BME'))
all_factors['month_start'] = dates.isin(pd.date_range(factor_start_date, universe_end_date, freq='BMS'))
all_factors['qtr_end'] = dates.isin(pd.date_range(factor_start_date, universe_end_date, freq='BQE'))
all_factors['qtr_start'] = dates.isin(pd.date_range(factor_start_date, universe_end_date, freq='BQS'))

sector_lookup = pd.read_csv('../../data/project_7_sector/labels.csv', index_col='Sector_i')['Sector'].to_dict()
sector_columns = [f'sector_{name}' for name in sector_lookup.values()]     # one-hot сектора
for sector_i, sector_name in sector_lookup.items():
    all_factors[f'sector_{sector_name}'] = all_factors['sector_code'] == sector_i

# сдвиг цели: признаки дня t ↔ доходность за t → t+5, строго внутри каждой акции
all_factors['target'] = all_factors.groupby(level=1)['return_5d'].shift(-5)
```

### 8.3.2. Проверка IID: скользящая автокорреляция меток

Чтобы увидеть перекрытие меток, проект строит метки, сдвинутые на 1–4 дня относительно
`target`, и для каждого дня считает корреляцию Спирмена между `target` и сдвинутой меткой
по кросс-секции акций.

```python
from scipy.stats import spearmanr

def sp(group, col1_name, col2_name):
    return spearmanr(group[col1_name], group[col2_name])[0]

for k in range(1, 5):
    all_factors[f'target_{k}'] = all_factors.groupby(level=1)['return_5d'].shift(-(5 - k))

g = all_factors.dropna().groupby(level=0)
for k in range(1, 5):
    g.apply(sp, 'target', f'target_{k}').plot(ylim=(-1, 1), label=f'target_{k}')
plt.legend()
plt.title('Скользящая автокорреляция меток, сдвинутых на 1–4 дня')
```

Индексация: `target` — окно, заканчивающееся в $t+5$; `target_k` = `shift(-(5-k))` — окно
с концом в $t+5-k$. Окна перекрываются на $5-k$ дней: `target_1` — на четыре, `target_4` —
на один. Ответ автора репозитория (по существу): **автокорреляция положительна для сдвигов
1–3 и близка к нулю для 4**; будущая 5-дневная доходность тем сильнее связана с соседней,
чем ближе их окна. Это эмпирическое доказательство того, что метки не независимы, а степень
зависимости задаётся горизонтом метки.

### 8.3.3. `train_valid_test_split`: разбиение по датам

```python
import math


def train_valid_test_split(all_x, all_y, train_size, valid_size, test_size):
    """Разбить признаки и метки на train/valid/test по датам, не разрывая день."""
    assert 0 <= train_size <= 1.0 and 0 <= valid_size <= 1.0 and 0 <= test_size <= 1.0
    assert math.isclose(train_size + valid_size + test_size, 1.0)

    # уникальные даты, реально присутствующие в индексе (а не атрибут index.levels —
    # в нём остаются значения, уже удалённые dropna)
    days = all_x.index.get_level_values(0).unique().sort_values()
    n_days = len(days)
    train_cutoff = int(train_size * n_days)
    valid_cutoff = int((train_size + valid_size) * n_days)

    d_train, d_valid, d_test = days[:train_cutoff], days[train_cutoff:valid_cutoff], days[valid_cutoff:]

    x_train, y_train = all_x.loc[d_train[0]:d_train[-1]], all_y.loc[d_train[0]:d_train[-1]]
    x_valid, y_valid = all_x.loc[d_valid[0]:d_valid[-1]], all_y.loc[d_valid[0]:d_valid[-1]]
    x_test, y_test = all_x.loc[d_test[0]:d_test[-1]], all_y.loc[d_test[0]:d_test[-1]]
    return x_train, x_valid, x_test, y_train, y_valid, y_test
```

Две модернизации. Уникальные даты берутся через `get_level_values(0).unique()`, а не через
атрибут `index.levels`: после `dropna()` уровни мультииндекса не пересчитываются, и в них
остаются даты, которых в данных уже нет (B §B.3). Проверка суммы долей — через
`math.isclose`: точное `== 1.0` ломается уже на 0.7 + 0.2 + 0.1. Срез `.loc[первая:последняя]`
по первому уровню отсортированного мультииндекса захватывает все акции за эти дни.

```python
features = [
    'Mean_Reversion_Sector_Neutral_Smoothed', 'Momentum_1YR', 'Overnight_Sentiment_Smoothed',
    'adv_120d', 'adv_20d', 'dispersion_120d', 'dispersion_20d',
    'market_vol_120d', 'market_vol_20d', 'volatility_20d',
    'is_January', 'is_December', 'weekday', 'month_end', 'month_start', 'qtr_end', 'qtr_start',
] + sector_columns

temp = all_factors.dropna().copy()
X, y = temp[features], temp['target']
X_train, X_valid, X_test, y_train, y_valid, y_test = train_valid_test_split(X, y, 0.6, 0.2, 0.2)
```

### 8.3.4. Одно дерево и первый лес

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

clf_random_state = 0
simple_clf = DecisionTreeClassifier(max_depth=3, criterion='entropy', random_state=clf_random_state)
simple_clf.fit(X_train, y_train)
project_helper.rank_features_by_importance(simple_clf.feature_importances_, features)

n_days, n_stocks = 10, 500
clf_parameters = {
    'criterion': 'entropy',
    'min_samples_leaf': n_stocks * n_days,   # 5000
    'oob_score': True,
    'n_jobs': -1,
    'random_state': clf_random_state,
}
n_trees_l = [50, 100, 250, 500, 1000]

train_score, valid_score, oob_score, feature_importances = [], [], [], []
for n_trees in tqdm(n_trees_l, desc='Training Models', unit='Model'):
    clf = RandomForestClassifier(n_trees, **clf_parameters)
    clf.fit(X_train, y_train)
    train_score += [clf.score(X_train, y_train.values)]
    valid_score += [clf.score(X_valid, y_valid.values)]
    oob_score += [clf.oob_score_]
    feature_importances += [clf.feature_importances_]
```

### 8.3.5. `show_sample_results`: AI-альфа как фактор в Alphalens

`predict_proba` возвращает матрицу $[p_0, p_1]$; скалярное произведение с $[-1, 1]$ даёт
$p_1 - p_0 = 2p_1 - 1 \in [-1, 1]$ — положительный сигнал для акций, которым модель
предсказывает «выше медианы». Столбец добавляется к исходным факторам и оценивается в
Alphalens *так же*, как факторы в главе 5: факторные доходности, коэффициент Шарпа,
автокорреляция рангов (FRA).

```python
import alphalens as al   # alphalens-reloaded, API без изменений

all_assets = all_factors.index.get_level_values(1).unique().tolist()
all_pricing = get_pricing(data_portal, trading_calendar, all_assets, factor_start_date, universe_end_date)


def show_sample_results(data, samples, classifier, factors, pricing=all_pricing):
    # AI-альфа: p(выше медианы) − p(ниже медианы)
    alpha_score = classifier.predict_proba(samples).dot(np.array([-1, 1]))

    factors_with_alpha = data.loc[samples.index].copy()
    factors_with_alpha['AI_ALPHA'] = alpha_score

    factor_data = project_helper.build_factor_data(factors_with_alpha[factors + ['AI_ALPHA']], pricing)
    factor_returns = project_helper.get_factor_returns(factor_data)
    sharpe_ratio = project_helper.sharpe_ratio(factor_returns)

    print('             Sharpe Ratios')
    print(sharpe_ratio.round(2))
    project_helper.plot_factor_returns(factor_returns)
    project_helper.plot_factor_rank_autocorrelation(factor_data)
factor_names = ['Mean_Reversion_Sector_Neutral_Smoothed', 'Momentum_1YR',       # базовые факторы
                'Overnight_Sentiment_Smoothed', 'adv_120d', 'volatility_20d']   # для сравнения
```

В `get_pricing` проекта у `pd.Timestamp(...)` убран аргумент `offset` (удалён в pandas
2.x); список активов берётся через `get_level_values(1).unique()`.

### 8.3.6. `non_overlapping_samples`: каждый пятый день

Первое средство от перекрывающихся меток — оставить только дни, метки которых не
пересекаются: при 5-дневной метке это каждый пятый день (`n_skip_samples=4`).

```python
def non_overlapping_samples(x, y, n_skip_samples, start_i=0):
    """Оставить дни start_i, start_i + (n_skip_samples+1), ... — их метки не перекрываются."""
    assert len(x.shape) == 2
    assert len(y.shape) == 1

    days = x.index.get_level_values(0).unique().sort_values()
    keep_days = days[start_i::n_skip_samples + 1]

    non_overlapping_x = x.loc[x.index.get_level_values(0).isin(keep_days)]
    non_overlapping_y = y.loc[y.index.get_level_values(0).isin(keep_days)]
    return non_overlapping_x, non_overlapping_y


for n_trees in tqdm(n_trees_l, desc='Training Models', unit='Model'):
    clf = RandomForestClassifier(n_trees, **clf_parameters)
    clf.fit(*non_overlapping_samples(X_train, y_train, 4))
    ...
```

Параметр `start_i` понадобится в § 8.3.8: сдвигая начало на 0, 1, 2, 3, 4, получаем пять
непересекающихся подвыборок, вместе покрывающих все данные. Недостаток очевиден: **мы
выбрасываем 80 % данных** — обучение ускорилось в десять раз (1 мин против 9), результаты на
валидации стали правдоподобнее, но информация потеряна.

### 8.3.7. `bagging_classifier`: меньшие мешки

Второй способ — оставить все данные, но заставить каждое дерево видеть небольшую их долю
(`max_samples=0.2` — порядка «средней уникальности» метки при пятикратном перекрытии).
`RandomForestClassifier` старых версий не принимал `max_samples`, поэтому проект собирает
лес из `BaggingClassifier` над `DecisionTreeClassifier`. Базовая модель передаётся параметром
`estimator` (старое имя устарело в scikit-learn 1.2 и удалено в 1.4, B §B.3).

```python
from sklearn.ensemble import BaggingClassifier


def bagging_classifier(n_estimators, max_samples, max_features, parameters):
    """Собрать бэггинг деревьев с ограничением доли выборки на дерево."""
    required = {'criterion', 'min_samples_leaf', 'oob_score', 'n_jobs', 'random_state'}
    assert not required - set(parameters.keys())

    dt_clf = DecisionTreeClassifier(
        criterion=parameters['criterion'],
        max_features=max_features,
        min_samples_leaf=parameters['min_samples_leaf'])

    return BaggingClassifier(
        estimator=dt_clf,
        n_estimators=n_estimators,
        max_samples=max_samples,
        bootstrap=True,
        oob_score=parameters['oob_score'],
        n_jobs=parameters['n_jobs'],
        random_state=parameters['random_state'])


for n_trees in tqdm(n_trees_l, desc='Training Models', unit='Model'):
    clf = bagging_classifier(n_trees, 0.2, 1.0, clf_parameters)
    clf.fit(X_train, y_train)
    ...
```

⚠️ В оригинальном решении `bagging_classifier` **внутри себя вызывает `fit(X_train,
y_train)`**, а затем цикл обучает ту же модель ещё раз — лишняя работа и побочный эффект
через глобальные переменные; выше модель только собирается. Обучение бэггинга — самое
долгое в проекте (24 мин на 5 моделей): `max_features=1.0` — перебор всех признаков в
каждом узле.

### 8.3.8. `calculate_oob_score` и `non_overlapping_estimators`

Третий способ: обучить **пять лесов на пяти непересекающихся подвыборках** (каждая — каждый
пятый день со своим началом) и объединить голосованием. Так используются все данные, внутри
каждого леса метки независимы, и OOB-оценка каждого леса честна; общая OOB — среднее.

```python
def calculate_oob_score(classifiers):
    """Средняя OOB-оценка по списку обученных лесов."""
    return np.mean([clf.oob_score_ for clf in classifiers])


def non_overlapping_estimators(x, y, classifiers, n_skip_samples):
    """Обучить i-й классификатор на i-й непересекающейся подвыборке."""
    samples = [non_overlapping_samples(x, y, n_skip_samples, start_i)
               for start_i in range(len(classifiers))]
    return [clf.fit(sample_x, sample_y) for clf, (sample_x, sample_y) in zip(classifiers, samples)]
```

### 8.3.9. `NoOverlapVoter`: собственный ансамбль на базе `VotingClassifier`

```python
import abc
from sklearn.ensemble import VotingClassifier
from sklearn.base import clone
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import Bunch


class NoOverlapVoterAbstract(VotingClassifier):
    @abc.abstractmethod
    def _calculate_oob_score(self, classifiers):
        raise NotImplementedError

    @abc.abstractmethod
    def _non_overlapping_estimators(self, x, y, classifiers, n_skip_samples):
        raise NotImplementedError

    def __init__(self, estimator, voting='soft', n_skip_samples=4):
        # по одной копии базовой модели на каждую из n_skip_samples + 1 подвыборок
        estimators = [('clf' + str(i), estimator) for i in range(n_skip_samples + 1)]
        self.n_skip_samples = n_skip_samples
        # в современных scikit-learn (≥ 1.0) всё после estimators передаётся только по имени
        super().__init__(estimators, voting=voting)

    def fit(self, X, y, sample_weight=None):
        estimator_names, clfs = zip(*self.estimators)
        self.le_ = LabelEncoder().fit(y)
        self.classes_ = self.le_.classes_

        clone_clfs = [clone(clf) for clf in clfs]
        self.estimators_ = self._non_overlapping_estimators(X, y, clone_clfs, self.n_skip_samples)
        self.named_estimators_ = Bunch(**dict(zip(estimator_names, self.estimators_)))
        self.oob_score_ = self._calculate_oob_score(self.estimators_)
        return self


class NoOverlapVoter(NoOverlapVoterAbstract):
    def _calculate_oob_score(self, classifiers):
        return calculate_oob_score(classifiers)

    def _non_overlapping_estimators(self, x, y, classifiers, n_skip_samples):
        return non_overlapping_estimators(x, y, classifiers, n_skip_samples)


for n_trees in tqdm(n_trees_l, desc='Training Models', unit='Model'):
    clf_nov = NoOverlapVoter(RandomForestClassifier(n_trees, **clf_parameters))
    clf_nov.fit(X_train, y_train)
    ...
```

Класс наследует `VotingClassifier` и переопределяет только `fit`: каждой модели — своя
подвыборка. `voting='soft'` — усреднение вероятностей `predict_proba`, а не голосов; именно
вероятности нужны AI-альфе. `clone` обязателен: без него пять «разных» моделей оказались бы
одним объектом, переобучаемым пять раз. Финальная модель обучается на train + valid (в
продакшене обучение «докатывали» бы до текущего дня), 500 деревьев в каждом из пяти лесов:

```python
clf_nov = NoOverlapVoter(RandomForestClassifier(500, **clf_parameters))
clf_nov.fit(pd.concat([X_train, X_valid]), pd.concat([y_train, y_valid]))

for sample in (X_train, X_valid, X_test):
    show_sample_results(all_factors, sample, clf_nov, factor_names)
```

## 8.4. Дополнительные темы из квизов: важность признаков и SHAP

### 8.4.1. Как `scikit-learn` считает `feature_importances_`

Квиз `m7l6/sklearn_feature_importance` разбирает алгоритм на игрушечном примере: 100
наблюдений, три бинарных признака, метка $y = x_0 \land x_1$, признак $x_2$ всегда 0.
Для каждого внутреннего узла $i$ считается **важность узла**

$$
NI_i = w_i I_i - \left( w_{l} I_{l} + w_{r} I_{r} \right),
$$

где $I$ — загрязнённость Джини узла, $w$ — взвешенное число наблюдений, дошедших до узла
(`tree_.weighted_n_node_samples`), $l, r$ — потомки. Важность признака $j$ — сумма
важностей узлов, где по нему делался сплит, делённая на сумму важностей всех узлов:
$FI_j = \sum_{i:\,\mathrm{feature}(i)=j} NI_i \big/ \sum_i NI_i$. Квиз воспроизводит это
вручную из атрибутов `tree_.impurity`, `tree_.weighted_n_node_samples`, `tree_.children_left`
и получает в точности `model.feature_importances_`.

Неожиданный результат: для AND признаки $x_0$ и $x_1$ логически равноценны, но
`scikit-learn` даёт **бо́льшую важность тому, по которому сплит сделан ниже** — после
второго сплита листья становятся чистыми, и этому узлу достаётся большее снижение
загрязнённости. Это свойство метода, а не данных, и ответ на вопрос квиза «нужно ли
понимать алгоритм, если можно просто вызвать функцию»: без понимания нет интерпретации.

### 8.4.2. Значения Шепли (SHAP)

Значения Шепли (Shapley values) пришли из кооперативной теории игр: как разделить выигрыш
коалиции между участниками по их вкладу. Лундберг (2017) применил их к признакам модели
(SHapley Additive exPlanations, SHAP). Для наблюдения $x$ и признака $i$:

$$
\phi_i = \sum_{S \subseteq M \setminus \{i\}} \frac{|S|!\,(|M| - |S| - 1)!}{|M|!}\,
\bigl[ f(S \cup \{i\}) - f(S) \bigr],
$$

где $M$ — множество всех признаков, $S$ — подмножество без $i$, $f(S)$ — предсказание
модели, использующей только признаки $S$ (для пустого $S$ — среднее меток обучения).
Квадратная скобка — **предельный вклад** признака $i$, добавленного к коалиции $S$. Дробь —
доля перестановок признаков, в которых перед $i$ стоит ровно набор $S$: из $|M|!$ порядков
таких $|S|!\,(|M|-|S|-1)!$ (признаки $S$ в любом порядке, затем $i$, затем остальные). Веса
суммируются в 1.

Свойства, делающие SHAP удобным для интерпретации:

* **аддитивность**: $\sum_i \phi_i = f(x) - \mathbb{E}[f]$ — вклады признаков раскладывают
  отклонение предсказания от среднего (в квизе `tree_shap`: прогноз 1 при $x_0 = x_1 = 1$ и
  среднем 0,25 даёт по 0,375 каждому из двух признаков и 0 третьему);
* **локальность**: $\phi_i$ считается для конкретного наблюдения; глобальная важность —
  среднее абсолютных значений по выборке, $\frac{1}{N}\sum_j |\phi_{i,j}|$;
* **согласованность**: для AND признаки $x_0$ и $x_1$ получают одинаковые значения, а
  бесполезный $x_2$ — ровно 0, в отличие от важности по Джини.

Квиз `calculate_shap` вычисляет $\phi_0$ «в лоб», обучая отдельную модель на каждом
подмножестве признаков ($2^{M-1}$ моделей). Квиз `tree_shap` реализует идею **TreeSHAP**:
одно дерево, обученное на всех признаках, *имитирует* дерево на подмножестве $S$ — при
встрече узла с признаком не из $S$ надо пройти в **обе** ветви, взвесив их долями
наблюдений, а при признаке из $S$ — только в ветвь, куда попадает $x$. Библиотека `shap`
(`shap.TreeExplainer(model).shap_values(X)`) делает это за полиномиальное время; квиз
`rank_features` применяет её к лесу: для многоклассовой цели `shap_values` возвращает список
матриц (по классу), их конкатенируют, берут модуль и усредняют по строкам — глобальный рейтинг.

Почему интерпретируемость важна фондам? Риск-менеджер должен понимать, **на что** ставит
модель: если 80 % важности у признаков режима, стратегия — ставка на режим, а не на альфу;
инвесторам и регуляторам ответ «так сказал лес» не годится. Наконец, важности — инструмент
отбора признаков: признаки с нулевой важностью (в проекте — большинство секторов, `weekday`,
`qtr_start`) удаляются перед финальным обучением.

## 8.5. Выводы из результатов

**Одно дерево глубины 3.** Важности: `dispersion_20d` 0,47, `market_vol_120d` 0,19,
`sector_Real Estate` 0,13, `Momentum_1YR` 0,11, `sector_Healthcare` 0,10, остальные 0.
Вопрос ноутбука: почему у `dispersion_20d` наибольшая важность, если первый сплит сделан по
`Momentum_1YR`? Ответ автора репозитория: **суммарный прирост информации по
`dispersion_20d` больше** — он используется в нескольких узлах ниже корня, и сумма их
важностей превосходит важность корня (механика § 8.4.1: важность измеряет не «кто первый»,
а «кто больше очистил листья»). Уже одно дерево говорит: **режим рынка объясняет метку
лучше, чем сами альфы**.

**Лес 50…1000 деревьев, полные данные.** Ответ автора на вопросы ноутбука: с ростом числа
деревьев точность на обучении и валидации сначала растёт, затем выходит на плато;
OOB-оценка сначала падает, затем стабилизируется; **точность на обучении существенно выше
OOB** — большой разрыв указывает на переобучение. Усреднённые по пяти лесам важности:
`dispersion_20d` 0,127, `volatility_20d` 0,121, `market_vol_120d` 0,106, `market_vol_20d`
0,103, `Momentum_1YR` 0,096, `dispersion_120d` 0,080, `Overnight_Sentiment_Smoothed` 0,078,
далее `Mean_Reversion_Sector_Neutral_Smoothed`. Четыре признака режима — в первой шестёрке.

**AI-альфа через Alphalens, модель на полных данных.** На обучении результат, ожидаемо,
отличный. На валидации — тоже: **коэффициент Шарпа AI-альфы выше 2**, при том что исходные
факторы на этом отрезке шли вбок или вниз. Авторы курса формулируют верно: это «слишком
хорошо», и прежде чем радоваться, нужно исправить нарушение IID и снизить вероятность
переобучения.

**Три лекарства.** Подвыборка каждого пятого дня: разрыв train/OOB сжимается, но потеряно
80 % данных. `BaggingClassifier` с `max_samples=0.2`: кривые train, valid и OOB сходятся
гораздо ближе — «намного лучше в смысле согласия между тремя оценками». `NoOverlapVoter`:
сопоставимое согласие при использовании всех данных.

**Финальная модель** (`NoOverlapVoter` из пяти лесов по 500 деревьев, обучена на train +
valid). Напечатанные ноутбуком точности: `train: 0.514, oob: 0.512, valid: 0.507`. Все три
— **честные ~51 %**: модель едва лучше монетки, разрыв между обучением и остальными
оценками исчез. При этом AI-альфа на всех трёх отрезках, включая тест, даёт положительные
факторные доходности при заметно разном поведении исходных факторов. Так выглядит
реалистичный результат в финансах: 51 % на 500 акциях ежедневно — это много, если ошибки
не коррелированы, и это та «тонкая» альфа, которую оптимизация главы 9 должна донести до
PnL, не растеряв на издержках.

## 8.6. Типичные ошибки и ловушки

⚠️ **Случайное разбиение рядов.** `train_test_split(shuffle=True)` на дневных данных
гарантирует утечку: валидация увидит соседей обучающих строк с теми же признаками и той же
меткой. Только по датам, только вперёд во времени.

⚠️ **Сдвиг цели без `groupby`.** `all_factors['return_5d'].shift(-5)` на мультииндексе
(дата, акция) сдвинет строки по *позиции*, перемешав акции внутри дня. Всегда
`groupby(level=1).shift(-5)`.

⚠️ **OOB-оценка как доказательство.** При перекрывающихся метках OOB почти равна точности
на обучении (квиз `dependent_labels`, пятикратное дублирование строк: `train 0.98, oob 0.96,
valid 0.66`) — симптом зависимости наблюдений, а не хорошей модели.

⚠️ **Атрибут `index.levels` после фильтрации.** Уровни мультииндекса не пересчитываются при
`dropna()` и `.loc`; срез по ним может захватить несуществующие даты или сломать подсчёт
дней. Используйте `get_level_values(k).unique()`.

⚠️ **Подписи в финальном выводе проекта перепутаны.** Формат-строка печатает `train, oob,
valid`, а аргументы переданы в порядке `score(train), score(valid), oob_score_`: число под
подписью «oob» (0,512) — валидация, под подписью «valid» (0,507) — OOB. Вывод о ~51 % не
меняется, но помните: финальная модель обучена на train + valid, поэтому «валидация» здесь
уже *внутри* выборки; единственная настоящая оценка вне выборки — тест.

⚠️ **Устаревшие сигнатуры.** `VotingClassifier(estimators, voting)` с позиционным `voting`
падает в scikit-learn ≥ 1.0 — только `voting=voting`. У `BaggingClassifier` базовая модель
— параметр `estimator`. Аргумент `offset` у `pd.Timestamp` удалён; `'BM'`/`'BQ'` →
`'BME'`/`'BQE'`; в квизе `dependent_labels` `DataFrame.append` → `pd.concat`, `tqdm` — из `tqdm.auto`.

## 8.7. Вопросы для самопроверки

1. Узел содержит 60 % объектов класса 1 и 40 % класса 0. Вычислите энтропию и загрязнённость
   Джини. Почему для 50/50 Джини равна ровно 0,5?
2. Почему в бутстреп-выборку попадает в среднем около 2/3 уникальных наблюдений? Как из
   этого следует OOB-оценка и почему она ломается при перекрывающихся метках?
3. Квантизация цели `quantiles(2)` делает метку рыночно-нейтральной. Объясните, почему
   регрессия на сырую 5-дневную доходность дала бы модель, предсказывающую в основном
   рынок, а не относительную доходность.
4. `target_3 = shift(-2)`. На сколько дней перекрываются окна `target` и `target_3`? Какой
   знак и примерную величину автокорреляции вы ожидаете по сравнению с `target_1`?
5. Почему `min_samples_leaf = 500 × 10`, а не 50? Какое наблюдение в терминах коэффициента
   Шарпа заставило бы вас увеличить этот параметр на порядок?
6. Сравните три способа борьбы с перекрывающимися метками по двум осям: сколько данных
   используется и насколько честна OOB-оценка. Почему `NoOverlapVoter` считает OOB как
   среднее по лесам, а не по всем деревьям сразу?
7. Для AND на трёх признаках `scikit-learn` даёт $FI_0 \ne FI_1$, а SHAP — $\phi_0 = \phi_1$.
   Объясните расхождение через формулу важности узла (подсказка: какой сплит делает листья
   чистыми?).
8. В формуле Шепли вес $\frac{|S|!\,(|M|-|S|-1)!}{|M|!}$ при $|M| = 3$, $|S| = 0$ равен $1/3$.
   Назовите две перестановки признаков, соответствующие этому случаю для $x_0$.
9. Модель показала ~51 % и на обучении, и вне выборки. Почему это может быть *хорошим*
   результатом и почему этого недостаточно для торговли (подсказка: глава 9)?

## 8.8. Файлы репозитория и что читать дальше

* 📁 `Projects/7-Combining-alphas/project_7_starter.ipynb` — проект (признаки, разбиение, лес,
  три способа борьбы с перекрытием, `NoOverlapVoter`, Alphalens); рядом `project_helper.py`, `project_tests.py`.
* 📁 `Quiz/m7/m7l3/feature_engineering_solution.ipynb` — признаки: волатильность, объём,
  режим рынка, календарь, квантизация цели.
* 📁 `Quiz/m7/dependent_labels_solution.ipynb` — симуляция перекрывающихся меток
  дублированием строк и три решения на игрушечных данных.
* 📁 `Quiz/m7/m7l6/sklearn_feature_importance_solution.ipynb`, `calculate_shap_solution.ipynb`,
  `tree_shap_solution.ipynb`, `rank_features_solution.ipynb` — важность по Джини вручную,
  значения Шепли «в лоб», TreeSHAP, рейтинг признаков.
* 📁 `Quiz/m7/titanic_survival_exploration.ipynb`, `spam_rf.ipynb`, `Learning Curve.ipynb`
  — вводные упражнения модуля; 📁 `Notes/7-Combining-signals-for-enhanced-alpha/*.md` —
  конспекты (энтропия, OOB, перекрытие, метрики).

Что дальше. AI-альфа — это альфа-вектор, и его судьба решается в оптимизаторе: в
[главе 9](09-bektesting.md) мы перейдём от весов к позициям в долларах, добавим в целевую
функцию транзакционные издержки и проверим, переживает ли «тонкая» альфа столкновение с
рыночным влиянием. Для углубления — М. Лопес де Прадо, *Advances in Financial Machine
Learning* (главы о метках, уникальности выборок и кросс-валидации для финансов).

---

Навигация: [← Глава 7. Нейронные сети и анализ тональности StockTwits](07-neyroseti-i-sentiment.md) | [Глава 9. Бэктестинг →](09-bektesting.md)
