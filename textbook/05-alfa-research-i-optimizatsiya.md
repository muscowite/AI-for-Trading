# Глава 5. Альфа-ресёрч: конвейер факторов, оценка через Alphalens и оптимизация с риск-моделью

> Чему научитесь: строить альфа-факторы на всём универсуме акций через Zipline Pipeline,
> приводить их к сопоставимому виду (центрирование по сектору → ранжирование → z-оценка →
> сглаживание), оценивать в Alphalens (факторные доходности, квантили, FRA, коэффициент
> Шарпа, rank IC), объединять в один альфа-вектор и находить веса портфеля выпуклой
> оптимизацией с риск-моделью из главы 4. Модуль M4, проект 4 *Multi-factor Model*
> (части «Alpha Factors» и «Optimal Portfolio»): 📁 `Projects/4-Multi-factor-Model/project_4_starter.pdf`.

## 5.1. Мотивация и гипотеза

В главе 4 мы построили риск-модель — она говорит, *чего портфелю бояться*. Теперь нужна
вторая половина: альфа-модель, которая говорит, *на что ставить*. Альфа-ресёрч — это
цикл: гипотеза о поведении рынка → её формализация в виде фактора (вектора значений по
акциям на дату) → статистическая оценка фактора на истории → отбор и комбинирование
факторов → портфель.

В проекте 4 проверяются три гипотезы (и две их сглаженные версии):

1. **моментум на горизонте 1 год**: «высокая доходность за прошедшие 12 месяцев
   (252 торговых дня) пропорциональна будущей доходности»;
2. **возврат к среднему за 5 дней, секторно-нейтральный**: «акции, краткосрочно обогнавшие
   (отставшие от) свой сектор, вернутся к нему»;
3. **ночной сентимент**: «накопленная за неделю ночная доходность (от закрытия до
   следующего открытия) измеряет настроение розничных инвесторов по отношению к компании
   и предсказывает доходность».

Ни одна из гипотез не является секретом. Смысл упражнения — освоить *процедуру*: как
превратить идею в фактор по 490 акциям за два года, честно его измерить и собрать портфель,
который риск-модель сочтёт приемлемым. Инструменты — **Zipline Pipeline** (вычисляет
факторы для всех акций за все даты) и **Alphalens** (стандартный набор диагностик); сегодня
используются их форки `zipline-reloaded` и `alphalens-reloaded`, код приведён как в проекте
с пометками об изменившихся импортах.

## 5.2. Теория

### 5.2.1. Устройство Zipline Pipeline

Pipeline — декларативное описание вычислений над таблицей (дата × акция). Вы описываете
*что* посчитать, а движок (`SimplePipelineEngine`) сам определяет, сколько истории
подгрузить, и возвращает результат одним DataFrame с `MultiIndex (date, asset)`. Три типа
узлов:

* **`Factor`** — число на каждую акцию и дату (`Returns`, `SimpleMovingAverage`,
  `AverageDollarVolume`, `AnnualizedVolatility`, любой `CustomFactor`);
* **`Filter`** — булево значение; задаёт **универсум**. В проекте
  `AverageDollarVolume(window_length=120).top(500)` — 500 акций с наибольшим средним
  долларовым объёмом за 120 дней; ликвидность важна и для исполнения, и для доступности
  коротких позиций;
* **`Classifier`** — категория (сектор, биржа). Класс `Sector` проекта читает коды
  секторов из `data.npy`: 11 категорий 0…10, `-1` — пропуск.

Произвольный фактор — наследник `CustomFactor` с атрибутами класса `inputs` (какие
столбцы данных подать, например `[USEquityPricing.close]`) и `window_length` (сколько
строк-дней истории передать) и методом `compute(self, today, assets, out, *inputs)`:
`today` — дата строки результата, `assets` — идентификаторы акций, `out` — заранее
выделенный массив, в который надо *записать* результат (а не вернуть), `*inputs` — по
одному 2D-массиву (окно × акции, последняя строка — самая свежая) на каждый элемент
`inputs`. Так, `Returns` реализован одной строкой `out[:] = (close[-1] - close[0]) / close[0]`.
Атрибут `window_safe = True` разрешает использовать выход фактора как вход
другого оконного фактора (Zipline требует явно пометить, что значения корректно переживают
корректировки на сплиты); `outputs = ['beta', 'gamma']` позволяет одному `compute`
возвращать несколько факторов через `out.beta[:]`, `out.gamma[:]`. Факторы поддерживают
арифметику (`f1 + f2`, `-f`, `f1 * f2`) и методы `demean`, `rank`, `zscore`, `top`,
`bottom`. Результат `engine.run_pipeline(pipeline, start, end)` — DataFrame с индексом
`(date, asset)` и столбцом на каждый добавленный фактор.

### 5.2.2. Конвейер преобразований фактора

Стандартная цепочка проекта: **`demean(groupby=sector)` → `rank()` → `zscore()`**.

**Центрирование по сектору** вычитает из значения акции среднее по её сектору. Без него
в год роста технологического сектора все IT-акции получили бы высокий моментум, и портфель
содержал бы *секторную ставку* — экспозицию к риск-фактору, который известен всем и не
даёт преимущества (глава 4). После центрирования фактор сравнивает акцию с соседями по
сектору, и портфель становится **секторно-нейтральным**. Квиз `sector_neutral` указывает
два параметра `demean`: `groupby` (классификатор для разбиения) и `mask` (какие акции
учитывать); для нейтрализации по всему универсуму достаточно `groupby`.

**Ранжирование** заменяет значения порядковыми номерами 1…N по сечению — фактор становится
**устойчивым к выбросам** (акция с доходностью +300 % получит ранг N, а не вес, в десять раз
превышающий остальные); цена — потеря информации о расстояниях между значениями.
**Z-оценка** $(x - \mu)/\sigma$ по сечению переводит ранги в величины со средним 0 и
дисперсией 1 (квиз `zscore`: «в основном между −2 и +2») — факторы разной природы
становятся **сопоставимыми**, их можно складывать или усреднять.

**Сглаживание** — `SimpleMovingAverage(inputs=[factor], window_length=5)`, затем снова
`rank()` и `zscore()` (повторно центрировать не нужно — вход уже нейтрален). Сглаженный
фактор меняется медленнее: портфель реже торгует, шум подавляется; плата — запаздывание.

### 5.2.3. Гипотезы факторов проекта

**Моментум 1 год.** `Returns(window_length=252)` — доходность за 252 дня, далее стандартный
конвейер. Победители прошлого года продолжают обгонять.

**Возврат к среднему за 5 дней.** `Returns(window_length=5)`, центрирование по сектору,
ранг, z-оценка — и **знак минус**: высокий сигнал должен соответствовать *низкой* прошлой
доходности. Без минуса получится краткосрочный моментум, который, судя по литературе, убыточен.

**Ночной сентимент.** Основан на статье Aboody, Even-Tov, Lehavy, Trueman «Overnight
Returns and Firm-Specific Investor Sentiment» (SSRN 2554010). Квиз `overnight_returns`
цитирует аннотацию: ночная доходность рассматривается как мера настроений инвесторов по
отношению к конкретной фирме; авторы находят краткосрочную *персистентность* ночных
доходностей (согласующуюся с устойчивостью спроса розничных инвесторов), более сильную для
трудно оцениваемых компаний, и долгосрочное отставание акций с высокой ночной доходностью.
Механизм, по Berkman et al. (2012): привлекающие внимание события создают спрос частных
инвесторов у открытия следующего дня и давление на цену при открытии. Формализация:

$$
\mathrm{CTO}_t = \frac{\mathrm{open}_t - \mathrm{close}_{t-1}}{\mathrm{close}_{t-1}},
\qquad
\mathrm{TrailingOvernight}_t = \sum_{\tau = t-4}^{t} \mathrm{CTO}_\tau ,
$$

где $\mathrm{CTO}$ — ночная (close-to-open) доходность, а недельная сумма реализована
через `np.nansum(cto, axis=0)` — по дням для каждой акции, пропуски считаются нулями.
Проект использует краткосрочную персистентность: высокий накопленный ночной сентимент →
длинная позиция.

**Регрессия цены против времени (квиз `regression_against_time`).** Статья «The Formation
Process of Winners and Losers in Momentum Investing» (SSRN 2610571) утверждает, что важна
не только прошлая доходность, но и *форма траектории*: из двух акций с одинаковой годовой
доходностью та, чьи цены шли по выпуклой траектории, имеет не меньшую ожидаемую
доходность, чем та, чья траектория вогнута. Авторы регрессируют дневные цены на порядковый
номер дня и его квадрат:

$$
\mathrm{Close}_t = \beta\, t + \gamma\, t^2 + \varepsilon_t ,
$$

где $t = 0, 1, \dots, 251$; $\gamma$ измеряет выпуклость (положительный — ускоряющийся
рост), $\beta$ — средний наклон. Условный фактор — произведение рангов $\beta$ и $\gamma$.

### 5.2.4. Оценка факторов в Alphalens

Alphalens начинает с выравнивания фактора и цен:
`get_clean_factor_and_forward_returns(factor, prices, periods=[1])`. Для каждой пары
(дата, акция) вычисляется **форвардная доходность** за следующий период (столбец `1D`) и
присваивается **квантиль** фактора (по умолчанию квинтили). Проект подчёркивает: оценка
идёт с **задержкой 1 (delay 1)** — фактор, посчитанный по данным до закрытия дня $t$,
сопоставляется с доходностью от $t$ к $t+1$. Это защита от заглядывания в будущее.

Метрики:

* **Факторные доходности** (`performance.factor_returns`) — дневная доходность
  теоретического портфеля с весами, пропорциональными стандартизованному альфа-вектору
  (глава 4, §4.2.2): долларово-нейтральный, плечо 1. График `(1 + r).cumprod()` должен идти
  «вверх и вправо».
* **Средняя доходность по квантилям** (`performance.mean_return_by_quantile`,
  `demeaned=True` — из доходностей вычитается среднее по универсуму) — в базисных пунктах в
  день (×10 000). Хороший фактор **монотонен**: Q1 < Q2 < … < Q5. Квиз `quantiles`:
  сглаженный фактор даёт более ровное распределение по квантилям, несглаженный
  «зарабатывает хвостами».
* **Автокорреляция рангов фактора (factor rank autocorrelation, FRA)**
  (`performance.factor_rank_autocorrelation`) — корреляция Спирмена рангов сегодня и вчера.
  Если ранги почти не меняются, портфель почти не торгует; FRA — прокси
  **оборачиваемости** без бэктеста. Квиз `turnover`: FRA сглаженного фактора выше — меньше
  сделок и издержек.
* **Коэффициент Шарпа факторных доходностей**: $\sqrt{252}\,\bar r / \sigma_r$ для дневных
  данных. Ориентир курса: для одиночного фактора на этом универсуме приемлемо значение
  около 1 и выше.
* **Информационный коэффициент по рангам (rank IC)**
  (`performance.factor_information_coefficient`) — корреляция Спирмена между
  значениями фактора и форвардными доходностями по сечению на каждую дату; положительное
  значение — фактор ранжировал акции «в правильную сторону». Для реальных альф это сотые
  доли; важна устойчивость знака.
* **Коэффициент переноса (transfer coefficient, TC)** — корреляция Пирсона между
  стандартизованным альфа-вектором и весами оптимизатора: сколько альфы «доехало» до
  портфеля сквозь ограничения. В квизе `transfer_coefficient` веса имитируются как
  `standardize_alpha(alpha) + шум(σ = 0,001)`, TC через `scipy.stats.pearsonr` близок к 1.

**Про unix-timestamp.** Alphalens 0.3.2 при вызове `mean_return_by_quantile` и
`factor_rank_autocorrelation` на индексе с датами выдавал ошибку, требующую целочисленный
тип (квизы `quantiles` и `turnover` просят посмотреть на неё). Обход — заменить уровень
`date` на unix-время (`x.timestamp()`); для `factor_returns` это не требовалось, отсюда два
словаря в проекте: `clean_factor_data` и `unixt_factor_data`. В `alphalens-reloaded` обход
обычно не нужен, но безвреден.

### 5.2.5. Оптимизация с альфа-вектором и риск-моделью

Задача: найти веса $\mathbf{x}$, максимально согласные с альфа-вектором $\boldsymbol{\alpha}$,
при ограничениях риск-модели и мандата фонда. Базовая постановка проекта:

$$
\max_{\mathbf{x}}\ \boldsymbol{\alpha}^T\mathbf{x}
\quad\text{при}\quad
\begin{cases}
\mathbf{x}^T(\mathbf{BFB}^T + \mathbf{S})\mathbf{x} \le \sigma_{\max}^2 & \text{(лимит риска)}\\
f_{\min} \le \mathbf{B}^T\mathbf{x} \le f_{\max} & \text{(лимиты экспозиций к риск-факторам)}\\
\mathbf{1}^T\mathbf{x} = 0 & \text{(рыночная нейтральность)}\\
\lVert \mathbf{x} \rVert_1 \le 1 & \text{(плечо)}\\
x_{\min} \le x_i \le x_{\max} & \text{(лимиты отдельных позиций)}
\end{cases}
$$

Здесь $\sigma_{\max}$ — лимит риска (годовое стандартное отклонение, в проекте 0,05),
$\mathbf{B}$, $\mathbf{F}$, $\mathbf{S}$ — матрицы риск-модели главы 4, $f_{\min}, f_{\max}$ —
границы экспозиции портфеля к каждому из 20 статистических факторов, $x_{\min}, x_{\max}$ —
границы веса одной акции. Все ограничения выпуклы, цель линейна — `cvxpy` находит
глобальный оптимум. Смысл ограничений: **лимит риска** связывает альфа-модель с
риск-моделью (без него оптимизатор наберёт альфу, не глядя на волатильность); **лимиты
экспозиций** не дают портфелю стать ставкой на риск-фактор (в проекте ±10 фактически не
активны — экспозиции порядка сотых; в `OptimalHoldingsStrictFactor` они сужены до ±0,015
и начинают действовать); **рыночная нейтральность** — торгуем относительную доходность, а не
направление рынка; **плечо** — валовая позиция не больше капитала, без него линейная цель
неограничена (квиз `Advanced_Opt`: решатель возвращает `UNBOUNDED`); **лимиты позиций** —
защита от концентрации, но ±0,55 при плече 1 позволяют вложить всё в две акции, что и
происходит.

Ограничения частично **заменяют друг друга**: концентрацию можно подавить жёсткими
лимитами позиций (±0,02 в `OptimalHoldingsStrictFactor`), штрафом $-\lambda\lVert \mathbf{x} \rVert_2$
в целевой функции (`OptimalHoldingsRegualization`) или сменой цели (минимизировать
расстояние до стандартизованной альфы, которая уже диверсифицирована). Квиз
`Regularization` показывает это на двух активах: при $\alpha = (0{,}1;\ 0{,}5)$ и
единственном ограничении $\lVert \mathbf{x} \rVert_1 \le 1$ весь вес уходит во второй актив,
$(0;\ 1)$; добавление $\lambda\lVert \mathbf{x} \rVert_2$ с $\lambda = 0{,}5099$ даёт
$(0{,}000335;\ 0{,}001674)$ — веса малы (при $\lambda \approx \lVert \boldsymbol{\alpha} \rVert_2$
штраф почти полностью подавляет альфу), но распределены между активами в пропорции альф
$0{,}1 : 0{,}5$. Уменьшая $\lambda$, получаем бóльшие веса в той же пропорции.

## 5.3. Разбор проекта шаг за шагом

### 5.3.1. Факторы моментума и возврата к среднему

```python
from zipline.pipeline.factors import Returns, SimpleMovingAverage

def momentum_1yr(window_length, universe, sector):
    """Моментум за год: доходность за window_length дней, нейтральная к сектору."""
    return Returns(window_length=window_length, mask=universe) \
        .demean(groupby=sector) \
        .rank() \
        .zscore()

def mean_reversion_5day_sector_neutral(window_length, universe, sector):
    """Возврат к среднему за 5 дней, секторно-нейтральный (знак минус!)."""
    factor = -(Returns(window_length=window_length, mask=universe)
               .demean(groupby=sector)
               .rank(ascending=True)
               .zscore())
    return factor

def mean_reversion_5day_sector_neutral_smoothed(window_length, universe, sector):
    """Сглаженная версия: SMA(5) от несглаженного фактора, затем снова rank и zscore."""
    factor_raw = mean_reversion_5day_sector_neutral(window_length, universe, sector)
    factor = SimpleMovingAverage(inputs=[factor_raw], window_length=window_length) \
        .rank() \
        .zscore()
    return factor
```

`momentum_1yr` дан в проекте как образец; две другие функции — задания. Содержательное
различие возврата к среднему и моментума — окно (5 вместо 252) и знак. `mask=universe`
ограничивает вычисление акциями универсума; `demean(groupby=sector)` получает экземпляр
`project_helper.Sector()`. Тесты проекта — интеграционные: мини-конвейер на 4 акциях за три
дня. Для просмотра данных берётся двухлетнее окно
`factor_start_date = universe_end_date - pd.DateOffset(years=2, days=2)`; авторское
примечание: ровно два года назад рынок был закрыт, а Pipeline не принимает начальную дату в
неторговый день, поэтому отступили на два дополнительных дня.

### 5.3.2. Фактор ночного сентимента

```python
from zipline.pipeline.data import USEquityPricing

class CTO(Returns):
    """Ночная доходность (close-to-open) по гипотезе Aboody et al. (SSRN 2554010)."""
    inputs = [USEquityPricing.open, USEquityPricing.close]

    def compute(self, today, assets, out, opens, closes):
        # opens и closes — матрицы 2 × N: opens[-1] — сегодняшнее открытие,
        # closes[0] — вчерашнее закрытие
        out[:] = (opens[-1] - closes[0]) / closes[0]

class TrailingOvernightReturns(Returns):
    """Сумма ночных доходностей за окно."""
    window_safe = True

    def compute(self, today, asset_ids, out, cto):
        out[:] = np.nansum(cto, axis=0)

def overnight_sentiment(cto_window_length, trail_overnight_returns_window_length, universe):
    cto_out = CTO(mask=universe, window_length=cto_window_length)
    return TrailingOvernightReturns(inputs=[cto_out],
                                    window_length=trail_overnight_returns_window_length) \
        .rank() \
        .zscore()

def overnight_sentiment_smoothed(cto_window_length, trail_overnight_returns_window_length, universe):
    unsmoothed_factor = overnight_sentiment(cto_window_length,
                                           trail_overnight_returns_window_length, universe)
    return SimpleMovingAverage(inputs=[unsmoothed_factor],
                               window_length=trail_overnight_returns_window_length) \
        .rank() \
        .zscore()
```

`CTO` наследует `Returns`, переопределяя `inputs` и `compute`; с `window_length=2` приходят
две строки: индекс 0 — вчера, −1 — сегодня. `TrailingOvernightReturns` принимает выход `CTO`
как вход (поэтому `window_safe = True`) и суммирует по оси 0 — по дням для каждой акции
(квиз `overnight_returns` разбирает, почему не `axis=1`). В проекте этот фактор *не*
центрируется по сектору, в квизе — центрируется; это решение стоит проверить самостоятельно.

### 5.3.3. Общий конвейер и запуск

```python
universe = AverageDollarVolume(window_length=120).top(500)
sector = project_helper.Sector()

pipeline = Pipeline(screen=universe)
pipeline.add(momentum_1yr(252, universe, sector), 'Momentum_1YR')
pipeline.add(mean_reversion_5day_sector_neutral(5, universe, sector),
             'Mean_Reversion_5Day_Sector_Neutral')
pipeline.add(mean_reversion_5day_sector_neutral_smoothed(5, universe, sector),
             'Mean_Reversion_5Day_Sector_Neutral_Smoothed')
pipeline.add(overnight_sentiment(2, 5, universe), 'Overnight_Sentiment')
pipeline.add(overnight_sentiment_smoothed(2, 5, universe), 'Overnight_Sentiment_Smoothed')
all_factors = engine.run_pipeline(pipeline, factor_start_date, universe_end_date)
```

`screen=universe` оставляет только акции универсума на каждую дату. Результат — DataFrame с
индексом `(date, asset)` и пятью столбцами z-оценок; на 2014-01-03 у A моментум 1,50, у AAPL
−1,48 (Apple в 2013 году отставала от сектора).

### 5.3.4. Подготовка данных для Alphalens

```python
import alphalens as al   # alphalens-reloaded: тот же API

# levels может содержать неиспользуемые значения, поэтому get_level_values(...).unique()
assets = all_factors.index.get_level_values(1).unique().tolist()
pricing = get_pricing(data_portal, trading_calendar, assets, factor_start_date, universe_end_date)

clean_factor_data = {
    factor: al.utils.get_clean_factor_and_forward_returns(
        factor=factor_data, prices=pricing, periods=[1])
    for factor, factor_data in all_factors.items()}   # pandas 2.x: items()

unixt_factor_data = {
    factor: factor_data.set_index(pd.MultiIndex.from_tuples(
        [(x.timestamp(), y) for x, y in factor_data.index.values],
        names=['date', 'asset']))
    for factor, factor_data in clean_factor_data.items()}
```

Alphalens сообщает, сколько строк отброшено («Dropped 1.2 % entries …»): для последней даты
форвардной доходности нет, плюс пропуски цен. Порог `max_loss` — 35 %; потери 0,4–2,3 % безопасны.

### 5.3.5. Факторные доходности, квантили, FRA

```python
ls_factor_returns = pd.DataFrame()
for factor, factor_data in clean_factor_data.items():
    ls_factor_returns[factor] = al.performance.factor_returns(factor_data).iloc[:, 0]
(1 + ls_factor_returns).cumprod().plot()

qr_factor_returns = pd.DataFrame()
for factor, factor_data in unixt_factor_data.items():
    qr_factor_returns[factor] = al.performance.mean_return_by_quantile(factor_data)[0].iloc[:, 0]
(10000 * qr_factor_returns).plot.bar(subplots=True, sharey=True, layout=(4, 2),
                                     figsize=(14, 14), legend=False)

ls_FRA = pd.DataFrame()
for factor, factor_data in unixt_factor_data.items():
    ls_FRA[factor] = al.performance.factor_rank_autocorrelation(factor_data)
ls_FRA.plot(title="Factor Rank Autocorrelation")
```

`factor_returns` возвращает столбец на период (`1D`), `mean_return_by_quantile` — кортеж
(средние, стандартные ошибки), нужен первый элемент; ×10 000 переводит доли в базисные пункты.

### 5.3.6. Коэффициент Шарпа факторов

```python
def sharpe_ratio(factor_returns, annualization_factor):
    """Годовой коэффициент Шарпа для каждого столбца факторных доходностей."""
    return annualization_factor * factor_returns.mean() / factor_returns.std()

daily_annualization_factor = np.sqrt(252)
sharpe_ratio(ls_factor_returns, daily_annualization_factor).round(2)
```

```text
Mean_Reversion_5Day_Sector_Neutral             1.37
Mean_Reversion_5Day_Sector_Neutral_Smoothed    1.27
Momentum_1YR                                   1.13
Overnight_Sentiment                            0.12
Overnight_Sentiment_Smoothed                   0.45
```

Безрисковая ставка не вычитается — для долларово-нейтрального портфеля капитал «в рынке» не
требуется. Квиз `sharpe_ratio` обобщает функцию на месячные данные (множитель $\sqrt{12}$).

### 5.3.7. Комбинированный альфа-вектор

```python
selected_factors = all_factors.columns[[1, 2, 4]]
# Mean_Reversion_5Day_Sector_Neutral_Smoothed, Momentum_1YR, Overnight_Sentiment_Smoothed

all_factors['alpha_vector'] = all_factors[selected_factors].mean(axis=1)
alphas = all_factors[['alpha_vector']]
alpha_vector = alphas.loc[all_factors.index.get_level_values(0)[-1]]   # последняя дата
```

Простейшее **комбинирование альф** — среднее трёх z-оценок: сглаженного возврата к
среднему, годового моментума и сглаженного ночного сентимента. Все три уже стандартизованы,
поэтому среднее не даёт преимущества ни одному. Автор выбирает сглаженные версии (меньше
оборачиваемости), а ночной сентимент берёт несмотря на низкий Шарп — он слабо коррелирует
с остальными и может добавить диверсификации сигнала. Более умное комбинирование — задача
машинного обучения (глава 8). Для оптимизации берётся срез на последнюю дату: 490 чисел,
например −0,586 для A, −0,068 для AAPL.

### 5.3.8. Абстрактный оптимизатор

```python
import cvxpy as cvx
from abc import ABC, abstractmethod

class AbstractOptimalHoldings(ABC):
    @abstractmethod
    def _get_obj(self, weights, alpha_vector):
        """Целевая функция cvxpy."""
        raise NotImplementedError()

    @abstractmethod
    def _get_constraints(self, weights, factor_betas, risk):
        """Список ограничений cvxpy."""
        raise NotImplementedError()

    def _get_risk(self, weights, factor_betas, alpha_vector_index, factor_cov_matrix,
                  idiosyncratic_var_vector):
        # экспозиции портфеля f = B'x — только для акций, присутствующих в альфа-векторе
        f = factor_betas.loc[alpha_vector_index].values.T @ weights
        X = factor_cov_matrix
        S = np.diag(idiosyncratic_var_vector.loc[alpha_vector_index].values.flatten())
        # дисперсия портфеля: f'Ff + x'Sx
        return cvx.quad_form(f, X) + cvx.quad_form(weights, S)

    def find(self, alpha_vector, factor_betas, factor_cov_matrix, idiosyncratic_var_vector):
        weights = cvx.Variable(len(alpha_vector))
        risk = self._get_risk(weights, factor_betas, alpha_vector.index,
                              factor_cov_matrix, idiosyncratic_var_vector)
        obj = self._get_obj(weights, alpha_vector)
        constraints = self._get_constraints(
            weights, factor_betas.loc[alpha_vector.index].values, risk)

        prob = cvx.Problem(obj, constraints)
        prob.solve(solver=cvx.ECOS, max_iters=500)   # см. §5.6 о решателях

        optimal_weights = np.asarray(weights.value).flatten()
        return pd.DataFrame(data=optimal_weights, index=alpha_vector.index)
```

Класс дан в проекте целиком. `_get_risk` реализует $\boldsymbol{\beta}_p^T\mathbf{F}\boldsymbol{\beta}_p + \mathbf{x}^T\mathbf{S}\mathbf{x}$
из главы 4: `quad_form(f, F)` — факторный риск, `quad_form(weights, S)` — специфический.
$\mathbf{S}$ восстанавливается из вектора дисперсий только для акций альфа-вектора
(`.loc[alpha_vector_index]`) — поэтому проект хранил диагональ отдельно. Модернизация
cvxpy: произведение матрицы на переменную — оператор `@`, а не `*` (см. §5.6).

### 5.3.9. `OptimalHoldings`: максимум альфы при ограничениях

```python
class OptimalHoldings(AbstractOptimalHoldings):
    def _get_obj(self, weights, alpha_vector):
        assert len(alpha_vector.columns) == 1
        # max alpha' x
        return cvx.Maximize(alpha_vector.values.flatten() @ weights)

    def _get_constraints(self, weights, factor_betas, risk):
        assert len(factor_betas.shape) == 2
        constraints = [
            risk <= self.risk_cap ** 2,                   # дисперсия <= (лимит риска)^2
            factor_betas.T @ weights <= self.factor_max,  # экспозиции к факторам
            factor_betas.T @ weights >= self.factor_min,
            sum(weights) == 0.0,                          # рыночная нейтральность
            sum(cvx.abs(weights)) <= 1.0,                 # плечо <= 1
            weights >= self.weights_min,                  # лимиты позиций
            weights <= self.weights_max]
        return constraints

    def __init__(self, risk_cap=0.05, factor_max=10.0, factor_min=-10.0,
                 weights_max=0.55, weights_min=-0.55):
        self.risk_cap = risk_cap
        self.factor_max = factor_max
        self.factor_min = factor_min
        self.weights_max = weights_max
        self.weights_min = weights_min

optimal_weights = OptimalHoldings().find(
    alpha_vector, risk_model['factor_betas'],
    risk_model['factor_cov_matrix'], risk_model['idiosyncratic_var_vector'])
optimal_weights.plot.bar(legend=None, title='Portfolio % Holdings by Stock')
```

`alpha_vector` — DataFrame 490 × 1, поэтому `.values.flatten()` даёт одномерный массив, и
`@ weights` — скалярное произведение. `risk` — дисперсия, поэтому сравнивается с
*квадратом* лимита риска. `factor_betas.T @ weights` — вектор из 20 экспозиций портфеля,
неравенства применяются поэлементно.

Результат с параметрами по умолчанию автор комментирует одним словом: «Yikes» — **веса
сконцентрированы в нескольких акциях**. Это ожидаемо: линейная цель достигает максимума на
границе выпуклой области, в «углах», а лимиты ±0,55 и плечо 1 допускают углы вида «+0,5 в
одной акции, −0,5 в другой»; лишь лимит риска 5 % мешает взять самые волатильные бумаги.
График `project_helper.get_factor_exposures(factor_betas, optimal_weights)`
($\mathbf{B}^T\mathbf{x}$) показывает заметную экспозицию к нескольким статистическим факторам.

### 5.3.10. `OptimalHoldingsRegualization`: регуляризация

```python
class OptimalHoldingsRegualization(OptimalHoldings):
    def _get_obj(self, weights, alpha_vector):
        assert len(alpha_vector.columns) == 1
        # max alpha' x - lambda * ||x||_2
        return cvx.Maximize(alpha_vector.values.flatten() @ weights
                            - self.lambda_reg * cvx.norm(weights, 2))

    def __init__(self, lambda_reg=0.5, **kwargs):
        # остальные параметры (risk_cap, factor_max, ...) — как у OptimalHoldings
        super().__init__(**kwargs)
        self.lambda_reg = lambda_reg

optimal_weights_1 = OptimalHoldingsRegualization(lambda_reg=5.0).find(
    alpha_vector, risk_model['factor_betas'],
    risk_model['factor_cov_matrix'], risk_model['idiosyncratic_var_vector'])
```

Имя класса с опечаткой (`Regualization`) сохранено как в проекте; `__init__` в проекте
дублирует присваивания родителя — здесь он сведён к `super().__init__`. Целевая функция —
$\boldsymbol{\alpha}^T\mathbf{x} - \lambda\lVert \mathbf{x} \rVert_2$ (текст ноутбука пишет
«+», но при максимизации штраф должен вычитаться — код верен). Норма $\ell_2$ при
фиксированной $\ell_1$-норме минимальна при равномерных весах, поэтому штраф «расталкивает»
вес по многим акциям. При $\lambda = 5$ автор получает «хорошо диверсифицированный»
портфель с заметно меньшими экспозициями к факторам. Цена — часть альфы (TC ниже 1), выигрыш
— меньшая оборачиваемость и зависимость от нескольких имён.

### 5.3.11. `OptimalHoldingsStrictFactor`: приближение к целевым весам

```python
class OptimalHoldingsStrictFactor(OptimalHoldings):
    def _get_obj(self, weights, alpha_vector):
        assert len(alpha_vector.columns) == 1
        # целевые веса x* — стандартизованный альфа-вектор (сумма 0, плечо 1)
        alpha_vector_norm = ((alpha_vector.values - alpha_vector.values.mean())
                             / np.sum(np.abs(alpha_vector.values))).flatten()
        # min ||x - x*||_2
        return cvx.Minimize(cvx.norm(weights - alpha_vector_norm))

optimal_weights_2 = OptimalHoldingsStrictFactor(
    weights_max=0.02, weights_min=-0.02,
    risk_cap=0.0015,
    factor_max=0.015, factor_min=-0.015).find(
        alpha_vector, risk_model['factor_betas'],
        risk_model['factor_cov_matrix'], risk_model['idiosyncratic_var_vector'])
```

Третья формулировка меняет цель: не «максимум альфы», а «**как можно ближе к целевому
портфелю** $\mathbf{x}^*$» при соблюдении ограничений. Целевой портфель — стандартизованный
альфа-вектор (глава 4, §4.2.2); вместо него подойдёт любой другой, например квантильный
(длинные позиции в Q5, короткие в Q1). Ограничения **жёсткие**: позиции ±2 %, экспозиции
±0,015, лимит риска 0,0015 (дисперсия $\le 0{,}0015^2$ — годовая волатильность 0,15 %,
крайне малая), поэтому оптимизатор сильно отклоняется от $\mathbf{x}^*$ ради нейтральности
к факторам. Результат — равномерно диверсифицированный портфель с экспозициями около нуля.

Три класса — три способа контролировать концентрацию: ограничением (лимиты позиций),
штрафом (регуляризация) и целью (расстояние до диверсифицированного эталона); на практике
их комбинируют, подбирая параметры по TC, оборачиваемости и предсказанному риску.

## 5.4. Дополнительные темы из квизов

### 5.4.1. Геометрия ограничений (`Advanced_Opt`)

Квиз решает ту же задачу для трёх акций, чтобы всё можно было нарисовать в 3D.
Альфа-вектор — моментум за 252 дня, ранжированный и приведённый к z-оценке:
$(0;\ 1{,}22;\ -1{,}22)$ для A, B, C. Без ограничений `cvx.Minimize(-alpha @ x)` даёт
`UNBOUNDED`: цель убывает бесконечно при росте длинной позиции в B и короткой в C. Далее
строится двухфакторная PCA-модель (глава 4, §4.4.3) и визуализируется каждое ограничение:
эллипсоид риска, октаэдр $\lVert \mathbf{x} \rVert_1 \le 1$, плоскость $\sum x_i = 0$, пары
плоскостей $\pm 0{,}1$ для экспозиций и куб $\pm 0{,}55$ для весов. Допустимая область — их
пересечение; оптимум — её точка, наиболее удалённая в направлении $\boldsymbol{\alpha}$:
$(\approx 0;\ 0{,}5;\ -0{,}5)$ — капитал поделён между лучшей и худшей акциями, A не участвует
(её альфа нулевая). Код `find_optimal_holdings` квиза повторяет `OptimalHoldings` в
функциональном виде (`f = B.values.T @ x`), но ограничение записано как `risk <= risk_cap` —
лимит на *дисперсию*, тогда как в проекте `risk <= risk_cap**2`; значения параметра
несопоставимы, и при переносе кода легко ошибиться на порядок величины.

### 5.4.2. Регуляризация на двух активах (`Regularization`)

Квиз в два шага: `cvx.Minimize(-alpha @ x)` при `[cvx.norm(x, 1) <= 1]` даёт «угол»
$(0;\ 1)$; `cvx.Minimize(-alpha @ x + l * cvx.norm(x, 2))` с тем же ограничением — внутреннюю
точку $(0{,}000335;\ 0{,}001674)$. Штраф $\ell_2$ делает цель строго выпуклой, оптимум сходит
с границы внутрь, веса распределяются пропорционально альфам. Выбранное
$l = 0{,}509902 \approx \lVert \boldsymbol{\alpha} \rVert_2$ — пограничное: при
$l \ge \lVert \boldsymbol{\alpha} \rVert_2$ оптимум — нуль, поэтому веса едва отличны от нуля.

### 5.4.3. Фактор формы траектории (`regression_against_time`)

```python
from sklearn.linear_model import LinearRegression
from zipline.pipeline.factors import CustomFactor

class RegressionAgainstTime(CustomFactor):
    window_length = 252
    inputs = [USEquityPricing.close]
    outputs = ['beta', 'gamma']   # два выхода из одного compute

    def compute(self, today, assets, out, dependent):
        t1 = np.arange(self.window_length)
        X = np.array([t1, t1 ** 2]).T
        for i in range(len(out)):
            y = dependent[:, i]
            if np.all(np.isfinite(y)):   # регрессия только по полным рядам
                regressor = LinearRegression().fit(X, y)
                out.beta[i], out.gamma[i] = regressor.coef_
            else:
                out.beta[i] = out.gamma[i] = np.nan

beta_factor = RegressionAgainstTime(mask=universe).beta.rank()
gamma_factor = RegressionAgainstTime(mask=universe).gamma.rank()
conditional_factor = (beta_factor * gamma_factor).rank()
```

Пример `outputs`: `out.beta` и `out.gamma` — отдельные массивы длиной в число акций.
Произведение рангов выделяет акции, у которых и наклон, и выпуклость велики (ускоряющиеся
победители), либо оба малы.

## 5.5. Выводы из результатов

**Коэффициенты Шарпа.** Возврат к среднему за 5 дней (1,37) и его сглаженная версия (1,27)
— лучшие; моментум — 1,13; ночной сентимент — 0,12, после сглаживания 0,45. Три фактора
проходят ориентир «около 1», ночной сентимент — нет. Сглаживание подействовало по-разному:
у возврата к среднему Шарп чуть упал (сигнал быстрый, задержка вредит), у ночного
сентимента вырос почти вчетверо (сырой сигнал слишком шумный). Вывод квиза `sharpe_ratio`
подтверждается: сглаживание — не универсальное улучшение, его эффект надо измерять.

**Квантили.** Наблюдения автора проекта:

* **ни один из факторов не строго монотонен** по квинтилям; у всех, кроме сглаженного
  возврата к среднему, акции с самым высоким значением фактора — не лучшие. Повод
  задуматься, что в конструкции факторов приводит к такому эффекту, и продолжить исследование;
* **большая часть доходности приходит с короткой стороны**: отрицательная доходность
  квинтиля 1 велика у всех факторов. Это тревожно: для шорта нужно найти бумагу взаймы, он
  может быть дорогим или недоступным, так что реализуемая доходность меньше бумажной;
* **величина эффекта**: спред Q1 − Q5 порядка 3 базисных пунктов в день (≈ 0,03 %); при
  252 днях это примерно **7,56 % годовых до всех издержек** (транзакционных и на шорт),
  которые могут съесть половину. Вывод автора: такие альфы могут жить только в
  институциональной среде, и для привлекательной доходности придётся применять плечо.

**Сглаживать ли моментум?** Ответ автора: нет, у моментума и так самая высокая
автокорреляция рангов (годовое окно меняется на 1/252 в день), и дальнейшее сглаживание
почти ничего не изменит. Сглаживание имеет смысл для быстрых и шумных сигналов.

**Оптимизация.** Три постановки дают три портфеля из одного альфа-вектора:
концентрированный, диверсифицированный ($\lambda = 5$) и равномерный, нейтральный к
факторам. Универсального «правильного» нет: выбор определяется мандатом (плечо, лимиты
концентрации), стоимостью сделок и доверием к альфе.

**Ошибка интеграционного теста.** При проверке `_get_constraints` тест проекта упал:
решатель выдал `[-0.01095332, 0.00275889, 0.02684955, -0.01865511]` против ожидаемых
`[-0.01095207, 0.0027576, 0.02684978, -0.01865519]` — расхождение в 6-м знаке. Логика
ограничений верна (`_get_obj` и оба других класса тесты прошли); различие — в допуске
численного решателя: иная версия ECOS или другие настройки точности дают чуть иную точку
останова итераций метода внутренней точки. Тест сравнивал результаты с жёстким допуском, не
рассчитанным на такие различия. Урок: результаты выпуклой оптимизации воспроизводимы лишь до
точности решателя, и сравнивать их надо через `np.allclose(..., atol=1e-4)`, а не поэлементно.

## 5.6. Типичные ошибки и ловушки

⚠️ **Решатели и `max_iters`.** `prob.solve(max_iters=500)` в проекте — опция конкретных
решателей (ECOS, SCS). В cvxpy 1.5+ решатель по умолчанию для конических задач сменился
(ECOS → Clarabel), и `max_iters` без `solver=` может вызвать ошибку или быть молча
проигнорирован. Указывайте решатель явно: `solver=cvx.ECOS` для задач с нормами и
квадратичными формами (как в ноутбуках курса), `cvx.OSQP` для чисто квадратичных задач,
`cvx.SCS` для больших задач с невысокой точностью. Разные решатели дают разные решения в
5–7-м знаке — см. историю с тестом выше.

⚠️ **`*` против `@` в cvxpy.** Старый код `alpha.T.values * weights` в современном cvxpy
выдаёт предупреждение, а в будущих версиях станет поэлементным умножением с
широковещанием — задача тихо изменится. Используйте `@` для матричного произведения и
`cvx.multiply` для поэлементного.

⚠️ **Знак у возврата к среднему.** Забытый минус превращает фактор в краткосрочный моментум
с противоположным знаком Шарпа. Правило: «высокое значение фактора ⇒ ожидаем высокую доходность».

⚠️ **Начальная дата конвейера в неторговый день.** Pipeline не принимает `start_date` без
сессии; проект отступает на 2 дня, надёжнее брать ближайшую сессию из календаря.

⚠️ **`index.levels[1]` для списка акций.** После фильтрации `MultiIndex.levels` может
содержать значения, которых в данных уже нет; используйте `get_level_values(1).unique()`.
Аналогично не путайте `clean_factor_data` (даты; для `factor_returns`) и `unixt_factor_data`
(unix-время; для квантилей и FRA) — иначе ошибка типа или нечитаемая ось времени.

⚠️ **Шарп ≈ 1,3 без издержек — не повод для радости, а два года истории — не
доказательство.** Доходность портфеля с плечом 1 — около 7–8 % годовых до издержек,
большая часть — с короткой стороны; реальная стратегия потребует плеча, стоимости
заимствования бумаг и модели издержек (глава 9). Пять факторов, ручной отбор трёх из них по
Шарпу на тех же данных, где они придуманы, — классический сценарий оверфиттинга (глава 9).
Результаты главы — демонстрация процедуры, а не доказательство существования альфы.

## 5.7. Вопросы для самопроверки

1. Зачем в цепочке `demean(groupby=sector) → rank() → zscore()` нужен каждый шаг? Что
   произойдёт с портфелем, если пропустить центрирование по сектору в год, когда один
   сектор вырос на 40 %?
2. Почему у фактора возврата к среднему стоит знак минус, а у моментума — нет? Как
   проверить правильность знака по графику факторных доходностей?
3. Опишите контракт метода `compute` класса `CustomFactor`: что означают `today`, `assets`,
   `out`, `*inputs`, и почему результат записывается в `out[:]`, а не возвращается? Для
   `CTO` с `window_length=2` — какие элементы `opens` и `closes` нужны и почему `opens[-1]`
   предпочтительнее `opens[1]`?
4. Что измеряет автокорреляция рангов фактора и почему она служит прокси оборачиваемости?
   Какой фактор проекта имеет самую высокую FRA и почему?
5. Шарп сглаженного ночного сентимента — 0,45 против 0,12 у несглаженного; сглаженный
   возврат к среднему — 1,27 против 1,37. Объясните разнонаправленный эффект сглаживания.
6. Спред Q1 − Q5 равен 3 б. п. в день. Какова годовая доходность долларово-нейтрального
   портфеля с плечом 1? С плечом 4? Какие издержки надо вычесть?
7. Запишите задачу `OptimalHoldings` формально и объясните, какие из ограничений делают
   задачу ограниченной (без них решатель вернёт `UNBOUNDED`).
8. Почему штраф $\lambda\lVert \mathbf{x} \rVert_2$ в целевой функции диверсифицирует
   портфель, а $\lambda\lVert \mathbf{x} \rVert_1$ — нет? (Подсказка: сравните нормы
   векторов $(1, 0)$ и $(0{,}5;\ 0{,}5)$.)
9. Что такое коэффициент переноса? Ранжируйте три оптимизатора проекта по ожидаемому TC и
   объясните порядок. Почему интеграционный тест `_get_constraints` упал на 6-м знаке и
   какой допуск сравнения был бы разумным?

## 5.8. Файлы репозитория и что читать дальше

📁 `Projects/4-Multi-factor-Model/project_4_starter.pdf` (и `.html`) — проект 4, части
«Create Alpha Factors», «Evaluate Alpha Factors», «Optimal Portfolio Constrained by Risk Model».

📁 `Quiz/m4_multifactor_models/m4l1/zipline_coding_exercises.ipynb`,
📁 `Quiz/m4_multifactor_models/Zipline-Pipeline/Zipline Pipeline.ipynb` — введение в Pipeline.

📁 `Quiz/m4_multifactor_models/m4l3/` — `sector_neutral`, `rank`, `zscore`, `smoothing`
(конвейер преобразований); `overnight_returns` (`CTO`, цитаты Aboody et al.),
`regression_against_time` (`outputs`, фактор формы траектории); `clean_forward_returns`,
`factor_returns`, `quantiles`, `turnover`, `sharpe_ratio`, `rank_ic`, `transfer_coefficient`
(метрики Alphalens и TC) — все с суффиксом `_solution.ipynb`; `quiz_helper.py` — `Sector`,
`build_pipeline_engine`, `get_pricing`, `get_factor_exposures`.

📁 `Quiz/m4_multifactor_models/Advanced_Opt/Advanced_Opt_solution.ipynb` — геометрия
ограничений на трёх акциях; 📁 `Quiz/m4_multifactor_models/Regularization.ipynb` —
регуляризация на двух активах.

Что читать дальше: главы 6 и 7 добавляют новые источники альфы (текст отчётов 10-K и
сообщения StockTwits), глава 8 заменяет усреднение факторов моделью машинного обучения,
глава 9 добавляет к оптимизации транзакционные издержки и проводит честный бэктест.

---

Навигация: [← Глава 4. Факторные модели: альфа-факторы, риск-факторы и PCA-модель риска](04-faktornye-modeli-riska.md) | [Глава 6. NLP на финансовых отчётах: сентимент из 10-K →](06-nlp-finansovye-otchety.md)
