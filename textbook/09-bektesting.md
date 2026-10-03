# Глава 9. Бэктестинг: транзакционные издержки, оптимизация и атрибуция PnL

> Чему научитесь: отличать честный бэктест от самообмана; работать с данными Barra; оценивать
> факторные доходности кросс-секционной регрессией; строить целевую функцию с риском,
> альфой и транзакционными издержками и минимизировать её L-BFGS за секунды на 2000+ акций;
> раскладывать PnL на альфу, риск и издержки. Модуль M8 *Backtesting*, проект
> 📁 `Projects/8-Backtesting/project_8_starter.ipynb`.

## 9.1. Мотивация и гипотеза

Предыдущие главы заканчивались оценкой альфа-факторов в Alphalens — оценкой *сигнала*, а не
*стратегии*. Между сигналом и прибылью стоят оптимизатор, риск-модель, издержки исполнения и
время: данные приходят вечером, торговать можно завтра, доходность реализуется послезавтра.
Бэктест (backtest) — симуляция всей этой цепочки на истории, день за днём.

Конспект `Intro.md` формулирует главный экономический тезис модуля — об **овертрейдинге**
(overtrading): «если ожидаемый мисприцинг составляет 10 базисных пунктов, ни одна модель не
должна торговать так, чтобы рыночное влияние составило 15 б. п. — это не может быть
устойчиво прибыльным». Размер позиций определяется не только силой сигнала, но и
ликвидностью: чем крупнее книга, тем дороже каждая сделка. Поэтому позиции здесь — **в долларах**.

Гипотеза проекта 8: четыре стилевых фактора Barra (однодневный разворот, доходность по
прибыли, стоимость, сентимент), объединённые равновесно и пропущенные через оптимизатор с
риск-моделью Barra и квадратичной моделью издержек, дают положительный PnL за 2004 год, а
атрибуция покажет, какая его часть — альфа, какая — риск-экспозиции и сколько съели издержки.

## 9.2. Теория

### 9.2.1. Валидность бэктеста

Бэктест валиден, если на каждый день $t$ использует только известное к концу дня $t$ и
торгует по ценам, по которым реально можно было торговать. Типичные нарушения:

* **Заглядывание в будущее** (lookahead bias): доходность дня $t$ при формировании
  портфеля на день $t$; экспозиции по данным, опубликованным позже. В проекте защита
  механическая — столбец `DlyReturn` **удаляется** из универсума до оптимизации.
* **Ошибка выжившего** (survivorship bias): универсум из бумаг, доживших до конца периода.
  Данные Barra содержат все бумаги, существовавшие в каждый день.
* **Нереалистичные задержки.** Реалистичная цепочка: данные за день $t$ → оптимизация
  вечером $t$ → исполнение в течение $t+1$ → первая доходность новых позиций — за день
  $t+2$. Отсюда **сдвиг на 2 дня** между `DataDate` и `DlyReturnDate`.
* **Издержки и ликвидность.** Бэктест без издержек предпочитает высокооборотные стратегии
  (вроде однодневного разворота), которые в жизни убыточны.
* **Структурные изменения** (`Intro.md`): рынок 2004–2008 годов — не рынок 2020-х. Бэктест,
  прибыльный только в одном режиме, — ставка на возвращение этого режима.

### 9.2.2. Оверфиттинг бэктеста

Если перебрать достаточно много стратегий, одна покажет отличный бэктест случайно. Это
**множественное тестирование**: при $N$ независимых испытаниях ожидаемый максимум
коэффициента Шарпа растёт примерно как $\sqrt{2 \ln N}$ даже при нулевой истинной альфе.
Идея **deflated Sharpe ratio** (Бэйли и Лопес де Прадо): наблюдаемый Шарп сравнивают не с
нулём, а с максимумом, который даёт шум при данном числе попыток, длине истории и
ненормальности доходностей. Конспект отсылает к демонстрации на `datagrid.lbl.gov/backtest`.

Практические правила: фиксировать гипотезу **до** бэктеста; держать тестовый период
нетронутым до конца; менять параметры на порядок, а не на единицы (глава 8); считать число
испытаний; предпочитать простые модели; проверять устойчивость к периоду, универсуму и модели
издержек; смотреть на атрибуцию, а не только на итоговую кривую (переобученный бэктест — § 9.4.2).

### 9.2.3. Данные Barra

Barra (MSCI Barra) — отраслевой стандарт факторных риск-моделей: для каждой бумаги на
каждый день даются **экспозиции** к стилевым и отраслевым факторам, специфический риск и
ковариационная матрица факторов. Проект использует модель *US Fast* (префикс `USFASTD_`),
2003–2008 годы, заранее разобранные в pickle: `pandas-frames.<год>.pickle` (экспозиции и
атрибуты), `covariance.<год>.pickle` (ковариации в длинном формате `Factor1, Factor2,
VarCovar`), `price.<год>.pickle` (дневные доходности); разбор сырых файлов медленный.

**Стилевые факторы** (определения из квиза `optimization_with_tcosts`, по документации Barra):

| Фактор | Что описывает |
|---|---|
| Beta | рыночный риск, не объяснённый страновым фактором; регрессия избыточных доходностей акции на рынок; обычно самый важный стилевой фактор |
| 1-Day Reversal (`1DREVRSL`) | однодневный разворот |
| Dividend Yield | исторический и прогнозный дивиденд к цене |
| Downside Risk | максимальная просадка |
| Earnings Quality | начисления (accruals) в составе прибыли |
| Earnings Yield (`EARNYILD`) | прибыль к цене; главный дескриптор — прогноз прибыли на 12 месяцев к цене; сильный сигнал стоимости |
| Growth | прогноз долгосрочного роста прибыли, рост выручки и прибыли за пять лет |
| Leverage | рыночное и балансовое плечо, долг к активам |
| Liquidity | доля акций в обращении, проторгованная за окно |
| Long-Term Reversal | поведение цены за пять лет без последних 13 месяцев |
| Management Quality | рост активов, числа акций и капзатрат (наклоны регрессий на время за пять лет); капзатраты к среднему |
| Mid Capitalization | куб экспозиции Size, ортогонализованный к Size; «штанга» long mid-cap / short small- и large-cap |
| Momentum | доходность за 12 месяцев без последних (исключить краткосрочный разворот); часто второй по силе |
| Profitability | эффективность операций и активов |
| Residual Volatility | волатильность избыточных и остаточных доходностей, годовой диапазон цены; ортогонализован к Beta и Size |
| Seasonality | сезонность |
| Sentiment (`SENTMT`) | сентимент (пересмотры прогнозов аналитиков) |
| Size | логарифм капитализации |
| Short-Term Reversal | краткосрочный разворот |
| Value (`VALUE`) | стоимость |
| Prospect | функция асимметрии и максимальной просадки |

**Отраслевые факторы** — one-hot-признаки отрасли: аэрокосмос и оборона, авиалинии, алюминий
и сталь, одежда, автомобили, банки, напитки и табак, биотехнологии, стройматериалы, химия,
строительство, машиностроение, электроника, конгломераты, упаковка, финансы, энергетика,
продукты питания, ритейл, газ и утилиты, медоборудование, домостроение, страхование, отдых,
нефть и газ, бумага, фармацевтика, драгметаллы, недвижимость, рестораны, дороги, полупроводники,
софт, телекоммуникации, транспорт, беспроводная связь, группы `SPTY*`/`SPLTY*`. Всего 81
фактор `USFASTD_*`; четыре проект использует как альфа-факторы, остальные 77 — как риск-факторы.

**Единицы измерения.** Из квиза: всегда проверяйте единицы, при необходимости сверкой одной
бумаги за день с внешним источником. `SpecRisk` дан в **процентах** (порядка 9–12), ковариации
— в процентах в квадрате: $S_i = (0{,}01 \cdot \mathrm{SpecRisk}_i)^2$,
$F_{jj} = 10^{-4} \cdot \mathrm{VarCovar}_{jj}$. `ADTCA_30` — средний объём за 30 дней.

### 9.2.4. Факторные доходности из кросс-секционной регрессии

Факторная модель главы 4: $r_{i,t} = \sum_{j=1}^{k} \beta_{i,j,t-2}\, f_{j,t} + s_{i,t}$,
где $r_{i,t}$ — доходность бумаги $i$ за день $t$, $\beta_{i,j,t-2}$ — её экспозиция к
фактору $j$, известная за два дня до этого, $f_{j,t}$ — факторная доходность, $s_{i,t}$ —
специфическая доходность. Barra даёт $\beta$, рынок — $r$; неизвестны $f$. Для каждого дня
это линейная регрессия по кросс-секции: $N \approx 2000$ наблюдений, $k = 81$ регрессор,
без свободного члена (`~ 0 + ...` в формуле patsy), коэффициенты — $f_{j,t}$.

Регрессия оценивается на **универсуме оценки** с капитализацией выше 1 млрд долларов — так
факторные доходности отражают торгуемые бумаги (близко к Russell 3000). Доходности перед
регрессией **винзоризируются** (winsorize) на $\pm 25\%$: один сплит или ошибка в данных не
должны определять факторную доходность дня. Ridge или lasso дали бы близкие оценки.

### 9.2.5. Позиции в долларах, альфа в базисных пунктах, модель издержек

Из `optimization.md`: размер книги влияет на стоимость исполнения, а та — на то, какие альфы
вообще работают. Поэтому вместо весов $x$ (главы 3–5) оптимизируем **позиции (holdings)**
$\mathbf{h}$ в долларах. Чтобы $\boldsymbol{\alpha}^T \mathbf{h}$ был ожидаемым доходом в
долларах, $\alpha_i$ должен быть ожидаемой *дневной доходностью*. Экспозиции Barra лежат в
$[-5, 5]$; курс принимает допущение **«экспозиция 1 ↔ 1 базисный пункт дневной доходности»**
и умножает сумму альфа-экспозиций на $10^{-4}$.

**Транзакционные издержки** институционального трейдера — в основном **рыночное влияние**
(market impact): покупка толкает цену вверх, продажа — вниз. Бенчмарк — **мидпоинт** книги
лимитных заявок; разница между ценой исполнения и мидпоинтом — **проскальзывание** (slippage),
сумма временного и постоянного влияния. Допущение курса: **1 % ADV сдвигает цену на 10 б. п.**:

$$
\%\Delta P_{i,t} = \frac{10^{-3}}{10^{-2}} \cdot \frac{h_{i,t} - h_{i,t-1}}{\mathrm{ADV}_{i,t}}
= \lambda_{i,t}\,(h_{i,t} - h_{i,t-1}), \qquad \lambda_{i,t} = \frac{0{,}1}{\mathrm{ADV}_{i,t}},
$$

а издержки — сдвиг цены, умноженный на объём сделки:

$$
\mathrm{tcost}_t = \sum_{i=1}^{N} \lambda_{i,t}\,(h_{i,t} - h_{i,t-1})^2
= (\mathbf{h}_t - \mathbf{h}_{t-1})^T \boldsymbol{\Lambda}_t (\mathbf{h}_t - \mathbf{h}_{t-1}),
$$

где $\boldsymbol{\Lambda}_t = \mathrm{diag}(\lambda_{i,t})$, $\mathrm{ADV}_{i,t}$ — средний
дневной объём в долларах (среднее за 30 дней стабильнее объёма одного дня). Издержки
**квадратичны** по сделке: удвоение сделки учетверяет издержки — математическое выражение
тезиса об овертрейдинге. Если ADV отсутствует или равен нулю, бумага считается неликвидной:
ADV := 10 000 долларов, $\lambda$ огромна, оптимизатор её не торгует. Статья *Crossover from
Linear to Square-Root Market Impact* (`project.md`) обсуждает модели с корнем из объёма.

### 9.2.6. Целевая функция без ограничений и её градиент

$$
f(\mathbf{h}) = \tfrac{1}{2}\kappa\, \mathbf{h}^T \mathbf{Q}^T \mathbf{Q}\, \mathbf{h}
+ \tfrac{1}{2}\kappa\, \mathbf{h}^T \mathbf{S}\, \mathbf{h}
- \boldsymbol{\alpha}^T \mathbf{h}
+ (\mathbf{h} - \mathbf{h}_0)^T \boldsymbol{\Lambda} (\mathbf{h} - \mathbf{h}_0),
$$

где $\kappa$ — коэффициент **неприятия риска** (risk aversion), $\mathbf{S}$ — диагональная
матрица специфических дисперсий, $\boldsymbol{\alpha}$ — альфа-вектор в долях дневной
доходности, $\mathbf{h}_0 = \mathbf{h}_{t-1}$ — вчерашние позиции, $\mathbf{Q}$ — матрица с
$\mathbf{Q}^T\mathbf{Q} = \mathbf{B}\mathbf{F}\mathbf{B}^T$. Четыре члена: факторный риск +
специфический риск − ожидаемый доход + издержки; все в долларах (дисперсия — в долларах в
квадрате, поэтому $\kappa$ измеряется в обратных долларах).

**Зачем $\mathbf{Q}$.** Матрица $\mathbf{B}\mathbf{F}\mathbf{B}^T$ имеет размер $N \times N$
(2000 × 2000 — 32 МБ, пересчитываемые каждый день). Пусть $\mathbf{G} = \mathbf{F}^{1/2}$ —
матричный корень ($\mathbf{G}\mathbf{G} = \mathbf{F}$). Тогда
$\mathbf{B}\mathbf{F}\mathbf{B}^T = \mathbf{B}\mathbf{G}\mathbf{G}\mathbf{B}^T = \mathbf{Q}^T\mathbf{Q}$
при $\mathbf{Q} = \mathbf{G}\mathbf{B}^T$ размера $k \times N$, и
$\mathbf{h}^T\mathbf{Q}^T\mathbf{Q}\mathbf{h} = \|\mathbf{Q}\mathbf{h}\|^2$ считается через
вектор длины $k = 77$. $\mathbf{Q}$ не зависит от $\mathbf{h}$ и вычисляется один раз в
день; $\mathbf{R} = \mathbf{Q}\mathbf{h}$ — внутри целевой функции при каждом вызове. Из
квиза: **никогда не умножайте $\mathbf{Q}^T\mathbf{Q}$ явно.**

**Диагональная $\mathbf{F}$.** Квиз предлагает оставить только дисперсии факторов, обнулив
ковариации: корреляции факторов шумны; оптимизатор и так снижает экспозиции к каждому фактору,
а при малых экспозициях их ковариация почти не важна; корень $\mathbf{G}$ становится
тривиальным. В данных Barra часть ковариаций отсутствует — ещё один повод.

**Ограничения как члены целевой функции.** В главах 3 и 5 рыночную нейтральность, лимит
позиции и диверсификацию задавали ограничениями `cvxpy`. Здесь ограничений нет, и из
`optimization.md`: ту же роль играют члены функции. Нейтральность — факторный риск
(экспозиция к Beta и отраслям штрафуется); размер позиций и диверсификация — специфический
риск ($h_i^2 S_i$ растёт квадратично с концентрацией) и издержки (большая позиция требует
большой сделки).

**Градиент.** `cvxpy` считал его сам, теперь — вручную. Для симметричной $\mathbf{A}$
$\partial(\mathbf{h}^T\mathbf{A}\mathbf{h})/\partial\mathbf{h} = 2\mathbf{A}\mathbf{h}$, откуда

$$
\nabla f(\mathbf{h}) = \kappa\, \mathbf{Q}^T (\mathbf{Q}\mathbf{h}) + \kappa\, \mathbf{S}\mathbf{h}
- \boldsymbol{\alpha} + 2\boldsymbol{\Lambda}(\mathbf{h} - \mathbf{h}_0)
$$

— вектор длины $N$. Половинки при риск-членах сокращаются с двойкой от дифференцирования;
$\mathbf{S}\mathbf{h}$ и $\boldsymbol{\Lambda}(\cdot)$ — поэлементные произведения, так как матрицы диагональны.

**Неприятие риска $\kappa$.** Из квиза: $\kappa$ подбирают под целевую валовую стоимость
позиций (GMV) или волатильность. Начинающий квант держит книгу порядка 50 млн долларов;
$\kappa = 10^{-6}$ даёт GMV в десятках миллионов (зависимость нелинейная). $\kappa$ держат
постоянным и меняют только при притоке или оттоке капитала: ежедневная подстройка породила
бы торговлю, не связанную с альфой.

**Оптимизаторы.** Проект использует `scipy.optimize.fmin_l_bfgs_b` — L-BFGS-B,
квазиньютоновский метод с ограниченной памятью, масштабируемый на тысячи переменных.
Конспект советует сравнить с Powell, Nelder–Mead (без градиента) и сопряжёнными градиентами.

### 9.2.7. Атрибуция PnL

Из `Attribution.md`: экспозиция портфеля к факторам — $\mathbf{b} = \mathbf{B}^T\mathbf{h}^*$
(долларов на единицу фактора), и PnL раскладывается на составляющие:

$$
\mathrm{PnL}_t = \sum_i h_i^* r_{i,t}, \qquad
\mathrm{PnL}^{\alpha}_t = \mathbf{f}_t^T \mathbf{b}_{\alpha}, \qquad
\mathrm{PnL}^{\mathrm{risk}}_t = \mathbf{f}_t^T \mathbf{b}_{\mathrm{risk}}, \qquad
\mathrm{cost}_t = (\mathbf{h}^* - \mathbf{h}_0)^T \boldsymbol{\Lambda} (\mathbf{h}^* - \mathbf{h}_0),
$$

где $\mathbf{f}_t$ — факторные доходности дня (§ 9.2.4), а
$\mathbf{b}_\alpha = \mathbf{B}_\alpha^T \mathbf{h}^*$ и $\mathbf{b}_{\mathrm{risk}} = \mathbf{B}^T \mathbf{h}^*$
— экспозиции к альфа- и риск-факторам. Остаток — специфический PnL. Хорошая стратегия
зарабатывает в основном $\mathrm{PnL}^\alpha$; большой $\mathrm{PnL}^{\mathrm{risk}}$ любого
знака — непреднамеренные ставки, которые риск-модель не подавила. Так же раскладывается
дисперсия PnL — на факторную и специфическую.

## 9.3. Разбор проекта шаг за шагом

Код приведён к pandas 2.x; медиана берётся методом `.median()` вместо модуля `statistics`;
исправлено присваивание в копию (§ 9.3.3).

### 9.3.1. Загрузка и сдвиг дат

```python
import pickle, scipy, scipy.optimize, scipy.linalg, patsy
import numpy as np, pandas as pd
from statsmodels.formula.api import ols
from tqdm import tqdm

barra_dir = '../../data/project_8_barra/'
data, covariance, daily_return = {}, {}, {}
for year in [2004]:
    data.update(pickle.load(open(f'{barra_dir}pandas-frames.{year}.pickle', 'rb')))
    covariance.update(pickle.load(open(f'{barra_dir}covariance.{year}.pickle', 'rb')))
for year in [2004, 2005]:      # доходности нужны и за первые дни следующего года
    daily_return.update(pickle.load(open(f'{barra_dir}price.{year}.pickle', 'rb')))

# сдвиг на 2 торговых дня: данные дня t ↔ доходность дня t+2
dlyreturn_n_days_delay = 2
date_shifts = zip(
    sorted(data.keys()),
    sorted(daily_return.keys())[dlyreturn_n_days_delay:len(data) + dlyreturn_n_days_delay])

frames = {}
for data_date, price_date in date_shifts:
    frames[price_date] = data[data_date].merge(daily_return[price_date], on='Barrid')
    frames[price_date]['DlyReturnDate'] = price_date   # скаляр расширяется на все строки

my_dates = sorted(pd.to_datetime(date, format='%Y%m%d') for date in frames)
```

Ключ словаря `frames` — **дата доходности**: PnL отчитывают на дату реализации. Поле
`DataDate` хранит дату экспозиций (на два дня раньше); по нему ищется ковариация. Доходности
загружаются и за 2005 год: последние дни 2004 года со сдвигом попадают в январь. В проекте
`DlyReturnDate` создавался через `pd.Series` той же длины — это работает только при
`RangeIndex`; присваивание скаляра надёжнее.

### 9.3.2. Винзоризация и оценка факторных доходностей

```python
def wins(x, a, b):
    """Винзоризация: обрезать значения по границам [a, b]."""
    return np.clip(x, a, b)


def get_formula(factors, Y):
    return Y + ' ~ 0 + ' + ' + '.join(factors)       # без свободного члена


def factors_from_names(n):
    return [x for x in n if 'USFASTD_' in x]


def estimate_factor_returns(df):
    """Кросс-секционная OLS: доходности ~ экспозиции на универсуме cap > 1 млрд долл."""
    estu = df.loc[df.IssuerMarketCap > 1e9].copy(deep=True)
    estu['DlyReturn'] = wins(estu['DlyReturn'], -0.25, 0.25)
    form = get_formula(factors_from_names(list(df)), 'DlyReturn')
    return ols(form, data=estu).fit()


facret = {date: estimate_factor_returns(frames[date]).params for date in frames}

alpha_factors = ['USFASTD_1DREVRSL', 'USFASTD_EARNYILD', 'USFASTD_VALUE', 'USFASTD_SENTMT']
facret_df = pd.DataFrame({dt: facret[dt.strftime('%Y%m%d')] for dt in my_dates}).T
facret_df[alpha_factors].cumsum().plot(xlabel='Date', ylabel='Cumulative Factor Returns')
```

`wins` в проекте написана через два вложенных `np.where`; `np.clip` делает то же. `ols`
принимает формулу patsy `DlyReturn ~ 0 + USFASTD_1DREVRSL + ...`; `.params` — вектор
$\mathbf{f}_t$ с именами факторов. Сбор в `DataFrame` в проекте — двойной цикл с `.at`.

### 9.3.3. Очистка, специфический риск и универсум

```python
def clean_nas(df):
    """NaN → 0 во всех числовых столбцах."""
    for numeric_column in df.select_dtypes(include=[np.number]).columns:
        df[numeric_column] = np.nan_to_num(df[numeric_column])
    return df


previous_holdings = pd.DataFrame({'Barrid': ['USA02P1'], 'h.opt.previous': np.array(0)})
df = frames[my_dates[0].strftime('%Y%m%d')]
df = df.merge(previous_holdings, how='left', on='Barrid')
df = clean_nas(df)
# нулевой SpecRisk — скорее пропуск, чем ноль: заменяем медианой
df.loc[df['SpecRisk'] == 0, 'SpecRisk'] = df['SpecRisk'].median()


def get_universe(df):
    """Универсум: cap ≥ 1 млрд долл. ИЛИ есть вчерашняя позиция; без столбца доходности."""
    universe = df.loc[(df['IssuerMarketCap'] >= 1e9) | (abs(df['h.opt.previous']) > 0)].copy()
    return universe.drop(columns='DlyReturn')


universe = get_universe(df)
date = str(int(universe['DataDate'].iloc[0]))     # дата экспозиций — ключ к ковариациям
```

⚠️ **Ошибка оригинала.** В проекте и в квизе строка выглядела так:
`df.loc[df['SpecRisk'] == 0]['SpecRisk'] = median(df['SpecRisk'])`. Это **chained
assignment**: `df.loc[mask]` возвращает копию, в неё записывается медиана, копия
выбрасывается, `df` **не меняется** — нули остаются, и оптимизатор считает такие бумаги
безрисковыми (в pandas 2.x с Copy-on-Write — ещё и `ChainedAssignmentError`). Правильно — одна
операция `.loc` с маской строк и именем столбца: `df.loc[df['SpecRisk'] == 0, 'SpecRisk'] = df['SpecRisk'].median()` (B §B.3).

Почему в универсум входят бумаги с существующей позицией, даже если капитализация упала ниже
порога? Ответ квиза: иначе бумага не попадёт в оптимизацию, получит позицию ноль — и бэктест
молча продаст всё в день пересечения порога; позиция должна закрываться оптимизатором
постепенно, с учётом издержек. Начальный портфель — одна бумага с нулевой позицией: иначе все
инициализированные бумаги навсегда «зависнут» в универсуме. Ещё правка: `universe['DataDate'][1]`
индексирует по **метке** 1, которой после `.loc[...]` может не быть; `.iloc[0]` — по позиции.

### 9.3.4. Матрицы экспозиций и факторная ковариация

```python
def setdiff(superset, subset):
    s = set(subset)
    return [x for x in superset if x not in s]


def model_matrix(formula, data):
    outcome, predictors = patsy.dmatrices(formula, data)
    return predictors                   # outcome (SpecRisk) не нужен — он тут формально


def colnames(B):
    return B.design_info.column_names   # имена столбцов DesignMatrix (у DataFrame — B.columns)


all_factors = factors_from_names(list(universe))          # 81
risk_factors = setdiff(all_factors, alpha_factors)         # 77
h0 = np.asarray(universe['h.opt.previous'])                # 2265 бумаг в первый день

B = model_matrix(get_formula(risk_factors, 'SpecRisk'), universe)   # (2265, 77)
BT = B.transpose()
specVar = (0.01 * universe['SpecRisk']) ** 2                        # проценты → доли, квадрат


def get_cov(cv, factor1, factor2):
    try:
        return cv.loc[(cv.Factor1 == factor1) & (cv.Factor2 == factor2), 'VarCovar'].iloc[0]
    except IndexError:
        print(f"didn't find covariance for: factor 1: {factor1} factor2: {factor2}")
        return 0


def diagonal_factor_cov(date, B):
    """Диагональная F: только дисперсии факторов в порядке столбцов B, %² → доли²."""
    cv = covariance[date]
    k = np.shape(B)[1]
    Fm = np.zeros([k, k])
    for i in range(k):
        fac = colnames(B)[i]
        Fm[i, i] = (0.01 ** 2) * get_cov(cv, fac, fac)
    return Fm


Fvar = diagonal_factor_cov(date, B)
```

`patsy.dmatrices` строит матрицы регрессии по формуле: слева — «зависимая переменная»
(`SpecRisk`, выбранный просто потому, что это не фактор; `outcome` отбрасывается), справа —
регрессоры; будь категория строкой, patsy сам сделал бы one-hot. Порядок столбцов $\mathbf{B}$
задаёт порядок факторов в $\mathbf{F}$ — отсюда `colnames(B)` внутри `diagonal_factor_cov`.
Полная ковариация (вариант квиза) упирается в отсутствующие пары факторов.

### 9.3.5. Издержки и альфа-вектор

```python
def get_lambda(universe, composite_volume_column='ADTCA_30'):
    """λ_i = 0.1 / ADV_i; нет объёма → считаем бумагу неликвидной (ADV = 1e4)."""
    universe.loc[universe[composite_volume_column].isna(), composite_volume_column] = 1.0e4
    universe.loc[universe[composite_volume_column] == 0, composite_volume_column] = 1.0e4
    return 0.1 / universe[composite_volume_column]


def get_B_alpha(alpha_factors, universe):
    return model_matrix(get_formula(alpha_factors, 'SpecRisk'), universe)   # (N, 4)


def get_alpha_vec(B_alpha):
    """Равновесная комбинация альф; 1e-4 переводит экспозиции в доли дневной доходности."""
    return 1e-4 * np.sum(B_alpha, axis=1)


Lambda = get_lambda(universe)
B_alpha = get_B_alpha(alpha_factors, universe)
alpha_vec = get_alpha_vec(B_alpha)
```

Комбинирование альф здесь — сумма строк $\mathbf{B}_\alpha$. Необязательное задание проекта:
взвешивать альфы по скользящему Шарпу их факторных доходностей (только по данным **до** даты
оптимизации), обрезая отрицательные веса нулём; ещё вариант — AI-альфа из главы 8 (§ 9.4.3).

### 9.3.6. Целевая функция, градиент и оптимизация

```python
risk_aversion = 1.0e-6


def get_obj_func(h0, risk_aversion, Q, specVar, alpha_vec, Lambda):
    def obj_func(h):
        factor_risk = 0.5 * risk_aversion * np.sum((Q @ h) ** 2)       # ||Qh||², без N×N
        idiosyncratic_risk = 0.5 * risk_aversion * np.dot(h ** 2, specVar)
        portfolio_return = np.dot(h, alpha_vec)
        trans_cost = np.dot((h - h0) ** 2, Lambda)
        return factor_risk + idiosyncratic_risk - portfolio_return + trans_cost
    return obj_func


def get_grad_func(h0, risk_aversion, Q, QT, specVar, alpha_vec, Lambda):
    def grad_func(h):
        g = (risk_aversion * (QT @ (Q @ h)) + risk_aversion * specVar * h
             - alpha_vec + 2 * (h - h0) * Lambda)
        return np.asarray(g)
    return grad_func


def get_h_star(risk_aversion, Q, QT, specVar, alpha_vec, h0, Lambda):
    """Минимизировать целевую функцию L-BFGS-B, стартуя с вчерашних позиций."""
    obj_func = get_obj_func(h0, risk_aversion, Q, specVar, alpha_vec, Lambda)
    grad_func = get_grad_func(h0, risk_aversion, Q, QT, specVar, alpha_vec, Lambda)
    optimizer_result = scipy.optimize.fmin_l_bfgs_b(obj_func, h0, fprime=grad_func)
    return optimizer_result[0]


Q = np.matmul(scipy.linalg.sqrtm(Fvar), BT)      # (77, N): G·Bᵀ, один раз в день
QT = Q.transpose()
```

Порядок умножений в градиенте: `QT @ (Q @ h)` — сначала вектор длины 77, затем длины $N$;
`(QT @ Q) @ h` построил бы матрицу $N \times N$, как и `np.diag(specVar)`. Начальная точка —
$\mathbf{h}_0$: при малой альфе и больших издержках оптимум близок к вчерашним позициям.

💡 `scipy.linalg.sqrtm` на диагональной матрице — лишняя работа и источник комплексного
`dtype`; `np.diag(np.sqrt(np.diag(Fvar)))` быстрее. `fmin_l_bfgs_b` — унаследованный
интерфейс; современный эквивалент — `scipy.optimize.minimize(obj_func, h0, jac=grad_func,
method='L-BFGS-B').x`.

### 9.3.7. Экспозиции, издержки и `form_optimal_portfolio`

```python
def get_risk_exposures(B, BT, h_star):
    return pd.Series(BT @ h_star, index=colnames(B))                # Bᵀh*, 77 чисел в долларах


def get_portfolio_alpha_exposure(B_alpha, h_star):
    return pd.Series(B_alpha.T @ h_star, index=colnames(B_alpha))   # B_αᵀh*, 4 числа


def get_total_transaction_costs(h0, h_star, Lambda):
    return np.dot((h_star - h0) ** 2, Lambda)


def form_optimal_portfolio(df, previous, risk_aversion):
    """Один день бэктеста: очистка → универсум → матрицы → оптимизация → экспозиции."""
    df = clean_nas(df.merge(previous, how='left', on='Barrid'))
    df.loc[df['SpecRisk'] == 0, 'SpecRisk'] = df['SpecRisk'].median()     # исправлено

    universe = get_universe(df)
    date = str(int(universe['DataDate'].iloc[0]))
    risk_factors = setdiff(factors_from_names(list(universe)), alpha_factors)
    h0 = np.asarray(universe['h.opt.previous'])

    B = model_matrix(get_formula(risk_factors, 'SpecRisk'), universe)
    specVar = (0.01 * universe['SpecRisk']) ** 2
    Lambda = get_lambda(universe)
    B_alpha = get_B_alpha(alpha_factors, universe)
    Q = np.matmul(scipy.linalg.sqrtm(diagonal_factor_cov(date, B)), B.transpose())
    h_star = get_h_star(risk_aversion, Q, Q.transpose(), specVar, get_alpha_vec(B_alpha), h0, Lambda)

    return {
        'opt.portfolio': pd.DataFrame({'Barrid': universe['Barrid'], 'h.opt': h_star}),
        'risk.exposures': get_risk_exposures(B, B.transpose(), h_star),
        'alpha.exposures': get_portfolio_alpha_exposure(B_alpha, h_star),
        'total.cost': get_total_transaction_costs(h0, h_star, Lambda)}
```

Экспозиция портфеля к фактору $j$ — $\sum_i B_{ij} h_i^*$: сколько долларов «ставки» на
фактор несёт книга. У работающего оптимизатора риск-экспозиции малы по сравнению с GMV, а
альфа-экспозиции положительны и велики: мы *хотим* быть длинными в своих альфах.

### 9.3.8. Список сделок и цикл по дням

```python
def build_tradelist(prev_holdings, opt_result):
    # список сделок h_t − h_{t−1}; бумаги, пропавшие из универсума, получают 0
    tmp = prev_holdings.merge(opt_result['opt.portfolio'], how='outer', on='Barrid')
    tmp['h.opt.previous'] = np.nan_to_num(tmp['h.opt.previous'])
    tmp['h.opt'] = np.nan_to_num(tmp['h.opt'])
    return tmp


def convert_to_previous(result):
    return result['opt.portfolio'].rename(columns={'h.opt': 'h.opt.previous'})


trades, port = {}, {}
for dt in tqdm(my_dates, desc='Optimizing Portfolio', unit='day'):
    date = dt.strftime('%Y%m%d')
    result = form_optimal_portfolio(frames[date], previous_holdings, risk_aversion)
    trades[date] = build_tradelist(previous_holdings, result)
    port[date] = result
    previous_holdings = convert_to_previous(result)
```

Цикл по 252 дням 2004 года занял в среде курса около 21 минуты (5 с/день) — приемлемо благодаря
pickle, диагональной $\mathbf{F}$ и $\mathbf{Q}$ вместо $N \times N$. `rename(index=str, ...)` упрощён до `rename(columns=...)`.

### 9.3.9. Атрибуция PnL и характеристики портфеля

```python
def partial_dot_product(v, w):
    """Скалярное произведение двух Series по общим меткам индекса."""
    common = v.index.intersection(w.index)
    return np.sum(v[common] * w[common])


def build_pnl_attribution():
    df = pd.DataFrame(index=my_dates)
    for dt in my_dates:
        date = dt.strftime('%Y%m%d')
        p, fr = port[date], facret[date]
        mf = p['opt.portfolio'].merge(frames[date], how='left', on='Barrid')
        mf['DlyReturn'] = wins(mf['DlyReturn'], -0.5, 0.5)
        df.at[dt, 'daily.pnl'] = np.sum(mf['h.opt'] * mf['DlyReturn'])          # Σ h·r
        df.at[dt, 'attribution.alpha.pnl'] = partial_dot_product(fr, p['alpha.exposures'])
        df.at[dt, 'attribution.risk.pnl'] = partial_dot_product(fr, p['risk.exposures'])
        df.at[dt, 'attribution.cost'] = p['total.cost']
    return df


def build_portfolio_characteristics():
    df = pd.DataFrame(index=my_dates)
    for dt in my_dates:
        date = dt.strftime('%Y%m%d')
        h = port[date]['opt.portfolio']['h.opt']
        tradelist = trades[date]
        long, short = np.sum(h[h > 0]), np.sum(h[h < 0])
        df.at[dt, 'long'], df.at[dt, 'short'] = long, short
        df.at[dt, 'net'], df.at[dt, 'gmv'] = long + short, np.abs(long) + np.abs(short)
        df.at[dt, 'traded'] = np.sum(np.abs(tradelist['h.opt'] - tradelist['h.opt.previous']))
    return df


attr, pchar = build_pnl_attribution(), build_portfolio_characteristics()
attr.cumsum().plot(ylabel='PnL Attribution'); pchar.plot(ylabel='Portfolio')
```

Реализованный PnL дня — $\sum_i h_i^* r_{i,t}$, где $r$ взято **на дату доходности** (ключ
`frames`), через два дня после экспозиций. Доходности винзоризируются на $\pm 50\%$ — шире,
чем при оценке факторов: не занижать реальные убытки, но защититься от ошибок в данных.
`partial_dot_product` нужен потому, что `facret[date]` содержит 81 фактор, а экспозиции —
77 или 4. Характеристики: `long`/`short` — суммы положительных и отрицательных позиций,
`net` — чистая экспозиция к рынку (у нейтрального портфеля около нуля), `gmv` — валовая
стоимость позиций $\sum|h_i|$, `traded` — дневной оборот; `traded/gmv` — оборачиваемость.

## 9.4. Дополнительные темы

### 9.4.1. Бустинг и AdaBoost

Из `Intro.md`: **бустинг** (boosting), в отличие от бэггинга, — последовательный ансамбль. В
AdaBoost первый слабый ученик минимизирует число ошибок; второй получает увеличенные веса на
объектах, которые первый классифицировал неверно; третий и далее поступают так же; итог —
взвешенное голосование. **Градиентный бустинг** обобщает идею: каждая следующая модель
обучается на градиенте функции потерь ансамбля (для MSE — на остатках). Бустинг силён, но
переобучается легче леса — что и демонстрирует следующее упражнение.

### 9.4.2. Упражнение на оверфиттинг: XGBoost на SPY

Квиз `overfitting_exercise` пытается предсказать дневную доходность ETF на S&P 500 (задачу,
которую большинство считают безнадёжной), чтобы показать, как выглядит переобучение. Признаки
для лагов $L = 1..25$: доходность $R_L$, квадрат $Rsq_L$, логарифм объёма
$V_L = \ln(1 + \mathrm{Volume}_{t-L})$ и $RV_L = R_L V_L$ — 100 признаков. В квизе они
строятся вложенными циклами по строкам; в pandas — сдвигами:

```python
import xgboost as xgb

model_df = pd.DataFrame({'y': df['Return']})
K = 25
for L in range(1, K + 1):
    model_df[f'R{L}'] = df['Return'].shift(L)
    model_df[f'Rsq{L}'] = df['Return'].shift(L) ** 2
    model_df[f'V{L}'] = np.log1p(df['Volume'].shift(L))
    model_df[f'RV{L}'] = model_df[f'R{L}'] * model_df[f'V{L}']

model = model_df.iloc[K:]
breakpoint_ = round(len(model) * 2 / 3)               # первые 2/3 — обучение, остаток — тест
X_train, Y_train = model.iloc[:breakpoint_, 1:], model.iloc[:breakpoint_, 0]
X_test, Y_test = model.iloc[breakpoint_:, 1:], model.iloc[breakpoint_:, 0]

param = {'max_depth': 20, 'verbosity': 0}            # verbosity заменил старый флаг тишины
xgModel = xgb.train(param, xgb.DMatrix(X_train, Y_train), num_boost_round=20)
preds_test = xgModel.predict(xgb.DMatrix(X_test))
```

Результат квиза: MSE на обучении $1{,}6 \cdot 10^{-6}$, на тесте $7{,}7 \cdot 10^{-5}$ — в
50 раз хуже. Затем строится «стратегия»: позиция = прогноз / дисперсия (оптимизация
среднее–дисперсия для одного актива), PnL = позиция × реализованная доходность. Кривая
накопленного PnL круто растёт на обучении и становится **абсолютно плоской** после границы:
деревья глубины 20 за 20 раундов запомнили обучающую выборку. Вопрос квиза «чем ещё
нереалистична симуляция?» остаётся читателю; наши соображения: нет издержек при ежедневной
смене позиции; позиция $\hat r/\sigma^2$ не ограничена и при малой волатильности даёт
огромное плечо; нет задержки из § 9.2.1; 100 признаков — сотни неявных испытаний (§ 9.2.2).

### 9.4.3. Своя риск-модель вместо Barra и AI-альфа вместо стилевых факторов

Данных Barra вне курса нет, но конструкция бэктеста от них не зависит — ей нужны лишь
$\mathbf{B}$, $\mathbf{F}$, $\mathbf{S}$, $\boldsymbol{\alpha}$ и ADV на каждый день.
**PCA-риск-модель главы 4** даёт всё это: $\mathbf{B}$ — нагрузки главных компонент на окне
(например, 252 дня), $\mathbf{F}$ — диагональная матрица дисперсий компонент (они
ортогональны по построению, так что диагональность здесь точна), $\mathbf{S}$ — дисперсии
остатков. Единицы — доли. Чтобы не заглядывать в будущее, PCA на день $t$ считается по
доходностям до $t$ включительно и применяется к доходности $t+2$. ADV — `AverageDollarVolume`
из Zipline (глава 5) или цена × объём.

**AI-альфа из главы 8** уже имеет нужную форму: один сигнал на акцию на день в $[-1, 1]$.
Остаётся стандартизовать его (центрировать, нормировать на $\sum|\alpha_i|$ — глава 4) и
отмасштабировать в доли дневной доходности, чтобы $\boldsymbol{\alpha}^T\mathbf{h}$ был в
долларах — например, «сигнал 1 ↔ несколько базисных пунктов», масштаб по реализованному IC на
обучении. Трудность из `project.md`: для Barra нужна таблица CUSIP ↔ `Barrid`, которой в курсе
не было; при собственной риск-модели альфа и риск считаются на одних тикерах. Другие идеи:
годы 2005–2008 (кризис — тест на структурные изменения), другие стилевые факторы в роли альф,
веса по скользящему Шарпу, издержки с корнем из объёма.

## 9.5. Выводы из результатов

Проект не печатает итоговых метрик; его результат — графики: кумулятивные доходности четырёх
альф, кумулятивная атрибуция (`daily.pnl`, `attribution.alpha.pnl`, `attribution.risk.pnl`,
`attribution.cost`) и характеристики портфеля (`long`, `short`, `net`, `gmv`, `traded`). Что в них читать:

* **Альфа-PnL против риск-PnL.** Если `attribution.alpha.pnl` идёт вверх, а
  `attribution.risk.pnl` колеблется около нуля, оптимизатор делает своё дело. Риск-PnL,
  сопоставимый с альфа-PnL, означает, что $\kappa$ мал или риск-модель неполна.
* **Издержки против альфы.** Кумулятивная `attribution.cost` должна быть заметно меньше
  альфа-PnL. Если нет — овертрейдинг из § 9.1; лечить его надо не подкруткой $\kappa$, а
  снижением оборота: более медленные альфы (однодневный разворот — самая быстрая и дорогая
  из четырёх), больший штраф $\lambda$.
* **`net` около нуля, `gmv` в десятках миллионов** при $\kappa = 10^{-6}$ — ожидаемо;
  `gmv` растёт от нуля (стартуем с пустой книги, издержки сдерживают набор позиций) и
  выходит на плато.
* **`daily.pnl` − (alpha + risk)** — специфический PnL; должен быть шумом без тренда.

Главный урок методологический: Шарп фактора в Alphalens (глава 5) и Шарп стратегии после
задержки, риск-модели и издержек — разные величины, и вторая почти всегда заметно меньше.

## 9.6. Типичные ошибки и ловушки

⚠️ **Присваивание в копию.** `df.loc[mask]['col'] = value` ничего не меняет (§ 9.3.3);
только `df.loc[mask, 'col'] = value`. Та же ловушка — `df[df.a > 0].b = 1`.

⚠️ **Доходность в универсуме.** Если не удалить `DlyReturn` из `universe`, ничего не
сломается — и это худший случай: опечатка (например, `DlyReturn` в списке факторов)
превратит бэктест в машину заглядывания в будущее с фантастическим Шарпом.

⚠️ **Единицы.** Забыть `0.01` у `SpecRisk` — завысить специфическую дисперсию в $10^4$ раз;
забыть `1e-4` у альфы — получить «ожидаемый доход» в тысячи процентов и бесконечное плечо.
Проверяйте порядки членов целевой функции на первом дне.

⚠️ **Не та дата для ковариации.** Ковариацию и экспозиции берём по `DataDate` ($t$),
доходность — по `DlyReturnDate` ($t+2$); перепутать — использовать будущую риск-модель.

⚠️ **Матрица $N \times N$.** `B @ F @ B.T`, `np.diag(specVar)`, `(QT @ Q) @ h` — каждое
выражение создаёт 2000 × 2000 чисел при каждом вызове целевой функции.

⚠️ **Нулевой ADV без замены.** $\lambda = 0{,}1 / 0$ даёт `inf`, целевая функция — `nan`,
L-BFGS молча возвращает мусор. Проверяйте `np.isfinite(Lambda).all()`.

⚠️ **Подстройка $\kappa$ перебором.** Каждая попытка — испытание в смысле § 9.2.2. Решите
заранее, какой GMV нужен, и зафиксируйте $\kappa$.

## 9.7. Вопросы для самопроверки

1. Объясните цепочку «данные $t$ → оптимизация → торговля $t+1$ → доходность $t+2$». Что
   изменится в бэктесте при задержке в один день? А в ноль?
2. Почему факторные доходности оцениваются регрессией без свободного члена на универсуме с
   капитализацией выше 1 млрд долларов и после винзоризации?
3. Выведите $\lambda_i = 0{,}1/\mathrm{ADV}_i$ из допущения «1 % ADV сдвигает цену на
   10 б. п.». Во сколько раз вырастут издержки при удвоении сделки?
4. Докажите, что $\mathbf{Q} = \mathbf{F}^{1/2}\mathbf{B}^T$ даёт
   $\mathbf{Q}^T\mathbf{Q} = \mathbf{B}\mathbf{F}\mathbf{B}^T$. Почему $\mathbf{Q}\mathbf{h}$
   считают внутри целевой функции, а $\mathbf{Q}$ — нет?
5. Продифференцируйте целевую функцию и получите градиент. Куда делись множители $1/2$?
6. Какие ограничения глав 3 и 5 какими членами целевой функции заменены? Что потеряно?
7. Почему `df.loc[df['SpecRisk'] == 0]['SpecRisk'] = ...` ничего не меняет и как
   проявляется эта ошибка в результатах бэктеста?
8. Атрибуция показала: альфа-PnL +3 млн, риск-PnL +4 млн, издержки −1 млн. Хороша ли
   стратегия? Что бы вы изменили?
9. Как заменить Barra риск-моделью из PCA (глава 4)? Почему для неё диагональная $\mathbf{F}$
   — не приближение? Как избежать заглядывания в будущее при оценке PCA?

## 9.8. Файлы репозитория и что читать дальше

* 📁 `Projects/8-Backtesting/project_8_starter.ipynb` — проект: Barra, сдвиг дат, факторные
  доходности, универсум, издержки, целевая функция, L-BFGS, цикл по дням, атрибуция PnL.
* 📁 `Quiz/m8/optimization_with_tcosts_solution.ipynb` — оптимизация одного дня по шагам:
  факторы Barra, единицы, сдвиг дат, винзоризация, OLS, $\mathbf{Q}$ вместо $N \times N$,
  $\kappa$ и GMV.
* 📁 `Quiz/m8/overfitting_exercise.ipynb` — XGBoost на SPY, демонстрация переобучения.
* 📁 `Notes/8-Backtesting/Intro.md`, `optimization.md`, `Attribution.md`, `project.md` —
  конспекты: валидность и оверфиттинг, овертрейдинг, издержки, атрибуция, кастомизация.

Что читать дальше: Р. Гринольд, Р. Кан, *Active Portfolio Management*; М. Лопес де Прадо,
*Advances in Financial Machine Learning* (оверфиттинг бэктестов, deflated Sharpe ratio);
*Crossover from Linear to Square-Root Market Impact*. Итоговая карта курса — в
[главе 10](10-zaklyuchenie.md).

---

Навигация: [← Глава 8. Машинное обучение для комбинирования альфа-факторов](08-ml-kombinirovanie-alf.md) | [Глава 10. Заключение →](10-zaklyuchenie.md)
