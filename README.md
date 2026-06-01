# nct-question-generation

Генерация MCQ-задач на казахском языке для тестов ҰТО (математика + қазақ тілі/әдебиеті). Pipeline: Generator → Critic (с опцией ensemble и symbolic-verify для математики).

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # вписать OPENROUTER_API_KEY
```

## Как сгенерить

### 1. Посмотреть топики

```bash
grep "^\s*- id:" prompts/topics_math.yaml
grep "^\s*- id:" prompts/topics_kazakh.yaml
```

Полный список — внизу README.

### 2. Уровни

| Level | Описание | Доля |
|---|---|---|
| `A` | Basic — определения, 1 шаг | 26% |
| `B` | Medium — 2–4 шага | 60% |
| `C` | Hard — нестандартное, несколько тем | 14% |

### 3. Сгенерить одну задачу — `generate.py`

```bash
# Математика
python generate.py --subject math --level A --format text
python generate.py --subject math --level B --format text --topic triangles
python generate.py --subject math --level C --format image --topic polygons_circles

# Қазақ тілі / әдебиет
python generate.py --subject kazakh --level A --format text
python generate.py --subject kazakh --level B --format text --topic syntax
python generate.py --subject kazakh --level C --format text --topic literature
```

`--format image` — только для math.

### 4. Сгенерить батч — `scripts/run_batch.py`

```bash
# 50 задач по математике
python scripts/run_batch.py --subject math --count 50

# 50 задач по казахскому
python scripts/run_batch.py --subject kazakh --count 50

# Оба предмета сразу
python scripts/run_batch.py --subject both --count 50

# С ensemble (3 критика, ~3× стоимость критика)
python scripts/run_batch.py --subject math --count 50 --ensemble

# Возобновить прерванный ран
python scripts/run_batch.py --subject math --count 50 --resume
```

Распределение по уровням делается автоматически (26 / 60 / 14).

### 5. Сменить модель

```bash
# Шорткаты
--api gpt-5.5
--api claude-sonnet-4.6
--api gemini-3.1-pro          # дефолт

# Любая модель OpenRouter
--model anthropic/claude-sonnet-4.6
```

## Результаты

```
output/<subject>/<model_slug>/level_<A|B|C>/<timestamp>_<id>.json   # данные
output/<subject>/<model_slug>/level_<A|B|C>/<timestamp>_<id>.md     # превью
output/<subject>/<model_slug>/_manifest_<timestamp>.json            # сводка батча
```

## Топики

### Math (20)

| ID | KZ |
|---|---|
| `triangles` | Үшбұрыштар |
| `polygons_circles` | Көпбұрыштар және шеңберлер |
| `vectors` | Векторлар |
| `quadratic_irrational` | Квадраттық және иррационал теңдеулер |
| `inequalities` | Теңсіздіктер |
| `coordinate_3d` | Кеңістіктік координаталар геометриясы |
| `sequences` | Прогрессиялар |
| `trigonometry` | Тригонометрия |
| `combinatorics_probability` | Комбинаторика және ықтималдық теориясы |
| `perpendicularity_3d` | Кеңістіктегі перпендикулярлық және параллельдік |
| `polyhedra` | Көпжақтар |
| `solids_of_revolution` | Айналу денелері |
| `exponential_logarithmic_functions` | Көрсеткіштік және логарифмдік функциялар |
| `exponential_log_equations` | Көрсеткіштік және логарифмдік теңдеулер |
| `derivatives` | Туындылар |
| `integrals` | Интегралдар |
| `powers_roots` | Дәреже және түбірлер |
| `polynomials` | Көпмүшелер |
| `functions_limits` | Функциялар және шектер |
| `complex_numbers` | Кешен сандар |

### Kazakh (11)

| ID | KZ |
|---|---|
| `phonetics` | Фонетика |
| `morphology` | Морфология |
| `word_formation` | Сөзжасам |
| `syntax` | Синтаксис |
| `lexicology` | Лексикология |
| `orthography` | Орфография |
| `punctuation` | Пунктуация |
| `text_analysis` | Мәтін талдау |
| `literature` | Қазақ әдебиеті |
| `teaching_methodology` | Қазақ тілін оқыту әдістемесі |
| `language_history` | Қазақ тілінің тарихы |

## Калибровка критика (по желанию)

```bash
# CDI на реальных НТЦ-задачах
python scripts/calibrate_critic.py --subject math --ensemble

# Пересчитать CDI из CSV без новых API-вызовов
python scripts/compute_cdi.py output/_critic_validation/math_ensemble_*.csv
```
