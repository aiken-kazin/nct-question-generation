# Criteria

Критерии генерации и валидации MCQ-задач для тестов ҰТО (teacher-certification).
Источник: `prompts/generator_*.md`, `prompts/critic_eval.md`, `src/agents.py`, `src/ensemble.py`, `src/symbolic.py`.

---

## 1. Как и по каким критериям генерируется задача

### Вход (что подаётся в `GeneratorAgent`)
Пользователь указывает 3 параметра + опционально 4-й:
- `subject` — `math` или `kazakh`
- `level` — `A` / `B` / `C`
- `format` — `text` или `image` (image только для math)
- `topic` — конкретный ID или random из YAML

### Сборка промпта (`src/config.py::render_generator`)
Из этих 4 параметров автоматически собирается промпт из **5 блоков**:

| Блок | Источник | Что даёт генератору |
|---|---|---|
| 1. **Topic info** | `prompts/topics_<subject>.yaml` | EN/KZ название топика + subtopics + keywords (= о чём писать) |
| 2. **Difficulty** | `prompts/difficulty.yaml` | Bloom-уровень + characteristics + distractor_guidance (= какой уровень сложности и каких ошибок ждать от дистракторов) |
| 3. **Format instructions** | inline | Для `text` — figure_spec=null. Для `image` — обязательно figure-dependent question («Суреттегі...»), figure_spec.parameters содержит все измерения, текст вопроса их **не** дублирует |
| 4. **Few-shot exemplars** | `files/mathematics_questions_kz.json` / `files/kazakh_questions_no_context.json` | 1–3 реальных НТЦ-задачи по тому же топику+уровню (`FEWSHOT_K=2` по умолчанию). Это стилевой и difficulty якорь. Только для text format |
| 5. **Feedback (retry)** | `CriticAgent` предыдущей попытки | Если предыдущий ран reject — конкретные замечания критика, которые надо исправить |

### Уровни (из `prompts/difficulty.yaml`)
| Level | Доля | Bloom | Когнитивная операция |
|---|---|---|---|
| `A` | 26% | Remember, Understand | 1 шаг, подстановка в формулу, узнавание определения |
| `B` | 60% | Apply, Analyze | 2–4 шага с понятной логической цепочкой, выбор формулы, составные фигуры |
| `C` | 14% | Evaluate, Synthesize | Нестандартная постановка, синтез ≥2 тем, оптимизация, обратная задача |

### Что генератор обязан вернуть (JSON-схема)

**Общие правила (math + kazakh):**
- Всё на казахском (literary norm, без русизмов/калек, official terminology).
- Ровно 4 опции `A/B/C/D`, один правильный, label рандомизирован.
- Опции примерно одной длины и формата.
- Запрещено: `«all of the above»`, `«none of the above»`, дубликаты.
- **Дистракторы по правилу:** каждый достижим **конкретной идентифицируемой ошибкой** (wrong формула, знак, off-by-one, типичное заблуждение). Не «случайные неправильные числа».
- Полное `explanation` со step-by-step решением.

**Math — дополнительно:**
- Вся математика — в LaTeX (`$...$`, `$$...$$`, `\frac`, `\sqrt`, `\vec`, `\lim`). Дроби в прозе через `/` запрещены.
- **Обязательный `verification` блок** (новелла #3): SymPy-сниппет, который независимо пересчитывает ответ:
  - `applicable: true` → код печатает ровно одно значение, `expected_output` совпадает byte-for-byte, `matches_option` указывает букву.
  - `applicable: false` — только если задача символьно не верифицируется (геометрия по чертежу, текстовая задача).
  - Разрешённые импорты: `sympy, math, cmath, fractions, decimal, numpy, itertools, functools, operator, statistics, os.path`. Network/process заблокированы.

**Image (math) — дополнительно:**
- Вопрос обязан быть **figure-dependent**: начинается с «Суреттегі...» / «Берілген суреттегі...».
- Все измерения (стороны, углы, координаты) лежат в `figure_spec.parameters`, **не** в тексте вопроса.
- Топик-зависимый figure_type (triangle / circle / function_graph / vector_diagram / ...).

---

## 2. Критерии валидации (Critic)

Критик работает **в 2 шага**:
1. **Solve** (`critic_solve.md`) — независимо решает задачу, не видя `correct_answer`.
2. **Eval** (`critic_eval.md`) — выставляет оценки 0–10 по 6 измерениям.

### Измерения и веса

| Измерение | Вес | Анкер 10 / 0 |
|---|---|---|
| **Correctness** | 3 | 10 = ответ критика == key + explanation корректен / 0 = key демонстрируемо неверен или нет ответа среди опций |
| **Distractor quality** | 2 | 10 = каждый дистрактор достижим конкретной ошибкой / 0 = правильный ответ виден сразу |
| **Difficulty alignment** | 2 | 10 = ровно матчит запрошенный A/B/C / 0 = ≥2 уровня в сторону |
| **Kazakh language quality** | 2 | 10 = literary norm, official terminology / 0 = machine-translated tone, ошибки терминологии |
| **LaTeX validity** | 1 | 10 = всё рендерится / 0 = сломанный LaTeX или математика без LaTeX (для text-only — авто 10) |
| **Figure relevance** | 1 | Только если есть фигура; иначе `null` |

### Hard rules (жёсткие фильтры)
- **`critic_answer != correct_answer` без ambiguity** → correctness ≤ 4 (обязательно).
- **`correctness < 7`** → reject, даже если overall ≥ 6.
- **`overall_score < 6.0`** → reject.
- **Symbolic verify contradiction** (для math): если sandbox выдал результат ≠ `expected_output` или `matches_option ≠ correct_answer` → correctness clamped to 0, hard reject.

### Vision routing
- Текстовая задача → text-критик (по умолчанию = генератор).
- Image-задача или real item с `metadata.context` → vision-критик (default `openai/gpt-4o-2024-11-20`), картинка передаётся base64 data URL.

### Retry-логика
- При reject — до **4 попыток** с фидбеком критика, передаваемым обратно в генератор.

---

## 3. Методология (приоритеты для публикации)

Три заявленные новеллы (см. `feature/symbolic-verification`):

### Novelty #1 — Critic Discrimination Index (CDI)
- **Что:** метрика валидации критика **без экспертной разметки**.
- **Как:** для каждой реальной НТЦ-задачи генерируем 2 degraded-варианта (wrong-key, weak-distractors) → критик оценивает обе версии → считаем gap.
- **Формулы:** `CDI = mean(real) − mean(degraded)` + Wilcoxon p-value + Cohen's d (paired).
- **Ожидание:** большой положительный CDI на dimensions, которые degradation атакует (wrong-key → correctness CDI ≫ 0); ~0 на нерелевантных.
- **Код:** `src/cdi.py`, `scripts/compute_cdi.py`.

### Novelty #2 — Multi-critic ensemble
- **Что:** N критиков из разных вендоров параллельно (`ThreadPoolExecutor`).
- **Дефолт:** GPT-5.5 + Claude Sonnet 4.6 + Gemini 3.1 Pro.
- **Агрегация:** majority vote по ответам, mean по dimensions, `pass_fail` = majority (или `--ensemble-strict` = unanimity).
- **Метрика для пейпера:** pairwise Cohen's κ между критиками = inter-rater reliability **без людей**.
- **Каждый critic preserved** в `ensemble.per_critic` JSON.
- **Код:** `src/ensemble.py`.

### Novelty #3 — Symbolic self-verification (math)
- **Что:** генератор сам пишет SymPy-сниппет, критик исполняет в hardened sandbox перед оценкой.
- **Расширяет:** Kadyrov et al. 2025 (+27% UNT math с SymPy-ассистом) — с **решения** на **генерацию**.
- **Sandbox:**
  - `python -I -c ...` (isolated, no user-site).
  - 5-second wall-clock timeout.
  - Import denylist: `subprocess, multiprocessing, socket, urllib, http, ssl, ctypes, ftplib, smtplib, …`.
  - Patched `importlib.import_module`, обнулённые `os.system / popen / exec* / remove / fork / kill`.
- **Эффект на оценку:** см. таблицу выше (clamp to 0 при contradiction).
- **Тесты:** 28 unit-tests против атак-векторов в `tests/test_symbolic.py`.
- **Код:** `src/symbolic.py`.

---

## 4. Что заявляем в пейпере (honest scope)

- **Заявляем:** CDI как метрика для valid-без-экспертов; ensemble κ как inter-rater reliability proxy; symbolic verify ловит generator-internal arithmetic inconsistencies.
- **НЕ заявляем:** что critic-ensemble == expert ground truth (3 confused critics могут единогласно ошибаться); что SymPy верифицирует все math items (только ~50–60%, у которых `applicable: true`); что sandbox airtight (research-grade, для production нужен Docker `--network=none`).
- **Минимум для публикации:** экспертная разметка на подвыборке для anchoring + CDI + κ + symbolic-verify hit-rate.
