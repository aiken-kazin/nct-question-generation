class Constants:
    
    ZERO_SHOT_QUESTION_GEN_PROMPT = """
    You are an expert educator and assessment designer for national tests aiming to assess school teachers' competency levels.
    Your task is to generate a single, high-quality multiple-choice question for a {subject} assessment.

    Subject: {subject}
    Topic: {topic}
    Subtopics: {subtopics}
    Learning outcomes: {learning_outcomes}
    Difficulty Level: {difficulty}
    Language: {language}

    Requirements:

    The question must be conceptually clear and perfectly aligned with the learning outcomes.

    Provide exactly {num_options} options. Only {num_correct_option} option(s) can be correct.

    The 3 incorrect distractors MUST be highly plausible, representing common student misconceptions or logical traps.

    All text must be naturally written in the {language} language.

    Output Format:
    You must respond ONLY with a valid JSON object using the exact schema below. Do not include markdown formatting like ```json or any conversational text before or after the JSON.

    {{
    "question": "Question text here",
    "difficulty_level": "Difficulty level here",
    "options": [
    {{"label": "A", "text": "Option text"}},
    {{"label": "B", "text": "Option text"}},
    {{"label": "C", "text": "Option text"}},
    {{"label": "D", "text": "Option text"}}
    ],
    "correct_answer": "Letter of the correct option",
    "explanation": "Brief explanation of why the answer is correct and what mistakes the distractors represent"
    }}
    """
    
    ONE_SHOT_QUESTION_GEN_PROMPT = r"""
    You are an expert educator and assessment designer for national tests aiming to assess school teachers' competency levels.
    Your task is to generate a single, high-quality multiple-choice question that assesses the same conceptual understanding as the provided Reference Example.

    # TARGET SPECIFICATIONS
    Subject: {subject}
    Topic: {topic}
    Subtopics: {subtopics}
    Difficulty Level: {difficulty}
    Language: {language}
    Number of Options: {num_options}
    Number of Correct Options: {num_correct_option}

    # REFERENCE EXAMPLE
    Analyze this example. Your generated question must match its style, complexity, and exact structural format, but test a DIFFERENT scenario or calculation within the same subtopic.
    Example Question: {example_question}
    Example Options: {example_options}
    Example Correct Answer: {example_correct_answer}

    # REQUIREMENTS
    1. The new question must be conceptually clear and perfectly aligned with the target specifications.
    2. Provide exactly {num_options} options. Only {num_correct_option} option(s) can be correct.
    3. The incorrect distractors MUST be highly plausible, representing common student misconceptions or logical traps.
    4. All text must be naturally written in the {language} language.
    5. FORMATTING CRITICAL: Any mathematical variables, formulas, or units must be written in standard LaTeX enclosed in dollar signs (e.g., $x^2$, $\sqrt{{3}}$, $F = ma$). Do not use plain text for math.

    # OUTPUT FORMAT
    You must respond ONLY with a valid JSON object using the exact schema below. Do not include markdown formatting like ```json or any conversational text before or after the JSON.

    {{
        "question": "Your newly generated question text here",
        "difficulty_level": "{difficulty}",
        "options": [
            {{"label": "A", "text": "Option text"}},
            {{"label": "B", "text": "Option text"}},
            {{"label": "C", "text": "Option text"}},
            {{"label": "D", "text": "Option text"}}
        ],
        "correct_answer": "Letter of the correct option",
        "explanation": "Brief explanation of why the answer is correct and what specific calculation mistakes or misconceptions the distractors represent."
    }}
    """
    
    DIFFICULTY_LEVELS ={
        "kk": {
            "А": "Базалық деңгейдегі тест тапсырмалары қарапайым білім мен дағдыларын пайдалануға, тестіленушінің ең төменгі дайындық деңгейіне баға беруге, белгілі бір нұсқаулардың көмегімен әрекеттерді орындауға, қарапайым дәлелдер мен ұғымдарды пайдалануға негізделген.",
            "B": "Орташа деңгейдегі тест тапсырмалары негізгі білім мен дағдыларын дұрыс пайдалануға, жаңа жағдайларда қарапайым модельдерді тануға, деректерді талдау мен салыстыруға, жүйелеуге, дәлелдерді қолданып, ақпаратты жалпылау мен қорытынды жасау қабілеттерін бағалауға негізделген.",
            "C": "Жоғары деңгейдегі тест тапсырмалары неғұрлым күрделі білім мен дағдыларын пайдалануды, тапсырмалардың күрделі модельдерін тануды, мәселелерді шешу үшін білім мен дағдыларын біріктіруді, күрделі ақпаратты немесе деректерді талдауды, пайымдауды, тұжырымдарды негіздеуге бағытталған."
        },
        "ru": {
            "А": "Базовый уровень трудности характеризует воспроизвдение простых знаний и навыков, позволяет провести оценки минимального уровня подготовленности тестируемого, выполнение простых действий с помощью определённых указаний, использование простых аргументов.",
            "B": "Средний уровень трудности характеризует правильное воспроизведение основных знаний и навыков, распознавание простых моделей в новых ситуациях, умение анализировать, сравнивать, обобщать и систематизировать данные, использовать аргументы, обобщать информацию и формулировать выводы.",
            "C": "Высокий уровень трудности характеризует воспроизведение более сложных знаний и навыков, распознавание более сложных моделей заданий, интегрирование знаний, умений и навыков, анализ сложной информации или данных, проводить рассуждение, обосновывать и формулировать выводы, направлено на разграничение фактов и их последствий, определение значимости представленных фактов."
        } 
    }