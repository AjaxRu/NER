# NER
Построение моделей для извлечения именованных сущностей (имена, места, организации и т.д.) из текстов.


# Разбор построения CRF 
!pip install datasets
from datasets import load_dataset
dataset = load_dataset("conll2003")

Изучение структуры датасета
Выводим доступные сплиты данных (train, validation, test)
print("Доступные сплиты:", dataset)

Извлечение тренировочного, валидационного и тестового наборов
train_data = dataset['train']
val_data = dataset['validation']
test_data = dataset['test']

Выводим пример данных из тренировочного набора
print("\nПример данных:")
print(train_data[0])

'tokens': Это список слов (или токенов) из одного предложения. В данном случае
'pos_tags': Это метки частей речи для каждого токена. Например, для слова 'EU' метка — 22, для слова 'rejects' — 42 и так далее. Эти числа представляют собой закодированные части речи.
'chunk_tags': Эти метки относятся к разбиению на "синтаксические чанки". Чанкинг — это процесс разбиения текста на небольшие, непрерывные группы слов, такие как "группы существительных" или "группы глаголов".
'ner_tags': Это основная разметка именованных сущностей для задачи NER. Здесь каждый токен имеет числовой код, который представляет определенную категорию сущностей.
0: 'O' — Это означает, что данный токен не является именованной сущностью (Outside).
3: 'B-ORG' — Токен 'EU' является началом именованной сущности типа Organization (организация).
7: 'B-MISC' — Токены 'German' и 'British' отмечены как начало сущности типа Miscellaneous (разные сущности, которые не попадают в другие категории, например, национальности или языки).

Категории сущностей (NER-теги):
O: Токен не является частью именованной сущности.
B-PER: Начало именованной сущности типа Person (человек).
I-PER: Внутри именованной сущности типа Person.
B-ORG: Начало именованной сущности типа Organization (организация).
I-ORG: Внутри именованной сущности типа Organization.
B-LOC: Начало именованной сущности типа Location (местоположение).
I-LOC: Внутри именованной сущности типа Location.
B-MISC: Начало именованной сущности типа Miscellaneous (разные категории, например, национальности, политические организации и т.д.).
I-MISC: Внутри именованной сущности типа Miscellaneous.

CRF требует особого формата, где каждая строка представляет один токен с его признаками (features) и целевым тегом (разметкой сущности).
Признаки для CRF могут включать POS-теги, информацию о токене (например, является ли слово заглавным, содержит ли цифры и т.д.), а также контекст (соседние слова).

!pip install sklearn-crfsuite
Функция для извлечения признаков из токена
def extract_features(tokens, pos_tags, chunk_tags, i):
    token = tokens[i]
    # Признаки самого токена
    features = {
        'bias': 1.0,  # bias feature (всегда равен 1 для всех токенов)
        'token.lower()': token.lower(),  # Токен в нижнем регистре
        'token.isupper()': token.isupper(),  # Весь токен в верхнем регистре?
        'token.istitle()': token.istitle(),  # Начинается ли токен с заглавной буквы?
        'token.isdigit()': token.isdigit(),  # Содержит ли токен цифры?
        'pos_tag': pos_tags[i],  # Часть речи (POS-тег)
        'chunk_tag': chunk_tags[i],  # Chunk-тег
    }

    # Признаки для соседних слов (контекст)
    if i > 0:
        prev_token = tokens[i-1]
        features.update({
            '-1:token.lower()': prev_token.lower(),
            '-1:token.isupper()': prev_token.isupper(),
            '-1:token.istitle()': prev_token.istitle(),
            'pos_tag-1': pos_tags[i-1],
            'chunk_tag-1': chunk_tags[i-1],
        })
    else:
        features['BOS'] = True  # Признак для начала предложения

    if i < len(tokens)-1:
        next_token = tokens[i+1]
        features.update({
            '+1:token.lower()': next_token.lower(),
            '+1:token.isupper()': next_token.isupper(),
            '+1:token.istitle()': next_token.istitle(),
            'pos_tag+1': pos_tags[i+1],
            'chunk_tag+1': chunk_tags[i+1],
        })
    else:
        features['EOS'] = True  # Признак для конца предложения

    return features

#Преобразуем данные в нужный формат для CRF
def prepare_data(data):
    sentences = []
    labels = []
    for entry in data:
        tokens = entry['tokens']
        pos_tags = entry['pos_tags']
        chunk_tags = entry['chunk_tags']
        ner_tags = entry['ner_tags']

        sentence_features = [extract_features(tokens, pos_tags, chunk_tags, i) for i in range(len(tokens))]
        sentences.append(sentence_features)
        labels.append([ner_tags[i] for i in range(len(tokens))])

    return sentences, labels

Преобразование всех сплитов в признаки и метки
X_train, y_train = prepare_data(train_data)
X_val, y_val = prepare_data(val_data)
X_test, y_test = prepare_data(test_data)

Функция для преобразования числовых меток в текстовые
def convert_labels(labels):
    return [[ner_tags[label] for label in sentence] for sentence in labels]

Преобразуем тренировочные, валидационные и тестовые метки из чисел в текстовые представления
y_train = convert_labels(y_train)
y_val = convert_labels(y_val)
y_test = convert_labels(y_test)

Пример того, как выглядят признаки для одного предложения
print("Пример признаков для одного токена в предложении:")
print(X_train[0][0])

Создание и настройка модели CRF
import sklearn_crfsuite
from sklearn_crfsuite import metrics
crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=100,
    all_possible_transitions=True
)

Обучаем модель на тренировочном наборе
crf.fit(X_train, y_train)

Предсказания на валидационном наборе
y_val_pred = crf.predict(X_val)

Оценка на валидационном наборе
labels = list(crf.classes_)
labels.remove('O')  # Убираем тег 'O', чтобы он не доминировал в метриках

Выводим метрики на валидационном наборе
print("Оценка на валидационном наборе:")
print(metrics.flat_classification_report(y_val, y_val_pred, labels=labels))

Предсказания на тестовом наборе
y_test_pred = crf.predict(X_test)

Оценка на тестовом наборе
print("Оценка на тестовом наборе:")
print(metrics.flat_classification_report(y_test, y_test_pred, labels=labels))
![image](https://github.com/user-attachments/assets/ebfa0701-f877-49ee-adf2-c98575d12b6d)

Основные метрики:
Precision (точность): Доля правильно предсказанных сущностей среди всех предсказанных как сущности. То есть, насколько "чистыми" являются предсказанные категории (меньше ложных срабатываний).
Recall (полнота): Доля правильно предсказанных сущностей среди всех фактических сущностей. Это показатель того, сколько реальных сущностей модель смогла "заметить".
F1-score: Среднее гармоническое между точностью и полнотой. Это общий показатель качества модели, который учитывает как точность, так и полноту.
Support: Количество истинных примеров каждой категории в наборе данных.

Общие выводы:
На валидационном наборе модель показывает высокие результаты: F1-score = 88%. Это говорит о том, что модель хорошо справляется с задачей на обучающей выборке.
На тестовом наборе модель немного теряет в качестве (F1 = 80%), что говорит о том, что тестовые данные содержат примеры, которые сложнее для модели.
Персоны (B-PER и I-PER) распознаются очень хорошо, что типично для таких датасетов, так как имена людей часто более однородны.
Организации (B-ORG, I-ORG) и локации (B-LOC, I-LOC) также распознаются довольно хорошо, но есть небольшие провалы в полноте на тестовом наборе (особенно для организаций).
Сущности типа Miscellaneous (B-MISC, I-MISC) сложнее для распознавания, что типично, так как эти сущности более разнообразны.

Проверим работу модели на примере:
Пример предложения для предсказания
example_sentence = ["John", "Smith", "is", "from", "New", "York", "and", "works", "at", "Google", "."]

POS-теги и chunk-теги для примера (можно взять случайные для демонстрации)
В реальных случаях это должны быть теги, предсказанные другой моделью или подготовленные вручную
example_pos_tags = [22, 22, 42, 35, 16, 16, 35, 42, 35, 16, 7]  # Это пример
example_chunk_tags = [11, 11, 21, 21, 11, 12, 21, 22, 21, 11, 0]  # Это пример

Преобразуем предложение в признаки для CRF
example_features = [extract_features(example_sentence, example_pos_tags, example_chunk_tags, i) for i in range(len(example_sentence))]

Делаем предсказание
predicted_labels = crf.predict([example_features])[0]  # Модель предсказывает текстовые метки (например, B-PER, O и т.д.)

Выводим результаты
for token, label in zip(example_sentence, predicted_labels):
    print(f"{token}: {label}")

![image](https://github.com/user-attachments/assets/1d357459-c6c2-4b80-8765-09b50a399615)

