# NER
Построение моделей для извлечения именованных сущностей (имена, места, организации и т.д.) из текстов.  

В качестве baseline модели используется Conditional Random Fields (CRF). Эта модель основана на последовательной классификации токенов, учитывающей их контекст и заранее определенные признаки, такие как POS-теги и синтаксические структуры.  
Также рассмотрен подход с использованием предобученной трансформерной модели BERT (Bidirectional Encoder Representations from Transformers). BERT захватывает более глубокие связи между токенами и их контекст, что делает его более точным в распознавании сложных и многословных сущностей.


# Разбор построения CRF 
```
!pip install datasets  
from datasets import load_dataset  
dataset = load_dataset("conll2003")  
```

Изучение структуры датасета  
Выводим доступные сплиты данных (train, validation, test)
```
print("Доступные сплиты:", dataset)
```  

Извлечение тренировочного, валидационного и тестового наборов  
```
train_data = dataset['train']  
val_data = dataset['validation']  
test_data = dataset['test']
```

Выводим пример данных из тренировочного набора  
```
print("\nПример данных:")  
print(train_data[0])
```

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

```
!pip install sklearn-crfsuite
```
Функция для извлечения признаков из токена  
```
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
```

Преобразуем данные в нужный формат для CRF  
```
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
```

Преобразование всех сплитов в признаки и метки  
```
X_train, y_train = prepare_data(train_data)  
X_val, y_val = prepare_data(val_data)  
X_test, y_test = prepare_data(test_data)
```

Функция для преобразования числовых меток в текстовые  
```
def convert_labels(labels):
    return [[ner_tags[label] for label in sentence] for sentence in labels]
```

Преобразуем тренировочные, валидационные и тестовые метки из чисел в текстовые представления  
```
y_train = convert_labels(y_train)  
y_val = convert_labels(y_val)  
y_test = convert_labels(y_test)
```

Пример того, как выглядят признаки для одного предложения  
```
print("Пример признаков для одного токена в предложении:")  
print(X_train[0][0])
```

Создание и настройка модели CRF  
```
import sklearn_crfsuite
from sklearn_crfsuite import metrics
crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=100,
    all_possible_transitions=True
)
```

Обучаем модель на тренировочном наборе  
```
crf.fit(X_train, y_train)
```

Предсказания на валидационном наборе  
```
y_val_pred = crf.predict(X_val)
```

Оценка на валидационном наборе  
```
labels = list(crf.classes_)  
labels.remove('O')  # Убираем тег 'O', чтобы он не доминировал в метриках
```

Выводим метрики на валидационном наборе  
```
print("Оценка на валидационном наборе:")  
print(metrics.flat_classification_report(y_val, y_val_pred, labels=labels))
```

Предсказания на тестовом наборе  
```
y_test_pred = crf.predict(X_test)
```

Оценка на тестовом наборе  
```
print("Оценка на тестовом наборе:")  
print(metrics.flat_classification_report(y_test, y_test_pred, labels=labels))  
![image](https://github.com/user-attachments/assets/ebfa0701-f877-49ee-adf2-c98575d12b6d)
```

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

# Пример использования:
Пример предложения для предсказания  
```
example_sentence = ["John", "Smith", "is", "from", "New", "York", "and", "works", "at", "Google", "."]
```

POS-теги и chunk-теги для примера (можно взять случайные для демонстрации)  
В реальных случаях это должны быть теги, предсказанные другой моделью или подготовленные вручную  
```
example_pos_tags = [22, 22, 42, 35, 16, 16, 35, 42, 35, 16, 7]  # Это пример  
example_chunk_tags = [11, 11, 21, 21, 11, 12, 21, 22, 21, 11, 0]  # Это пример
```

Преобразуем предложение в признаки для CRF  
```
example_features = [extract_features(example_sentence, example_pos_tags, example_chunk_tags, i) for i in range(len(example_sentence))]
```

Делаем предсказание  
```
predicted_labels = crf.predict([example_features])[0]  # Модель предсказывает текстовые метки (например, B-PER, O и т.д.)
```

Выводим результаты 
```
for token, label in zip(example_sentence, predicted_labels):
    print(f"{token}: {label}")
```

![image](https://github.com/user-attachments/assets/1d357459-c6c2-4b80-8765-09b50a399615)  


# Разбор построения BERT
Подготавливаем данные  
```
!pip install transformers datasets  
rom datasets import load_dataset  
dataset = load_dataset("conll2003")  
train_data = dataset['train']  
val_data = dataset['validation']  
test_data = dataset['test']  
print(train_data[0])
```

BERT требует специальной токенизации, которая учитывает подслова (subword tokenization).  
Нам нужно будет токенизировать предложения с помощью токенизатора BERT, а затем адаптировать разметку сущностей к новой токенизации.  
```
from transformers import BertTokenizerFast
```

Загрузка предобученного токенизатора BERT  
```
tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
```

Функция для токенизации текста и адаптации меток NER  
```
def tokenize_and_align_labels(examples):
    # Добавляем padding и truncation
    tokenized_inputs = tokenizer(
        examples['tokens'],
        truncation=True,
        is_split_into_words=True,
        padding='max_length',  # Добавляем padding до максимальной длины
        max_length=128  # Максимальная длина последовательности (можно варьировать)
    )

    labels = []
    for i, label in enumerate(examples['ner_tags']):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Получаем индексы слов после токенизации
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:  # Пропускаем спецсимволы
                label_ids.append(-100)
            elif word_idx != previous_word_idx:  # Новое слово
                label_ids.append(label[word_idx])
            else:  # Подслова
                label_ids.append(-100)  # Для подслов добавляем -100, чтобы их игнорировать при обучении
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs
```

Токенизация и выравнивание меток для всех сплитов  
```
tokenized_train = train_data.map(tokenize_and_align_labels, batched=True)  
tokenized_val = val_data.map(tokenize_and_align_labels, batched=True)  
tokenized_test = test_data.map(tokenize_and_align_labels, batched=True)
```

Теперь построим модель  
```
from transformers import BertForTokenClassification, TrainingArguments, Trainer
```

Загрузка модели BERT 
```
model = BertForTokenClassification.from_pretrained('bert-base-cased', num_labels=len(dataset['train'].features['ner_tags'].feature.names))
``` 

Установка параметров обучения  
```
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    # Добавляем padding и truncation
    gradient_accumulation_steps=2,  # Для более стабильного обучения
    fp16=True  # Использование 16-битных вычислений для ускорения
)
```
Создание Trainer  
```
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,  # Указываем токенизатор
)
```
Обучение модели  
```
trainer.train()
``` 
![image](https://github.com/user-attachments/assets/887508b4-6374-4553-8f72-b67143520740)  
Снижение Loss на валидации говорит о том, что модель не переобучается и продолжает хорошо обобщать на новых данных. В модели BERT для NER используется функция потерь, известная как кросс-энтропия с маскированием (CrossEntropyLoss), специально предназначенная для задач классификации токенов (Token Classification), таких как NER.  

Кросс-энтропия: Эта функция измеряет разницу между предсказанным распределением классов (в данном случае метки именованных сущностей) и истинным распределением.
  Для каждой позиции (токена) она вычисляет вероятность принадлежности токена к каждому классу (например, B-PER, I-ORG, O и т.д.)
  и штрафует модель за неправильные предсказания.  
Маскирование: Когда мы используем BERT, некоторые токены (подслова или специальные токены, такие как [PAD] и [CLS])
  не должны влиять на вычисление потерь. Поэтому для таких токенов используется специальная маска (значение -100), чтобы
  они не участвовали в подсчете потерь.  

Теперь перейдем к использованию обученной модели для предсказания именованных сущностей в произвольном предложении.  
```
import torch
```

Проверяем, доступен ли GPU  
```
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

Перемещаем модель на устройство  
```
model.to(device)
```
```
def predict_ner_for_sentence(sentence):
    # Токенизация предложения
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True, is_split_into_words=False)

    # Перемещаем входные данные на устройство (то же, что и модель)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Модель BERT делает предсказание для токенов
    outputs = model(**inputs)

    # Получаем предсказания (логиты) и преобразуем их в метки
    predictions = outputs.logits.argmax(dim=-1).squeeze().tolist()

    # Преобразуем токены и их предсказанные метки обратно в человекочитаемый формат
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'].squeeze().tolist())
    predicted_labels = [dataset['train'].features['ner_tags'].feature.names[pred] for pred in predictions]

    # Выводим токены вместе с предсказанными метками
    for token, label in zip(tokens, predicted_labels):
        print(f"{token}: {label}")
```
# Пример использования
```
sentence = "John Smith works at Google in New York"  
predict_ner_for_sentence(sentence)
```
![image](https://github.com/user-attachments/assets/602cb211-8b00-4724-b4ba-8c9f252bef31)  

# Выводы
Проведем сравнение построенных нами моделей:
Производительность: BERT превосходит CRF, особенно на сложных именованных сущностях и многословных фразах. Он лучше захватывает контекст и работает с неявными связями в предложениях. CRF хорошо справляется с простыми сущностями, но его производительность страдает на сложных категориях и при отсутствии богатых признаков.  
Гибкость: BERT не требует ручной настройки признаков, что делает его более универсальным. CRF зависит от тщательно подобранных признаков и может не показывать хорошие результаты на новых данных.  
Ресурсы: BERT требует больше вычислительных ресурсов (GPU и оперативной памяти), в то время как CRF — более легкая модель и может быть запущена даже на CPU.  
Таким образом, если задача требует высокой точности и работа с большими объемами данных оправдана, BERT является очевидным выбором. Однако, если ресурсы ограничены или нужна быстро обучаемая модель с приемлемой точностью, CRF — хороший выбор.

