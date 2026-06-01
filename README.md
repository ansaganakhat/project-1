# Midterm 2 Deep Learning Project

## Жоба тақырыбы

**Text Classification / Sentiment Analysis бағыты:** SMS хабарламаларын `ham` және `spam` класына жіктеу.

Бұл жоба built-in датасеттерді қолданбайды. Деректер сыртқы ашық дереккөзден қолмен жүктеледі.

## Датасет

- Датасет атауы: **SMS Spam Collection**
- Дереккөз: **UCI Machine Learning Repository**
- Dataset page: https://archive.ics.uci.edu/dataset/228/sms+spam+collection
- Direct download URL: https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip
- DOI: https://doi.org/10.24432/C5CC84
- Лицензия: Creative Commons Attribution 4.0 International (CC BY 4.0)

## Жоба құрылымы

```text
project-1/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   └── midterm2_sms_spam_deep_learning_kz.ipynb
├── src/
│   ├── config.py
│   ├── data.py
│   ├── eda.py
│   ├── neural_models.py
│   └── train.py
├── models/
│   ├── bilstm.keras
│   ├── mlp_tfidf.keras
│   ├── text_vectorizer_vocabulary.json
│   └── tfidf_vectorizer.joblib
├── results/
│   ├── figures/
│   ├── model_comparison.csv
│   └── metrics.json
├── README.md
└── requirements.txt
```

## Орнату және іске қосу

```bash
pip install -r requirements.txt
python src/train.py
```

`src/train.py` мына қадамдарды толық орындайды:

1. UCI датасетін қолмен жүктейді.
2. Raw файлды оқиды және тазалайды.
3. EDA summary және графиктерді жасайды.
4. Train/validation/test split орындайды.
5. Екі нейрондық желі моделін оқытады.
6. Accuracy, Precision, Recall, F1-score, loss curves және confusion matrix нәтижелерін сақтайды.

## EDA қысқаша нәтижелері

- Raw жазба саны: **5572**
- Тазаланғаннан кейінгі жазба саны: **5158**
- Өшірілген duplicate саны: **414**
- Missing values: **0**
- Class distribution:
  - `ham`: **4516** хабарлама, **87.55%**
  - `spam`: **642** хабарлама, **12.45%**
- SMS орташа ұзындығы: **79.20** таңба
- SMS орташа сөз саны: **15.42**

EDA графиктері `results/figures/` ішінде сақталған:

- `class_distribution.png`
- `missing_values.png`
- `message_length_distribution.png`
- `word_count_by_class.png`

## Оқытылған модельдер

### 1. Simple Neural Network: MLP + TF-IDF

Алдымен мәтін TF-IDF белгілеріне айналдырылады. Кейін Dense қабаттардан тұратын қарапайым нейрондық желі оқытылады.

### 2. Recurrent Neural Network: BiLSTM

Мәтін token ID тізбегіне айналдырылады. Кейін Embedding және Bidirectional LSTM қабаттары арқылы sequence-aware модель оқытылады.

## Model comparison

| Model | Accuracy | Precision | Recall | F1-score | Threshold |
|---|---:|---:|---:|---:|---:|
| MLP + TF-IDF | 0.9832 | 0.9368 | 0.9271 | 0.9319 | 0.37 |
| BiLSTM | 0.9767 | 0.9535 | 0.8542 | 0.9011 | 0.28 |

Test confusion matrices:

- MLP + TF-IDF: `[[672, 6], [7, 89]]`
- BiLSTM: `[[674, 4], [14, 82]]`

Бұл тәжірибеде **MLP + TF-IDF** F1-score бойынша жақсы нәтиже көрсетті. Себебі SMS spam мәтіндерінде "free", "win", "call", "prize" сияқты қысқа әрі айқын n-gram сигналдар көп. TF-IDF осы сөздік сигналдарды жақсы ұстайды. BiLSTM sequence құрылымын үйрене алады, бірақ шағын және теңгерімсіз датасетте ол көбірек overfitting қаупіне ұшырайды.

## Қорытынды талдау

MLP + TF-IDF:

- Күшті жағы: жылдам оқытылады, қысқа мәтіндегі маңызды сөздерді жақсы бөледі.
- Әлсіз жағы: сөз тәртібі мен ұзақ контексті толық түсінбейді.

BiLSTM:

- Күшті жағы: мәтіндегі сөз тәртібін және sequence pattern-дерді үйренеді.
- Әлсіз жағы: шағын датасетте overfitting болуы мүмкін және оқыту уақыты ұзағырақ.

## GitHub Repository

Repository link: https://github.com/ansaganakhat/project-1
