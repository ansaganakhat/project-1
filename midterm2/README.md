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
midterm2/
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
├── results/
├── README.md
└── requirements.txt
```

## Орнату және іске қосу

```bash
cd midterm2
pip install -r requirements.txt
python src/train.py
```

Jupyter notebook ашу:

```bash
cd midterm2
jupyter notebook notebooks/midterm2_sms_spam_deep_learning_kz.ipynb
```

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

## Оқытылған модельдер

1. **MLP + TF-IDF**: мәтінді TF-IDF белгілеріне айналдырып, Dense қабаттармен классификация жасайды.
2. **BiLSTM**: мәтінді token sequence ретінде алып, Embedding және Bidirectional LSTM қабаттарын қолданады.

## Model comparison

| Model | Accuracy | Precision | Recall | F1-score | Threshold |
|---|---:|---:|---:|---:|---:|
| MLP + TF-IDF | 0.9832 | 0.9368 | 0.9271 | 0.9319 | 0.37 |
| BiLSTM | 0.9767 | 0.9535 | 0.8542 | 0.9011 | 0.28 |

Test confusion matrices:

- MLP + TF-IDF: `[[672, 6], [7, 89]]`
- BiLSTM: `[[674, 4], [14, 82]]`

Бұл тәжірибеде **MLP + TF-IDF** F1-score бойынша жақсы нәтиже көрсетті. Себебі SMS spam мәтіндерінде қысқа әрі айқын n-gram сигналдар көп. TF-IDF осы сөздік сигналдарды жақсы ұстайды. BiLSTM sequence құрылымын үйрене алады, бірақ шағын және теңгерімсіз датасетте overfitting қаупі жоғарырақ.

## GitHub Repository

Repository link: https://github.com/ansaganakhat/project-1/tree/main/midterm2
