# МИДТЕРМ 2: Fashion-MNIST сурет классификациясы

Бұл жоба Fashion-MNIST dataset бойынша киім суреттерін 10 класқа классификациялайды. Жоба ішінде EDA, алдын ала өңдеу, екі нейрондық желіні оқыту, модельдерді салыстыру және бағалау бар.

## Dataset

- Dataset атауы: Fashion-MNIST
- Ашық дереккөз: Zalando Research / Kaggle
- Нақты сілтемелер:

  - Kaggle: https://www.kaggle.com/datasets/zalando-research/fashionmnist
- Көлемі: 60 000 training image және 10 000 test image
- Форматы: 28x28 grayscale суреттер
- Кластар саны: 10

Кластар:

| Label | Class |
|---:|---|
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

## Жоба талаптарының орындалуы

| Талап | Осы жобадағы орындалуы |
|---|---|
| Ашық dataset және сілтеме | Fashion-MNIST, GitHub және Kaggle links |
| EDA | Class distribution, sample images, pixel intensity analysis |
| Алдын ала өңдеу | Normalization, reshaping for MLP/CNN, train/validation split |
| Кемінде 2 neural network | MLP және CNN |
| Модельдерді салыстыру | Accuracy, loss, confusion matrix, classification report |
| Нәтижені рәсімдеу | Jupyter Notebook және VS Code script |
| GitHub repository | Төмендегі командалар арқылы push жасауға дайын |

## Project structure

```text
fashion_mnist_deep_learning/
├── notebooks/
│   └── fashion_mnist_midterm2.ipynb
├── src/
│   └── train.py
├── figures/
├── models/
├── results/
├── requirements.txt
├── .gitignore
└── README.md
```

## Орнату және іске қосу

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Jupyter Notebook арқылы:

```powershell
jupyter notebook notebooks/fashion_mnist_midterm2.ipynb
```

VS Code / terminal арқылы:

```powershell
python src/train.py --epochs 3 --batch-size 128
```

Скрипт аяқталған соң:

- `figures/` ішінде EDA және evaluation графиктері сақталады.
- `results/` ішінде metrics, classification report және model comparison сақталады.
- `models/` ішінде оқытылған модельдер сақталады.

## Қысқаша қорытынды

MLP моделі 28x28 суретті жай вектор ретінде қарайды, сондықтан пиксельдердің кеңістіктік байланысын толық пайдаланбайды. CNN моделі convolution қабаттары арқылы суреттегі локалды паттерндерді жақсы ұстайды. Сондықтан Fashion-MNIST сияқты image classification міндетінде CNN әдетте жоғары accuracy көрсетеді.

## Орындау нәтижесі

Notebook 3 epoch арқылы орындалды. Нәтиже random seed және ортаға байланысты аздап өзгеруі мүмкін.

| Model | Test accuracy | Test loss | Parameters |
|---|---:|---:|---:|
| CNN | 0.8970 | 0.2814 | 421 642 |
| MLP | 0.8643 | 0.3769 | 235 146 |

Осы нәтижеде CNN ең жақсы модель болды.

## GitHub-қа жүктеу

GitHub-та жаңа repository ашып, осы командаларды орында:

```powershell
git init
git add .
git commit -m "Add Fashion-MNIST deep learning midterm project"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/fashion-mnist-deep-learning.git
git push -u origin main
```

Содан кейін тапсыру форматына GitHub repository link және notebook/project файлын көрсетесің.
