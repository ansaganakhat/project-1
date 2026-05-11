# ============================================================
# Тәжірибелік жұмыс: Нейрондық желі — Iris жіктеу
# Студент аты-жөні: Ахат Аңсаған Қойшанұлы
# Топ: 
# ============================================================
#
# НҰСҚАУЛЫҚ:
#   Төменде # TODO деп белгіленген барлық жерлерді толтырыңыз.
#   Әр TODO-ның үстінде не жазу керек екені түсіндіріледі.
#   Кодты іске қосу үшін терминалда:  python iris_network.py
# ============================================================


# ============================================================
# 1-ҚАДАМ: Кітапханаларды жүктеу
# ============================================================
# Берілген кітапханаларды жүктеу жолдары толық жазылған.
# Оқып шығыңыз — бұл жолдарды өзгертпеңіз.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print("Кітапханалар жүктелді.")


# ============================================================
# 2-ҚАДАМ: Деректерді жүктеу
# ============================================================
# load_iris() функциясы Iris деректер жиынын қайтарады.
# iris.data    — белгілер матрицасы (150 x 4)
# iris.target  — кластар белгілері (0, 1, 2)

# TODO №1:
# load_iris() функциясын шақырып, нәтижесін iris деген айнымалыға сақтаңыз.
iris = load_iris()


# TODO №2:
# Белгілерді X айнымалысына, кластарды y айнымалысына сақтаңыз.
# Кеңес: iris.data және iris.target қасиеттерін қолданыңыз.
X = iris.data
y = iris.target


print("\n" + "=" * 50)
print("IRIS ДЕРЕКТЕР ЖИЫНЫ")
print("=" * 50)
print(f"Деректер өлшемі  : {X.shape}")
print(f"Белгілер атаулары: {iris.feature_names}")
print(f"Кластар атаулары : {list(iris.target_names)}")
print(f"Класс 0 - setosa    : {sum(y == 0)} мысал")
print(f"Класс 1 - versicolor: {sum(y == 1)} мысал")
print(f"Класс 2 - virginica : {sum(y == 2)} мысал")


# ============================================================
# 3-ҚАДАМ: Деректерді оқыту және тест жиындарына бөлу
# ============================================================
# train_test_split функциясы деректерді екіге бөледі:
#   X_train, y_train — оқыту үшін (80%)
#   X_test,  y_test  — тексеру үшін (20%)

# TODO №3:
# train_test_split функциясын толтырыңыз.
# Параметрлер:
#   - бірінші аргумент: X
#   - екінші аргумент: y
#   - test_size=0.2       (20% тест, 80% оқыту)
#   - random_state=42     (нәтиже қайталанатындай)
#   - stratify=y          (кластар тең бөлінеді)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


print("\n" + "=" * 50)
print("ДЕРЕКТЕРДІ БӨЛУ")
print("=" * 50)
print(f"Оқыту жиыны : {X_train.shape[0]} мысал")
print(f"Тест жиыны  : {X_test.shape[0]} мысал")


# ============================================================
# 4-ҚАДАМ: Деректерді стандартизациялау
# ============================================================
# StandardScaler деректерді нормализациялайды:
# әр белгінің орташа мәні 0-ге, дисперсиясы 1-ге тең болады.
# Бұл нейрондық желінің дұрыс оқытылуы үшін маңызды.

scaler = StandardScaler()

# TODO №4:
# X_train деректерін scaler арқылы өңдеңіз.
# fit_transform() әдісін қолданыңыз — ол алдымен fit жасап,
# содан кейін transform жасайды.
X_train = scaler.fit_transform(X_train)


# TODO №5:
# X_test деректерін scaler арқылы өңдеңіз.
# НАЗАР: тест деректері үшін тек transform() қолданыңыз,
# fit_transform() емес! Себебі scaler тест деректерін
# "көрмеуі" тиіс.
X_test = scaler.transform(X_test)


print("\n" + "=" * 50)
print("СТАНДАРТИЗАЦИЯ")
print("=" * 50)
print(f"X_train орташа мәні (бірінші белгі): {X_train[:, 0].mean():.4f}")
print(f"X_train дисперсия  (бірінші белгі): {X_train[:, 0].std():.4f}")


# ============================================================
# 5-ҚАДАМ: Нейрондық желіні жасау
# ============================================================
# MLPClassifier — көпқабатты нейрондық желі.
# Параметрлер:
#   hidden_layer_sizes — жасырын қабаттардағы нейрондар саны
#   activation         — активация функциясы
#   solver             — оңтайландырушы алгоритм
#   max_iter           — максималды итерациялар саны

# TODO №6:
# MLPClassifier моделін жасаңыз және model айнымалысына сақтаңыз.
# Мына параметрлерді қолданыңыз:
#   hidden_layer_sizes=(64, 32)
#   activation='relu'
#   solver='adam'
#   max_iter=500
#   random_state=42
model = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation='relu',
    solver='adam',
    max_iter=500,
    random_state=42
)


print("\n" + "=" * 50)
print("НЕЙРОНДЫҚ ЖЕЛІ АРХИТЕКТУРАСЫ")
print("=" * 50)
print("Кіріс қабаты    : 4 нейрон  (4 белгі)")
print("Жасырын қабат 1 : 64 нейрон (ReLU)")
print("Жасырын қабат 2 : 32 нейрон (ReLU)")
print("Шығыс қабаты    : 3 нейрон  (Softmax, 3 класс)")


# ============================================================
# 6-ҚАДАМ: Желіні оқыту
# ============================================================

print("\n" + "=" * 50)
print("ОҚЫТУ БАСТАЛДЫ...")
print("=" * 50)

# TODO №7:
# model.fit() әдісін шақырып, желіні оқытыңыз.
# fit() әдісіне X_train және y_train беріңіз.
# TODO
model.fit(X_train, y_train)

print(f"Оқыту аяқталды!")
print(f"Итерациялар саны  : {model.n_iter_}")
print(f"Соңғы жоғалту мәні: {model.loss_:.6f}")


# ============================================================
# 7-ҚАДАМ: Болжам жасау және бағалау
# ============================================================

# TODO №8:
# model.predict() әдісімен тест деректерінде болжам жасаңыз.
# Нәтижені y_pred айнымалысына сақтаңыз.
y_pred = model.predict(X_test)


# TODO №9:
# accuracy_score() функциясымен дәлдікті есептеңіз.
# Аргументтер: y_test (нақты), y_pred (болжам).
# Нәтижені accuracy айнымалысына сақтаңыз.
accuracy = accuracy_score(y_test, y_pred)


print("\n" + "=" * 50)
print("НӘТИЖЕЛЕР")
print("=" * 50)
print(f"Тест дәлдігі: {accuracy:.4f}  ({accuracy * 100:.2f}%)")

print("\nТолық есеп:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))


# TODO №10:
# confusion_matrix() функциясымен шатастыру матрицасын есептеңіз.
# Аргументтер: y_test, y_pred.
# Нәтижені cm айнымалысына сақтаңыз.
cm = confusion_matrix(y_test, y_pred)

print("Шатастыру матрицасы:")
print(cm)


# ============================================================
# 8-ҚАДАМ: Оқыту қисығын сызу
# ============================================================

print("\nГрафиктер салынуда...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# --- Сол жақ: оқыту жоғалту қисығы ---
# TODO №11:
# axes[0].plot() арқылы оқыту жоғалту қисығын сызыңыз.
# Деректер: model.loss_curve_
# Параметрлер: color='steelblue', linewidth=2
# TODO
axes[0].plot(model.loss_curve_, color='steelblue', linewidth=2)

axes[0].set_title('Оқыту барысындағы жоғалту', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Итерация')
axes[0].set_ylabel('Жоғалту (Loss)')
axes[0].grid(True, alpha=0.3)


# --- Оң жақ: шатастыру матрицасы ---
# TODO №12:
# axes[1].imshow() арқылы шатастыру матрицасын сызыңыз.
# Параметрлер: cm матрицасы, interpolation='nearest', cmap='Blues'
# Нәтижені im айнымалысына сақтаңыз.
im = axes[1].imshow(cm, interpolation='nearest', cmap='Blues')


axes[1].set_title('Шатастыру матрицасы', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Болжалды класс')
axes[1].set_ylabel('Нақты класс')
axes[1].set_xticks([0, 1, 2])
axes[1].set_yticks([0, 1, 2])
axes[1].set_xticklabels(list(iris.target_names), rotation=15)
axes[1].set_yticklabels(list(iris.target_names))

# Матрицаға сандарды жазамыз (өзгертпеңіз)
for i in range(3):
    for j in range(3):
        axes[1].text(j, i, str(cm[i, j]),
                     ha='center', va='center', fontsize=14, fontweight='bold',
                     color='white' if cm[i, j] > cm.max() / 2 else 'black')

plt.colorbar(im, ax=axes[1])
plt.suptitle(
    f'Нейрондық желі — Iris жіктеу  |  Дәлдік: {accuracy * 100:.1f}%',
    fontsize=14, fontweight='bold'
)
plt.tight_layout()
plt.savefig('result.png', dpi=120, bbox_inches='tight')
plt.show()
print("График 'result.png' файлына сақталды.")


# ============================================================
# 9-ҚАДАМ: Жаңа деректерде болжам жасау
# ============================================================
# Мұнда өзіміз жасаған жаңа гүлдің өлшемдерін береміз.
# Белгілер: [sepal length, sepal width, petal length, petal width]

new_flower = np.array([[5.1, 3.5, 1.4, 0.2]])

# TODO №13:
# new_flower деректерін scaler арқылы стандартизациялаңыз.
# transform() әдісін қолданыңыз (fit_transform емес!).
# Нәтижені new_flower_scaled айнымалысына сақтаңыз.
new_flower_scaled = scaler.transform(new_flower)


# TODO №14:
# model.predict() арқылы жаңа гүлдің класын болжаңыз.
# Нәтижені prediction айнымалысына сақтаңыз.
prediction = model.predict(new_flower_scaled)


# TODO №15:
# model.predict_proba() арқылы әр кластың ықтималдығын алыңыз.
# Нәтижені probabilities айнымалысына сақтаңыз.
probabilities = model.predict_proba(new_flower_scaled)


print("\n" + "=" * 50)
print("ЖАҢА ГҮЛДІ ЖІКТЕУ")
print("=" * 50)
print(f"Кіріс өлшемдер : {new_flower[0]}")
print(f"Болжалды класс : {iris.target_names[prediction[0]]}")
print(f"\nӘр кластың ықтималдығы:")
for name, prob in zip(iris.target_names, probabilities[0]):
    bar = '█' * int(prob * 30)
    print(f"  {name:12s}: {prob:.4f}  {bar}")


# ============================================================
# 10-ҚАДАМ: Қорытынды сұрақтар
# ============================================================
# Мына сұрақтарға жауапты осы файлдың соңына
# # ЖАУАП: деп бастап жазыңыз.
#
# Сұрақ 1: hidden_layer_sizes=(64, 32) дегені не білдіреді?
# ЖАУАП: Бұл нейрондық желіде 2 жасырын қабат бар екенін білдіреді.
# Бірінші қабатта 64 нейрон, екінші қабатта 32 нейрон орналасқан.

# Сұрақ 2: Неліктен X_test үшін fit_transform емес, transform қолдандық?
# ЖАУАП: Себебі тест деректері модельге алдын ала белгілі болмауы тиіс.
# fit_transform қолдансақ scaler тест деректерінің статистикасын үйреніп алады,
# бұл деректердің ағып кетуіне (data leakage) әкеледі.

# Сұрақ 3: Шатастыру матрицасының диагоналы нені білдіреді?
# ЖАУАП: Диагональдағы мәндер дұрыс жіктелген объектілер санын көрсетеді.

# Сұрақ 4: activation='relu' орнына 'tanh' жазсақ не өзгереді?
# ЖАУАП: Активация функциясы өзгереді.
# tanh функциясы шығысты -1 мен 1 аралығында береді,
# ал relu тек оң мәндерді өткізеді. Бұл оқыту жылдамдығы мен
# модель нәтижесіне әсер етуі мүмкін.

# Сұрақ 5: Дәлдік 100% болса, бұл әрқашан жақсы ма? Неге?
# ЖАУАП: Жоқ, әрқашан жақсы емес.
# Кейде модель деректерді жаттап алып (overfitting),
# жаңа деректерде нашар жұмыс істеуі мүмкін.


print("\n" + "=" * 50)
print("ЖҰМЫС АЯҚТАЛДЫ")
print("=" * 50)
