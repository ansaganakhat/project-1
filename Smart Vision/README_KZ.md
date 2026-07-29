# Smart Vision — финалдық машиналық оқыту жобасы

Бұл нұсқа екі түрлі модель мен екі түрлі dataset-ке арнайы бейімделген.

## Қолданылатын нақты жолдар

### YOLOv10

- Weights: `C:\Users\Ansagan\Documents\Ansagan\Ansagan\yolov10\runs\detect\runs\runs\detect\yolov10x_custom\weights`
- Train images: `C:\Users\Ansagan\Documents\Ansagan\Ansagan\dataset\images\train`
- Val images: `C:\Users\Ansagan\Documents\Ansagan\Ansagan\dataset\images\val`
- Labels автоматты түрде `dataset\labels\train` және `dataset\labels\val` жолдарынан табылады.

### YOLOv5

- Weights: `C:\Users\Ansagan\Documents\Project\yolov5\runs\train\exp\weights`
- Images root: `C:\Users\Ansagan\Documents\Project\dataset\images`
- Labels автоматты түрде `C:\Users\Ansagan\Documents\Project\dataset\labels` ішінен табылады.
- Егер `images\train` және `images\val` бар болса, солар қолданылады.
- Егер суреттер бір папкада тұрса, runtime код оларды тұрақты 80/20 train/val тізіміне бөледі. Бастапқы файлдар көшірілмейді және өзгермейді.

## Неге runtime YAML жасалады?

Модельдердің сыртқы `data.yaml` файлдарында ескі немесе басқа компьютерге арналған жолдар болуы мүмкін. Код YAML ішінен тек `names` класс атауларын алып, `outputs/runtime/` ішінде нақты жолдарға негізделген жаңа YAML жасайды.

## Орнату

Project-ті қысқа жолға шығару ұсынылады:

```text
C:\MLProject\smart_vision_final_project_ready
```

Ұзын Windows жолы PyTorch орнату кезінде `WinError 206` қатесін тудыруы мүмкін.

1. `install_windows.bat` іске қосыңыз.
2. `check_project.bat` іске қосыңыз.
3. Барлығы дұрыс болса, `start_notebook.bat` іске қосыңыз.
4. Kernel: **Python (ML YOLO Final)**.

Егер `C:\venvs\ml-yolo` ортасы бұрыннан бар және пакеттер орнатылған болса, бірден `check_project.bat` бастауға болады.

## Notebook реті

1. `00_config_and_paths.ipynb` — жолдар, runtime YAML және label диагностикасы.
2. `01_EDA_Feature_Engineering.ipynb` — толық EDA, brightness/contrast, bbox area/aspect ratio, class imbalance.
3. `02_Classical_ML_HOG_SVM_SHAP.ipynb` — HOG + SVM, GridSearchCV, confusion matrix, coefficient importance, permutation importance, SHAP.
4. `03_YOLOv5_Evaluation.ipynb` — YOLOv5 өз dataset-імен бағаланады.
5. `04_YOLOv10_Evaluation.ipynb` — YOLOv10 кеңейтілген dataset-пен бағаланады.
6. `05_Model_Comparison_Business_Deployment.ipynb` — модельдерді ғылыми дұрыс түсіндіру, бизнес KPI, шектеулер, deployment.
7. `06_Run_Project_Test_Demo.ipynb` — сурет, папка, видео немесе webcam inference.

## Маңызды ғылыми ескерту

YOLOv5 және YOLOv10 әртүрлі dataset-терде оқытылып, өз validation жиынтықтарында бағаланады. Сондықтан олардың mAP мәндерін таза архитектуралық benchmark ретінде тікелей түсіндіруге болмайды. YOLOv10 dataset-і үлкенірек әрі класс саны көбірек. HOG + SVM object-crop classification орындайды, ал YOLO — object detection.

## Тест суреттері

Суреттерді `test_images/` папкасына салыңыз. Содан кейін:

```bat
run_yolov10_demo.bat
```

немесе `06_Run_Project_Test_Demo.ipynb` ашыңыз.

## Нәтижелер

- `outputs/eda/` — EDA CSV файлдары.
- `outputs/models/` — HOG+SVM моделі.
- `outputs/metrics/` — барлық JSON метрикалары және comparison CSV.
- `outputs/predictions/` — тест детекциялары.
- `outputs/runtime/` — автоматты жасалған runtime YAML және train/val тізімдері.

## Егер check_project қатесі шықса

`missing_label_examples` бағаны бағдарлама қандай label жолын күткенін көрсетеді. Қалыпты құрылым:

```text
dataset/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

немесе flat YOLOv5 dataset:

```text
dataset/
├── images/
└── labels/
```

---

## Әлсіз ноутбуктағы CPU режимі

Бұл нұсқа Intel Iris графикасы бар, NVIDIA CUDA жоқ ноутбукке бейімделген.

- Virtual environment: `C:\venvs\ml-yolo-cpu`
- PyTorch: CPU-only
- Jupyter kernel: **Python (ML YOLO CPU)**
- EDA әдепкіде әр split-тен 300 сурет өңдейді.
- HOG+SVM әдепкіде әр кластан 50 объект алады.
- GridSearchCV `n_jobs=1` режимінде жұмыс істейді, сондықтан 8 GB RAM-та қауіпсіздеу.
- YOLO validation `batch=1`, `device="cpu"` режимінде жүреді.

YOLOv10x үлкен модель болғандықтан CPU-да validation және inference баяу болады. Алдымен `00`, `01`, `02` notebook-тарын тексеріп, demo үшін 1–2 тест суретін ғана қолданыңыз.

Бұл компьютерде сыртқы модель және dataset папкалары міндетті түрде болуы керек. Virtual environment модель файлдарын немесе суреттерді өзімен бірге көшірмейді.
