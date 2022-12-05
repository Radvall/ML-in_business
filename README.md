python-flask-docker

Итоговый проект курса "Машинное обучение в бизнесе"

Стек:

    ML: sklearn, pandas, numpy 
    API: flask 
    Данные: с kaggle - https://www.kaggle.com/datasets/pritsheta/heart-attack

Задача: предсказать сердечный приступ

Используемые признаки:

    1. Age: Age (in years) .
    2. Sex: gender (1 = male; 0 = female).
    3.ChestPain: Chest Pain type -- 1: typical angina (all criteria present) -- 2: atypical angina (two of three criteria satisfied) -- 3: non-anginal pain (less than one criteria satisfied) -- 4: asymptomatic (none of the criteria are satisfied).
    4. Restbps: Resting Blood pressure (in mmHg, upon admission to the hospital). 
    5. Chol: serum cholesterol in mg/dL. 
    6. Fbs: fasting blood sugar > 120 mg/dL (likely to be diabetic) 1 = true; 0 = false .
    7. RestECG: Resting electrocardiogram results -- Value 0: normal -- Value 1: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV) -- Value 2: showing probable or definite left ventricular hypertrophy by Estes' criteria .
    8. MaxHR: Greatest number of beats per minute your heart can possibly reach during all-out strenuous exercise.
    9.Exang: exercise induced angina (1 = yes; 0 = no).
    10. Oldpeak: ST depression induced by exercise relative to rest (in mm, achieved by subtracting the lowest ST segment points during exercise and rest).
    11. Slope: the slope of the peak exercise ST segment, ST-T abnormalities are considered to be a crucial indicator for identifying presence of ischaemia -- Value 1: upsloping -- Value 2: flat -- Value 3: downsloping .
    12.Ca: number of major vessels (0-3) colored by fluoroscopy. Major cardial vessels are as goes: aorta, superior vena cava, inferior vena cava, pulmonary artery (oxygen-poor blood --> lungs), pulmonary veins (oxygen-rich blood --> heart), and coronary arteries (supplies blood to heart tissue).
    13. AHD: 0 = normal; 1 = fixed defect (heart tissue can't absorb thallium both under stress and in rest); 2 = reversible defect (heart tissue is unable to absorb thallium only under the exercise portion of the test)
    14.Target: 0 = no disease, 1 = disease.


Преобразование признаков: OneHotEncoder

Модель: RandomForestClassifier


Клонируем репозиторий и создаем образ
```
$ git clone https://github.com/Radvall/ML_in_business.git
$ cd ML_in_busines
$ docker build -t ml/python_flask_docker .
```

Запускаем контейнер

Здесь Вам нужно создать каталог локально и сохранить туда предобученную модель (<your_local_path_to_pretrained_models> нужно заменить на полный путь к этому каталогу)
```
$ docker run -d -p 8180:8180 -v <your_local_path>/ML_in_busines/app/models:/app/app/models ml/python_flask_docker
```

POST запрос в формате JSON (в соответствии с названием признаков) принимает http://<server IP>:8180/predict
    
result['predictions'] -> вероятность наличия заболевания
