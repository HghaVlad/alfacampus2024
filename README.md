# alfacampus2024

## Решение ML Хакатона по предсказанию оттока партнеров Банка от команды *whisker max*

### Задача
Все материалы хакатона находятся здесь -> https://disk.yandex.ru/d/hmymgchrbREnnA

Вам предстоит выполнить задачу бинарной классификации партнеров банка: построить модель, прогнозирующую отток партнера из банка на основе исторических данных, и предсказать категорию партнёра на тестовой выборке.

**Формат ввода**
Для анализа и построения модели вам доступен датасет dataset.csv с записями о клиентах банка и значениями целевых переменных для каждого из них. Описание факторов из датасета доступно в файле description.xlsx.

С помощью построенной модели нужно сделать предсказание на тестовой выборке. Подробная информация доступна в презентации кейса.

**Формат вывода**
В качестве решения вам нужно загрузить файл с предсказаниями вашей модели на тестовой выборке.

Прикрепите в поле ответа .csv файл, в котором будет столбец-ПИН клиента и предсказание для него. Файл должен содержать N=4509 строк (включая заголовок) с предсказаниями модели на каждом из тестовых примеров. Файл должен содержать поля clientbankpartner_pin,score в заголовочной строке файла. Обратите внимание, что clientbankpartner_pin строк должны соответствовать требуемым, как в примере ниже.

Пример файла для отправки - submission_example.csv из файлов с диска.

Для оценки ваших решений будет использоваться метрика ROC-AUC.

**Примечания**
Во время соревнования в мониторе будет отображаться ваш лучший результат на тестовой выборке. Внимание – в тестовую выборку попали не все партнёры, а лишь те, кто был активен в последние 3 месяца, т.е. с 2020-09-01 по 2020-12-01.

Пожалуйста, сохраняйте файлы своих посылок и то, как они были получены, чтобы можно было заново отправить свою лучшую посылку и создать воспроизводимый пример для второго этапа соревнования.

Ограничение на частоту посылок - одна посылка раз в 10 минут. Ограничение на число посылок за всё соревнование - 100.

### Решение 
Исходный датасет с 4 признаками группировали и создали train dataset, в который входят 20 признаков, такие как:
* Кол-во привлечений в среднем в месяц/неделю
* Сколько дней проходит между первым-вторым, предпоследним-последним привлечением, медиана от всех разниц
* Сколько дней прошло с последнего/медианного привлечения
* Сколько дней партнер всего привлекает людей
* Сколько людей привлекли за последние 30/60/90/180/270/365 дней

В финальном решении использовались модели [CatBoost](https://catboost.ai) и [LightGBM](https://lightgbm.readthedocs.io/en/latest/)
Помимо этого были попытки использовать [XgBoost](https://xgboost.ai/). 

Для тюнинга моделей использовались алгоритмы GridSearchCV и optuna.

### Предсказание
Вы можем уже прямо сейчас создавать предсказания(на основе датасетов, похожих на те которые давались для решения задачи)
Для этого:
1. Скачайте репозиторий `git clone https://github.com/HghaVlad/alfacampus2024`
2. Установите все зависимости `pip3 install -r -requirements.txt`
3. Запустите файл `make_prediction.py`, который запросит путь на исходный датасет, дату отсчета, путь на выход.
4. Вы можете сразу передать необходимые аргуметны в команде `python3 main_prediction.py dataset.csv 2020-12-01 result.csv`
