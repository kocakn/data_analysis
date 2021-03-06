## Решение задач по курсу "Анализ данных"

Учимся извлекать из данных пользу

* [Вспоминаем тервер и матстат](I_probability.ipynb)
    * [Домашняя работа: задачки по терверу](I_probability_hw_1_proba.ipynb)
    * [Домашняя работа: наивный байесовский классификатор](I_probability_hw_2_naive_bayes.ipynb)
    * [Домашняя работа: проверка статистических гипотез](I_probability_hw_3_stat.ipynb)
    * [Видео: введение в теорию вероятностей](https://www.youtube.com/watch?v=KPdBdblatC4)
    * [Видео: как считать retention](https://www.youtube.com/watch?v=Nds_9ZTihIY)
    * [Видео: домашка по Naive bayes. Метод fit](https://youtu.be/OYmha7NDsuA)
    * [Видео: домашка по Naive bayes. Метод predict](https://youtu.be/imhuUiodhL4)
* [Введение в ML](II_machine_learning_intro.ipynb)
    * [Домашняя работа: обучаем линейную регрессию](II_machine_learning_intro_hw.ipynb)
    * [Видео: основные понятия ML](https://youtu.be/mrI-k8ItnOY)
    * [Видео: жизненный цикл ML проекта, CRISP-DM](https://youtu.be/DnGtiUzn-9k)
* [Машинное обучение с учителем](III_machine_learning_supervised.ipynb)
    * [Домашняя работа: выбираем лучший классификатор](III_machine_learning_supervised_hw.ipynb)
    * [Видео: машинное обучение с учителем](https://youtu.be/05KpKpAKOus)
* [Машинное обучение без учителя](IV_machine_learning_unsupervised.ipynb)
    * [Домашняя работа: выбираем применяем снижение размерности](IV_machine_learning_unsupervised_hw.ipynb)
	* [Видео: машинное обучение без учителя](https://youtu.be/COcSkDuVU1g)
* [Методы обучения моделей: регуляризация, градиентный спуск](V_machine_learning_tuning.ipynb)
    * [Домашняя работа: реализуем градиентный спуск с регуляризацией](V_machine_learning_tuning.ipynb)
    * [Видео: регуляризация](https://youtu.be/cIa3ogbF9TY)
    * [Видео: градиентный спуск](https://youtu.be/9f1B_D5K_9o)
    * [Видео: инжениринг фичей](https://youtu.be/d5TjzF_MNWo)
* [Машинное обучение в продакшн. Рекомендательные системы](VI_machine_learning_production.ipynb)
    * [Домашняя работа: изучаем пакет implicit в задаче построения рекомендательных систем](VI_machine_learning_production.ipynb)
    * [Видео: метрики ML сервиса](https://youtu.be/XYiC9tgnebk)
    * [Видео: Docker для ML сервиса](https://youtu.be/K37RlsZhH8s)
    * [Видео: рекомендательные системы]( https://youtu.be/PwIYGIfwvWo)

Сборка контейнера для работы Flask:
```shell
docker build --no-cache . -f Dockerfile -t flask-app
```

Запуск контейнера для взаимодействия с сервисом с помощью Flask:
```shell
docker run -it --rm -v $(pwd)/data:/www/data -p 5000:5000 -d flask-app start_service
```
