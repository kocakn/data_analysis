from flask import Flask, request, render_template, redirect
import logging
import pickle
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
app = Flask(__name__)

def load_model():
    # файл, куда посыпятся логи модели
    LOG_FORMAT = '%(asctime)s | %(levelname)-8s | %(filename)-25.25s:%(lineno)-4d | %(message)s'

    logFormatter = logging.Formatter(LOG_FORMAT)
    rootLogger = logging.getLogger()

    fileHandler = logging.FileHandler("/www/data/service.log")
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)

    logging.info('Загружаем объект для трансформации')
    with open('/www/pca_transformer.pkl', 'rb') as f:
        pca = pickle.load(f)
        logging.info('Объект загружен: %s' % pca)

    logging.info('Загружаем обученную модель')
    with open('/www/clf.pkl', 'rb') as f:
        classifier_model = pickle.load(f)
        logging.info('Модель загружена: %s' % classifier_model)
    return pca, classifier_model


@app.route('/classifier/')
def launch_classifier():
    try:
        param_x1 = float(request.args.get('x1'))
        param_x2 = float(request.args.get('x2'))
        param_x3 = float(request.args.get('x3'))
    except TypeError:
        param_x1 = (np.random.randint(100) - 50) / 10
        param_x2 = (np.random.randint(100) - 50) / 10
        param_x3 = (np.random.randint(100) - 50) / 10
    params_dict = {'x1': param_x1, 'x2': param_x2, 'x3': param_x3}

    pca, classifier_model = load_model()

    params_full = np.array(list(params_dict.values()))  # все параметры
    params_pca = pca.transform(params_full.reshape(1, -1)).flatten()  # сжимаем до 2 параметров
    params_dict_pca = {'x1_pca': params_pca[0], 'x2_pca': params_pca[1]}  # создаем новый словарь
    result = params_dict_pca

    user_features = np.array([params_dict_pca['x1_pca'], params_dict_pca['x2_pca']]).reshape(1, -1)
    predicted_class = int(classifier_model.predict(user_features)[0])
    result.update({'predicted_class': predicted_class})
    logging.info('predicted_class %s' % predicted_class)
    return render_template('classifier.html', input_variables=params_dict, results=result)


@app.route('/ping')
def pong_response():
    return 'pong'


@app.route('/hello')
def hello_world():
    return 'Hello, people. Every human is awesome'


@app.route('/english')
def english():
    return redirect("https://www.youtube.com/watch?v=HbvYeLxMKN8")


@app.route('/quokka')
def quokka_pictures():
    return redirect("https://www.instagram.com/explore/tags/quokka/")


@app.route('/')
def welcome_page():
    return render_template('index.html')


@app.route('/index.html')
def index_page():
    return render_template('index.html')


@app.route('/contact.html')
def contact_page():
    return render_template('contact.html')


@app.route('/portfolio.html')
def portfolio_page():
    return render_template('portfolio.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0')