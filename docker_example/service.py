"""
Умеет выполнять классификацию клиентов по трём фичам

Запускаем из python3:
    python3 service.py
Проверяем работоспособность:
    curl http://127.0.0.1:5000/
"""
import json
import http.server
import logging
import os
import pickle
import socketserver
import sys
from http import HTTPStatus
from re import compile

import numpy as np
from sklearn.tree import DecisionTreeClassifier

# файл, куда посыпятся логи модели

LOG_FORMAT = '%(asctime)s | %(levelname)-8s | %(filename)-25.25s:%(lineno)-4d | %(message)s'
# logging.basicConfig(filename="/www/classifier/data/service.log", level=logging.INFO, format=LOG_FORMAT)

logFormatter = logging.Formatter(LOG_FORMAT)
rootLogger = logging.getLogger()

fileHandler = logging.FileHandler("/www/classifier/data/service.log")
fileHandler.setFormatter(logFormatter)
rootLogger.addHandler(fileHandler)

consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
rootLogger.addHandler(consoleHandler)


def parse_params(params) -> dict:
    """
    Получаем и трансформируем
    параметры из GET-запроса
    """
    params_list = params.split('&')
    params_dict = {'x1': None, 'x2': None, 'x3': None}
    for param in params_list:
        key, value = param.split('=')
        params_dict[key] = float(value)
    params_full = np.array(list(params_dict.values()))  # все параметры
    params_pca = pca.transform(params_full.reshape(1, -1)).flatten()  # сжимаем до 2 параметров
    params_dict_pca = {'x1_pca': params_pca[0], 'x2_pca': params_pca[1]}  # создаем новый словарь
    return params_dict_pca


class Handler(http.server.SimpleHTTPRequestHandler):
    """Простой http-сервер"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_response(self) -> dict:
        """Пример запроса
        
        http://0.0.0.0:5000/classifier/?x1=1&x2=-2.2&x3=1.05
        """
        response = {'ping': 'ok'}
        params_parsed = self.path.split('?')
        if len(params_parsed) == 2 and self.path.startswith('/classifier'):
            params = params_parsed[1]
            params_dict = parse_params(params)
            response = params_dict
            user_features = np.array([params_dict['x1_pca'], params_dict['x2_pca']]).reshape(1, -1)
            predicted_class = int(classifier_model.predict(user_features)[0])
            logging.info('predicted_class %s' % predicted_class)
            response.update({'predicted_class': predicted_class})
        elif self.path.startswith('/ping/'):
            response = {'message': 'pong'}

        return response

    def do_GET(self):
        # заголовки ответа
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(self.get_response()).encode())


logging.info('Загружаем объект для трансформации')
with open('/www/classifier/pca_transformer.pkl', 'rb') as f:
    pca = pickle.load(f)
    logging.info('Объект загружен: %s' % pca)

logging.info('Загружаем обученную модель')
with open('/www/classifier/clf.pkl', 'rb') as f:
    classifier_model = pickle.load(f)
    logging.info('Модель загружена: %s' % classifier_model)

if __name__ == '__main__':
    classifier_service = socketserver.TCPServer(('', 5000), Handler)
    classifier_service.serve_forever()
