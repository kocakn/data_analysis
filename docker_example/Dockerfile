FROM morpheo/sklearn-base
FROM tiangolo/uwsgi-nginx-flask:python3.8-alpine

WORKDIR /www/classifier

COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

COPY ./requirements.txt requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY train.py /www/classifier/train.py
COPY service.py /www/classifier/service.py

ENTRYPOINT ["docker-entrypoint.sh"]
