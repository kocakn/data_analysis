FROM morpheo/sklearn-base

WORKDIR /www

COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

COPY ./requirements.txt requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY train.py /www/train.py
COPY app.py /www/app.py
COPY ./web/pages/* /www/templates/
COPY ./web/images/* /www/static/images/
COPY ./web/fonts/* /www/static/fonts/
COPY ./web/css/* /www/static/css/
COPY ./web/js/* /www/static/js/

ENTRYPOINT ["docker-entrypoint.sh"]
