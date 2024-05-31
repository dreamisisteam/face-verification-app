FROM python:3.10

WORKDIR /src
ADD ./ /src

RUN python3 -m pip install --upgrade pip
RUN pip install -r requirements.txt

RUN chmod +x run_api.sh

CMD [ "bash", "./run_api.sh" ]
