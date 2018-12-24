# KERAS_SSD_object_detector
Однопроходный детектор (SSD) объектов на изображении. Реализован на основе обученной сети из https://github.com/pierluigiferrari/ssd_keras
```
Может обнаруживать классы 'background'(фон), 'bicycle'(велосипед), 'bus'(автобус), 'car'(автомобиль), 
'motorbike'(мотоцикл), 'person'(человек), 'train'(поезд)
```
# Requirements
```
Docker version 18.09.0
```
# Run server locally
Чтобы запустить сервер на Flask локально, нужно прописать в консоли следующую команду:
```
 export FLASK_APP=my_ssd.py
```
Затем для запуска такого сервера используется команда
```
 flask run
```
ВАЖНО: для того, чтобы подтянуть все зависимости в проекте лежит файл requirements.txt. Чтобы не было конфликтов версий библиотек, необходимо создать виртуальное окружение. В python для этого можно использовать venv в python или virtualenvwrapper. https://python-scripts.com/virtualenv Затем, чтобы установить необходимые зависимости нужно выполнить следующую команду в созданном виртуальном окружении
```
pip install -r requirements.txt
```
# Run server with Docker
Для того, чтобы запустить сервис с помощью Docker нужно сначала собрать Docker image:
```
cd KERAS_SSD_object_detector/app
docker build -t object_detector_keras_ssd:latest .
```
Затем чтобы запустить образ, нужно применить следующую команду:
```
docker run --name object_detector_keras_ssd -d -p 80:5000 --rm object_detector_keras_ssd
```
После запуска сервис будет доступен по адресу 0.0.0.0:80

# Usage
https://objectdetectorkerasssd.docs.apiary.io
