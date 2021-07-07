# vkr-proctoring-algorithms

Исходный код дипломной работы на тему "Разработка алгоритма обработки видеопотока для системы прокторинга"

Требования: Python3.6+

### Предварительная установка

```bash
sudo apt-get update
sudo apt-get install build-essential cmake
sudo apt-get install libgl1-mesa-glx ffmpeg libsm6 libxext6
sudo apt-get install libopenblas-dev liblapack-dev libx11-dev libgtk-3-dev
sudo apt-get install libleptonica-dev libtesseract-dev tesseract-ocr
```

### Установка зависимостей

```bash
python3 -m venv venv
. ./venv/bin/activate

pip install -r requrements.txt
```

### Запуск

**Обработка видеопотока с веб-камеры**
Аргументы:

- -s, --source - тип источника данных, варианты: 'video' или 'image', соответственно обработка видеопотока или изображения
- -p, --path - путь до изображения/видеопотока
- -i, --image - путь до фотографии с лицом студента (для выделения целевых лицевых признаков)

Запуска скрипта:

```bash
python3 person_tracking.py -s video -p <path_to_video> -i <path_to_person_image>
```

**Обработка видеопотока с рабочего стола**
Аргументы:

- -s, --source - тип источника данных, варианты: 'video' или 'image', соответственно обработка видеопотока или изображения
- -p, --path - путь до изображения/видеопотока
- -t, --parallel - влючение распараллеливания при обработки

Запуска скрипта:

```bash
python3 screencast_detection.py -s video -p <path_to_video> --parallel
```
