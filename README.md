# Классификатор людей в очках

## Описание
В этом небольшом проекте я буду классифицировать фотографий лиц людей (с любого ракуска: селфи, портретное фото и т.д.) на два класса: с очками на лице и без. <br/>
<br/>
 Задача классификаций будет решаться следуещим образом: запускаеться детектор лиц, который опредляет лица на картинке и вырезает их, далее вырезанные лица подаются в классфикатор, который уже выдает конечный ответ для каждого лица на картинке (есть ли очки или нет). В качестве алгоритма определения лиц использован MTCNN а код позаимстван с рапозитория: https://github.com/kuaikuaikim/DFace . Остальные части проекта напсаны мною.  <br/>
<br/>
 В качестве классификатора будут использоваться нексолько архитектур, такие как ResNet, MobileNetV2 и кастомная архитуектура OlzhasNet. Реализации первых двух архитектур взяты с офицального сайта PyTorch. Ниже в таблице приведены результаты по точности классификации (accuracy, F1) и скорости обработки одной картинки (в ms) каждой архитектуры и ее размеры (в MB).


## Библиотеки 
Pytorch 0.4.1, Sklearn, Numpy, OpenCV, Augmentor, argparse


## Данные 

Картинки людей в очках собраны с сайта Google. Для парсинга была использована библиотка google_images_download по нескольким поисковым запросам: "faces_with_glasses", "man_with_glasses", "people with glasses", "selfie with glasses female", "selfie with glasses male", "women with glasses". Далее я вручную отчистил картинки от мусора и в итоге получил около 3194 картинок. После аугментации количество картинок увеличелось до 7423. В качестве способов аугментции я использовал рандомное вырезание, поворот и горизонтальный поворот по  реккомендациям авторов следуещего пейпера: Data augmentation for face recognition (Jiang-JingLv et al).  
<br/>
Для парсинга:
```bash
$ python ImageCrawler.py [Arguments ...]
```
<br/>
Картиник лиц без очков скачаны со следующего истчника: Selfie Data Set, http://crcv.ucf.edu/data/Selfie/
<br/>

## Тренировка

Данные для тренировки и валидации можно скачать со облака: 
<br/>
Скачаннай данные поместите в проект в папку под названием Dataset и запустите следующий скрипт чтобы сгенерировать аннотационный файл:

```bash
$ python GenAnnoFile.py --dataset_dir <your_path>
```

Для тренировки ResNet запустите:
```bash
$ python Train.py --lr 0.005 --model_name ResNet --pretrained None
```

Оценка алгоритма
В качастве метрики на валидационном датасете были использованы Accuracy, Precision, Recall, F1. Последняя метрика были использована по причине того, что картинок одного класса (людей в очках) было гораздо меньше. Эта метрика позваоляет оценить баланс между полнотой и точностью нашей модели.

## Результаты
Натренированный модели нахадятся в папке weights.
Для визуализации работы алгоритма и вывода результов запустите:
```bash
$ python Predict.py  --model_name <Название модел: ResNet, MobileNet>  --model_path <Путь к натренированной модели>  
                     --test_dir <Путь к папке с тестовыми картинками> --vis <визуализация результатов: True, False>
```


|    Модель     | Accuracy      |    F1         | Скорость      | Вес          |
| ------------- | ------------- | ------------- | ------------- |------------- |
| ResNet18      | 94            | 85            | 6             |44.7          |
| MobileNetV2   | 94            | 83            | 16            |9.1           |
