# Virtual try-on
Виртуальная примерка 3D модели одежды на видео
(что-то в духе [Clometrica](https://www.clometrica.com/))

Выполнен как финальный проект на [первом семестре продвинутого потока](https://stepik.org/course/101721/info) курса "Deep Learning" [Школы глубокого обучения](http://dlschool.org/) ФПМИ МФТИ.

<img align="right" alt="demo" src="https://raw.githubusercontent.com/cr00z/virtual-tryon/master/output/demo.jpg" width="369" height="648" />

## Установка

```
pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

### Запуск

В файле main.py отредактировать путь к входному видеофайлу, я использовал демо из frankmocap:
```
input_path = './sample_data/single_totalbody.mp4'
```


## Описание

### Основная часть (пайплайн):

- берется готовый меш одежды
- на вход приходит видео человека в полный рост (перед зеркалом/снятого от 3-го лица)
- предсказывается 3D-поза человека по этому видео (покадрово)
- далее на каждом кадре видео:
- поза конвертируется в нужный формат для перепозирования 3D модели одежды 
- перепозированный меш одежды рендерится (отрисовывается) поверх картинки

### Задача:

1) Изучить рекомендованный материал.
2) Скачать и настроить датасет с 3D-одеждой (лучше оба), которая позволяет менять свою позу.
3) Придумать решение проблемы адаптации формы одежды под конкретное тело человека.
4) Запустить и добиться корректности работы хотя бы 1-го метода предсказания 3D позы человека по видео.
5) Реализовать конвертацию формата 3D позы человека в 3D позу меша одежды.
6) Реализовать рендеринг (отрисовку) 3D модели одежды поверх видео.

### Полезные ссылки:

1. Статьи на русском про 3DML:
   * https://m.habr.com/ru/company/itmai/blog/503358/
2. Рассказ про моделирование 3D одежды:
   * https://youtu.be/ySx-iE-mb0s 
3. Модель камеры:
   * http://www.cse.psu.edu/~rtc12/CSE486/lecture12.pdf
   * http://www.cse.psu.edu/~rtc12/CSE486/lecture13.pdf 
4. Библиотеки:
   * pytorch3d: https://pytorch3d.org
   * https://pytorch3d.org/tutorials/render_textured_meshes
   * open3d: http://www.open3d.org
   * pyrender: https://github.com/mmatl/pyrender [опционально]
5. SMPL: 
   * https://www.youtube.com/watch?v=kuBlUyHeV5U
   * https://khanhha.github.io/posts/SMPL-model-introduction/ 
6. Предсказание 3D-позы: 
   * https://github.com/facebookresearch/frankmocap
   * https://github.com/vchoutas/expose [опционально]
   * https://google.github.io/mediapipe/solutions/pose [опционально]
7. Датасеты мешей одежды: 
   * https://github.com/bharat-b7/MultiGarmentNetwork
   * https://github.com/jby1993/BCNet [опционально] 
   * https://github.com/aymenmir1/pix2surf [опционально]