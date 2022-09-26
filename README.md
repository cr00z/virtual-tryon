# Virtual try-on
Виртуальная примерка 3D модели одежды на видео
(что-то в духе [Clometrica](https://www.clometrica.com/))

Выполнен как финальный проект на [первом семестре продвинутого потока](https://stepik.org/course/101721/info) курса "Deep Learning" [Школы глубокого обучения](http://dlschool.org/) ФПМИ МФТИ.

## Output

* [Штаны](https://github.com/cr00z/virtual-tryon/blob/master/output/out2.mp4)
* [Майка](https://github.com/cr00z/virtual-tryon/blob/master/output/out1.mp4)

## Описание

<img align="right" alt="demo2" src="https://raw.githubusercontent.com/cr00z/virtual-tryon/master/output/demo2.jpg" width="144" height="256"  border="1" hspace="5"/>
<img align="right" alt="demo1" src="https://raw.githubusercontent.com/cr00z/virtual-tryon/master/output/demo1.jpg" width="144" height="256" border="1" hspace="5"/>

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

## Установка (Colab)

Для установки и запуска демо можно использовать файл demo.ipynb. Загрузите его на Colab и выполните.

1. Clone repository
```
!git clone https://github.com/cr00z/virtual-tryon
!cp -r virtual-tryon/* .
```
2. Install detectron2 (for bbox detection)
```
!pip install 'git+https://github.com/facebookresearch/detectron2.git'
```
3. Install opendr
```
!sudo apt-get --yes install libglu1-mesa-dev freeglut3-dev mesa-common-dev
!sudo apt-get --yes install libosmesa6-dev
!pip install opendr
```
4. Install pytorch3d (for rendering)

*I ran on version 0.6.1, the old version 0.3.0 is installed on the colab by default, so we install from github*

Achtung! May take up to 10-20 minutes, please wait! 
```
%%time
pip -v install "git+https://github.com/facebookresearch/pytorch3d.git"
```
5. Install [Mesh](https://github.com/MPI-IS/mesh)
```
!sudo apt-get --yes install libboost-dev
!pip install 'git+https://github.com/MPI-IS/mesh.git'
```
6. Download extra data

Download the neutral SMPL model from http://smplify.is.tue.mpg.de/ and place it in the assets folder.
```
!echo "Download the neutral SMPL model"
!wget https://github.com/classner/up/raw/master/models/3D/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl -P assets

!echo "Downloading extra data from SPIN"
!wget http://visiondata.cis.upenn.edu/spin/data.tar.gz && tar -xvf data.tar.gz && rm data.tar.gz && mv data/smpl_mean_params.npz assets && rm -rf data

!echo "Downloading pretrained model"
!wget https://dl.fbaipublicfiles.com/eft/2020_05_31-00_50_43-best-51.749683916568756.pt -P assets

!echo "Download garment fts from MultiGarmentNetwork"
!wget https://github.com/bharat-b7/MultiGarmentNetwork/raw/master/assets/garment_fts.pkl -P assets
```
7. Download Multi-Garment dataset
```
!wget https://datasets.d2.mpi-inf.mpg.de/MultiGarmentNetwork/Multi-Garmentdataset.zip
!unzip Multi-Garmentdataset.zip
```

## Запуск

В файле main.py отредактировать путь к входному видеофайлу:
```
input_path = './sample_data/single_totalbody.mp4'
```
Run it:
```
!python main.py
```
Как забрать выходной файл output/out.mp4 через блокнот:
```
from IPython.display import FileLink
FileLink(r'output/out.mp4')
```

## Полезные ссылки:

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
