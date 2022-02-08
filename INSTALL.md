1. Clone repository:
'''
!git clone https://github.com/cr00z/virtual-tryon
!cp -r virtual-tryon/* .
'''
2. Install detectron2 (for :
'''
!pip install 'git+https://github.com/facebookresearch/detectron2.git'
'''
3. Install opendr:
'''
!sudo apt-get install libglu1-mesa-dev freeglut3-dev mesa-common-dev
!sudo apt-get install libosmesa6-dev
!pip install opendr
'''
4. Install pytorch3d (for rendering):
обкатывал на версии 0.6.1, на колабе по дефолту ставится старая версия 0.3.0, поэтому ставим с гитхаба
'''
!pip install "git+https://github.com/facebookresearch/pytorch3d.git"
'''
5. Install (Mesh)[https://github.com/MPI-IS/mesh]
'''
!sudo apt-get install libboost-dev
!pip install 'git+https://github.com/MPI-IS/mesh.git'
'''
6. Download extra data
* Download the neutral SMPL model from http://smplify.is.tue.mpg.de/ and place it in the assets folder.
'''
!echo "Download the neutral SMPL model"
!wget https://github.com/classner/up/raw/master/models/3D/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl -P assets

!echo "Downloading extra data from SPIN"
!wget http://visiondata.cis.upenn.edu/spin/data.tar.gz && tar -xvf data.tar.gz && rm data.tar.gz && mv data/smpl_mean_params.npz assets && rm -rf data

!echo "Downloading pretrained model"
!wget https://dl.fbaipublicfiles.com/eft/2020_05_31-00_50_43-best-51.749683916568756.pt -P assets

!echo "Download garment fts from MultiGarmentNetwork"
!wget https://github.com/bharat-b7/MultiGarmentNetwork/raw/master/assets/garment_fts.pkl -P assets
'''
7. Download Multi-Garment dataset
'''
!wget https://datasets.d2.mpi-inf.mpg.de/MultiGarmentNetwork/Multi-Garmentdataset.zip
!unzip Multi-Garmentdataset.zip
'''

??? !sudo apt-get install cmake libblkid-dev e2fslibs-dev libboost-all-dev libaudit-dev
