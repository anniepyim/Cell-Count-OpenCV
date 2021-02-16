# Cell Count OpenCV
A simple webapp that performs image processing on .lif images with Python OpenCV library to count the number of cells

**Demo: https://cell-count-opencv.herokuapp.com/** 
## Flask App Deployment 

1) Please make sure you have conda installed - [Install Conda](https://docs.anaconda.com/anaconda/install/)

2) Execute the following command at your terminal:

```sh
cd ./Cell-Count-OpenCV # the directory to this project
conda create -n cell-count-opencv python=3.7 
conda activate cell-count-opencv
pip install -r requirements.txt
gunicorn wsgi:app
```

Then the webapp should be running at `http://127.0.0.1:8000` which could be accessible via any browser.