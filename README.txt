The code in this directory is intended for two separate purposes: 

1- Training an image recognition model on the LFW dataset. 
2- Running a web application to test the similarity application built on top of facet. 

# Training image recognition model
For training this model, you need to download the LFW dataset from: http://vis-www.cs.umass.edu/lfw/lfw.tgz
1- Install the required dependencies by running pip install -r requirements.txt
2- To start the training, run python train.py —-data_dir [pathToLFW/] 
3- The model architecture and loss should be displayed on the terminal. 
4- The model would be saved to model, you can load it using Keras for further experimentation. 

# Running the similarity application demo
To run a demo for the similarity application, make sure you install the required dependencies as indicated in point 1 in the previous section. 
1- Download the pertrained facnet model and unzip it in folder “[pathToapp]/model/“ from https://drive.google.com/file/d/0B5MzpY9kBtDVOTVnU3NIaUdySFE 
2- Add the facnet to python path: export PYTHONPATH=[pathToapp]/facenet-master/src/ 
3 Run the app using: 
export FLASK_APP=app.py
flask run
