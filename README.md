# Fitzy ( Apparels Recommendation )

## About <a name = "about"></a>

This project made for apparels recommendation, We have trained the model with 44k image dataset using from kaggle, Here using Deep learning model Resnet50 CNN Model to extract image features and Using sklearn NearestNeighbors Algorithm to find the matching Images for Recommendation.

Project Hosted in AWS : [Project URL](http://fitzy.thanseefuddeen.xyz)

![newFitzy](https://user-images.githubusercontent.com/62167887/210856360-b25a6a57-a143-4a0e-9df3-1117dc472b6b.gif)

## Project is Docker containerized

Those who don't want to install the project on your device and install the requirements, please download the docker container to your device and run the project.

- [Docker Repository](https://hub.docker.com/r/thanseefpp/fitzy-flask)

### Run Project Docker Container

A step-by-step series of examples that tell you how to start this container.

1. pull the docker image

```
docker pull thanseefpp/fitzy-flask:0.0.1.RELEASE
```
2. Run the container
```
docker container run -d -p 4000:3000 thanseefpp/fitzy-flask:0.0.1.RELEASE
```
3. Go to the 127.0.0.1:4000 port on your machine to visit the app that docker has executed (if your device is using a 4000 port then change it to 5000 or 3000)

## Skip these below steps if you don't want to install Project on your Device.

### Prerequisites

Python3 Must be installed on your device.

### Installing

A step by step series of examples that tell you how to get a development env running.

1. Install Virtual Env

```
python3 -m pip install --user virtualenv pipenv
```

2. Initialize Env

```
python3 -m venv env
```
3. Activate Env
```
source env/bin/activate
```
4. Install Requirements

```
pip install -r requirements.txt
```

5. Create .env file

```
touch .env
```

6. Create .env file

```
touch .env
```

please follow my instagram [@thanseeftsf](https://www.instagram.com/thanseeftsf/) and ask any doubts.
