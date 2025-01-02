# Facial Recognition Pipeline with Deep Learning in TensorFlow

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
- [Data Preparation](#data-preparation)
- [Preprocessing](#preprocessing)
- [Generating Embeddings](#generating-embeddings)
- [Training the Classifier](#training-the-classifier)
- [Evaluation](#evaluation)
- [References](#references)

## Overview

This project implements a facial recognition pipeline using TensorFlow, Dlib, and Docker. The pipeline encompasses:

1. **Preprocessing Images**: Detecting and aligning faces in images using Dlib to ensure consistency.
2. **Generating Facial Embeddings**: Utilizing a pre-trained convolutional neural network to extract 128-dimensional feature vectors (embeddings) representing each face.
3. **Training an SVM Classifier**: Using the embeddings to train an SVM classifier that can distinguish between different identities.

## Prerequisites

Before running the project, ensure you have the following:

- Basic understanding of Linear Algebra
- Basic understanding of Convolutional Neural Networks
- Basic knowledge of TensorFlow

## Project Structure

```
├── Dockerfile
├── etc
│   ├── 20170511-185253
│       ├── 20170511-185253.pb
├── data
├── medium_facenet_tutorial
│   ├── align_dlib.py
│   ├── download_and_extract_model.py
│   ├── __init__.py
│   ├── lfw_input.py
│   ├── preprocess.py
│   ├── shape_predictor_68_face_landmarks.dat
│   └── train_classifier.py
├── requirements.txt
```

## Setup and Installation

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/YourUsername/your-repo-name.git
   cd your-repo-name
   ```

2. **Install Docker**:

   Ensure Docker is installed on your system. You can install it by running:

   ```bash
   curl -sSL https://get.docker.com/ | sh
   ```

3. **Build the Docker Image**:

   ```bash
   docker build -t facial_recognition_pipeline -f Dockerfile .
   ```

   *Note*: Building the image may take several minutes, depending on your hardware.

## Data Preparation

1. **Download the LFW Dataset**:

   The Labeled Faces in the Wild (LFW) dataset will be used for training. Download and extract it as follows:

   ```bash
   curl -O http://vis-www.cs.umass.edu/lfw/lfw.tgz
   tar -xzvf lfw.tgz
   ```

   *Note*: Verify the integrity of the downloaded dataset by comparing its checksum with the one provided on the official website to avoid using corrupted files.

   Ensure the dataset is structured as:

   ```
   data/
   ├── person1
   │   ├── image1.jpg
   │   └── image2.jpg
   ├── person2
   │   ├── image1.jpg
   │   └── image2.jpg
   ...
   ```

## Preprocessing

1. **Download Dlib's Face Landmark Predictor**:

   ```bash
   curl -O http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
   bzip2 -d shape_predictor_68_face_landmarks.dat.bz2
   ```

2. **Run the Preprocessing Script**:

   Use the provided script to detect, align, and crop faces from the dataset:

   ```bash
   docker run -v $PWD:/facial_recognition_pipeline \
   -e PYTHONPATH=$PYTHONPATH:/facial_recognition_pipeline \
   -it facial_recognition_pipeline python3 /facial_recognition_pipeline/medium_facenet_tutorial/preprocess.py \
   --input-dir /facial_recognition_pipeline/data \
   --output-dir /facial_recognition_pipeline/output/intermediate \
   --crop-dim 180
   ```

## Generating Embeddings

1. **Download the Pre-trained Model**:

   Execute the script to download the pre-trained FaceNet model:

   ```bash
   docker run -v $PWD:/facial_recognition_pipeline \
   -e PYTHONPATH=$PYTHONPATH:/facial_recognition_pipeline \
   -it facial_recognition_pipeline python3 /facial_recognition_pipeline/medium_facenet_tutorial/download_and_extract_model.py \
   --model-dir /facial_recognition_pipeline/etc
   ```

2. **Generate Embeddings**:

   Run the script to produce embeddings for each aligned face image.

## Training the Classifier

1. **Train the SVM Classifier**:

   Train the classifier using the generated embeddings:

   ```bash
   docker run -v $PWD:/facial_recognition_pipeline \
   -e PYTHONPATH=$PYTHONPATH:/facial_recognition_pipeline \
   -it facial_recognition_pipeline \
   python3 /facial_recognition_pipeline/medium_facenet_tutorial/train_classifier.py \
   --input-dir /facial_recognition_pipeline/output/intermediate \
   --model-path /facial_recognition_pipeline/etc/20170511-185253/20170511-185253.pb \
   --classifier-path /facial_recognition_pipeline/output/classifier.pkl \
   --num-threads 16 \
   --num-epochs 25 \
   --min-num-images-per-class 10 \
   --is-train
   ```

   *Note*: Adjust `--num-threads` and `--num-epochs` based on your system's capabilities.

## Evaluation

1. **Evaluate the Classifier**:

   After training, evaluate the classifier's performance.

## References

- [Building a Facial Recognition Pipeline with Deep Learning in TensorFlow](https://hackernoon.com/building-a-facial-recognition-pipeline-with-deep-learning-in-tensorflow-66e7645015b8)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Dlib Documentation](http://dlib.net/)