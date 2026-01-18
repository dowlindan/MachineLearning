# HW1: Principle Component Analysis
Dan Dowlin | djd439 | CS-383-001


## Description 
This is a script that takes 154 images and performs PCA using sklearn. It determines the smallest k such that the k largest eigenvalues constitute at least 95%. It projects subject02.centerlight onto the k most important principle components, then reconstructs the person. Finally the original and reconstructed images are displayed along with the k values.

## How to run
1. (Optional) Set and activate venv

 `python3 -m venv venv`

 `source ./venv/bin/activate`

2. Install requirements 

Note: You can probably remove pyqt6 from the requirements. I am not sure if the error I was seeing was because I am using WSL Ubuntu, but it fixed the problem.

`pip3 install -r requirements.txt`

3. Run script

`python3 ./HW1Attempt.py`

## Problems / Issues

None discovered