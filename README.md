# Tensorflow SRCNN

## Setup
Generate Virtual environment
```bash
python3 -m venv ./venv
```
Enter environment
```bash
source venv/bin/activate
```
Install required libraries
```bash
pip install -r requirements.txt
```
Install Dataset [UKbench100](https://drive.google.com/file/d/17B0_EVUMFWG96XwHOsnrR3KsLn7O1xWI/view?usp=sharing)

Setup config file to dataset.

Build project
```bash
python3 build.py
```
Train model
```bash
python3 train.py
```
Test model 
```bash
python3 test.py -i <path to ukbench100 dataset>/ukbench00118.jpg
```
