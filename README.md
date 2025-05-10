# Named Entity Recognition
BIO Tagger using BiLSTM-CRF Model for Named Entity Recognition application in Natural Language Processing

## Installation
The `requirements.txt` file has all the dependencies needed for the successful execution of the project till the end. It assumes you are having a CUDA compatible GPU for GPU accelerated PyTorch training. While the code automatically handles CPU/GPU device usage based on availability, you can also modify the requirements file yourself according to your needs.

To install the dependencies, ideally in a separate venv or a conda environment run:

`pip install -r requirements.txt`

Which should pull all the needed libraries.

## Structure
The main execution flow is inside `src/main.ipynb` which offers an interactive jupyter notebook, which imports the other required local libraries from python files. Please check the documentation for each class and function for more information.