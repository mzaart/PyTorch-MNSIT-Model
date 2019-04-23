# PyTorch MNSIT Handwritten Digits Demo


This project showcases PyTorch concepts by building 
a model for image classification on the classic MNSIT
Handwritten Digits data set.

A Dense Neural Network is chosen by performing Random Search
on different model and training hyper parameters.

## To run the code

- Install project requirements:

    `pip install -r requirements.txt`
    
- Type the following command to run the code:
    `python -m src`
    
(The code was tested with Python 3.5)


## Package Structure

- `data`: Contains classes for loading data sets including
Datasets, Transforms, DataSamplers and DataLoaders.

- `models`: Contains different models

- `training`: Contains different components for building
and training models
    - `scoring_funcs`: Contains functions used to measure a
    model's performance
    - `tuning`: Contains components used for model tuning
        - `parameters_domain`: Represents the domain where the model parameters live.
        - `scorers`: Measures the model's performance on a given data set.
        - `search`: Represents a searching algorithm for tuning a model. 
        
   