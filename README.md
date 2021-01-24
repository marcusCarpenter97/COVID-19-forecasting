# The effects of regularization on RNN models for time series forecasting
This repository contains the code for the paper *The effects of regularization on RNN models for time series forecasting*.

## How to run the experiments?
`python3 cross_val_experiments.py`
This took approximately 6 hours to run on a laptop with an Intel Core i7-8665U 2GHz CPU and 16GB RAM on Windows 10.
All results and data files are saved in the folder `cross_val_results`. When run the programme will download the newest
data from the Jonhs Hopkins GitHub repository automatically. This was tested in a linux environment especifically the Ubuntu
18.04 WSL and may not work in other systems without modifying the `data_loader.py` module.

## Interactive result viewer.
Running the command `python3 result_viewer.py` will load an interactive result viewer that allows the user to select and plot
data from certain result files. There is also an option to visualise a comparison of the real data and a model's predictions for
each country.

## What is the `cross_val_results` folder?
All output files generated by the experiments will be saved here. The files provided were used to analyse the models in
the research paper. Rerunning the experiments will overwrite these. The file named `country_names.csv` is created by running the
command `python3 process_results.py`.

## What is the `kaggle_data` folder?
The population data for the countries is extracted from the data provided by
[Kaggle](https://www.kaggle.com/c/covid19-global-forecasting-week-5/).

## test_vX.ipynb
These are old experiments that build up to the current model. Not used in the research paper.
