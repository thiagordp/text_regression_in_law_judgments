# Regression applied to legal judgments to predict compensation for immaterial damage

Regression applied to judgement texts from JEC - UFSC to predict the compensation for immaterial damage related to failures in air transport services.

The paper is in *Prelo*.

## Scope

We aim to check many techniques related to text representation, regression and final evaluation.

In terms of text representation, we apply the Bag of Words with TF values

In terms of regression models, we apply the following:

- AdaBoost
- Decision Tree
- Elastic Net
- Ensemble Voting
- Gradient Boosting
- Bagging
- feed-forwared Neural Network
- Random Forest
- Ridge 
- Support Vector Machine
- XGBoosting

In terms of performance measures, we apply the following:

- R²
- Root Mean Square Error (RMSE)
- Mean Absolute Error (MAE)


## How to run this project?

- Install the dependencies by running `pip install -r requirements.txt`
- First download the [dataset](https://figshare.com/s/0248f634a74d317405ff)
- Update the file `util/path_constants.py` with the path where the dataset is stored in your computer
- Run the project using the command `python run_tests.py`
- This command does the following:
   - Data preparation
   - Run the experiments
- However, it will take dozens of days to finish
- When the execution finishes, copy all the terminal logging lines to the file `data/paper/running_logs/execution_log.log`
- Run the evaluation: `python run_evaluation.py`
