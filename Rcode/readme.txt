### --------------------------------------------------------------------------
### R Code: Technical details
### --------------------------------------------------------------------------
### 
### R version 3.6.3
### Platform: x86_64-w64-mingw32/x64 (64-bit)
### Running under: Windows 10 x64 (build 18362)
### Intel(R) Core(TM) i7 - 8550U CPU @ 1.80GHz 1.99GHz - RAM 16 GB
###  
### --- Technical info about packages for LSTM implementation
### 
### Keras version (R interface): 2.2.4.1
### Tensorflow version (R interface): 1.14.1 
### Tensorflow version (Python module for backend tensor operations): 1.10.1 (Python module)
### Python version : 3.6
### Conda version : 4.5.11
### 
### Data source : Mortality Rates from Human Mortality Database
### ---------------------------------------------------------------------------

df_f.RData = Life Table female population
df_m.RData = Life Table male population
H_Parameter.RData = Best hyperparameters obtained during the training phases


A) DNN estimation ->  Script useful to obtain DNA estimation leveraging on the chosen best hyperparameters
   A1) fit_model.R
   A2) load fun_DNN.R

B) DNN training ->  Script useful to train DNN, exploring hyperparameter space, in order to minimize the out of sample error. 
In order to reduce the probability to run into a local minimum, a wide space is desirable
   B1) fit_autoning.R
   B2) FUN_AutoTuning
 