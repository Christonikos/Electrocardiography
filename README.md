# Analysis of the PTB ECG dataset.

:bangbang: The results and details about the analysis and the code structure can be found at the .pdf file <span style="color:blue"> *"idoven_results_presentation.pdf"* </span> that accompanies this repository.

-Code structure

The analysis is done in 5 different steps, each assigned to a separate script. To launch the analysis use the MAKEFILE and simply type in your terminal:
```
make main
```
The steps are the following: 
  1. Read the raw data and extract features from the raw signal. 
     To do that, use the script: 
     ```
     00_get_patient_info.py
     ```
     This script reads the patient info and logs information in the log file
     (i.e: number of patients). Additionally, it loads the header metadata for each patient
     and stores this info in different directories.

     Importantly, for each patient and each recording, the script extracts the following information:

        1. The variance of each channel.
        2. The mean amplitude of each channel.
        3. The median amplirude of each channel.
        4. The mean value of the 1st derivative 
        5. The median value of the 1st derivative
        6. The peak of the power-spectral density for each lead. 

     For info on how these values are derived see the .pdf file that accompanies this repository.   

     !!!! This function runs in parallel and uses all threads. To change the 
     number of threads, see the variable "n_jobs" @config.py

  2. Read all the header metadata and construct a dataframe with information for all patients. Extract the differenct classes (e.g: 'healthy control')        and store that in a pickle file. To do that, launch the script:
     ```
     01_get_cohort_statistics.py
     ```
  3. Using the metadata extracted @01_, perform explatory data analysis. Save images at the "images" dir.
     ```
     02_eda.py 
     ```
  4.  Preprocess the time series (smoothing with Gaussian kernal and Standarization). The time series are then saved as a numpy array per patient and           record at the "preprocessed" dir.
      ```
      03_data_preprocessing.py 
      ```
  5.  Perform univariate binary classification for each ELECTRODE and for each available pathologies against the "healthy control" sub-cohort.
      ```
      04_modelling.py 
      ```
  6.  Plot the results of the modelling analysis as a HEATMAP.
      ```
      05_plot_model_results.py 
      ```      
