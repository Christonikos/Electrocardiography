# Analysis of the PTB ECG dataset.

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
     (i.e: number of patients). Additionally, it loads the metadata for each patient
     and stores this info in different directories.

     This allows us to build a processing pipeline that runs at the patient level
     and can be fully parallelized in the future.

     !!!! This function runs in parallel and uses all threads. To change the 
     number of threads, see the variable "n_jobs" @config.py
