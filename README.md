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
     (i.e: number of patients). Additionally, it loads the header metadata for each patient
     and stores this info in different directories.

     Importantly, for each patient and each recording, the script extracts the following information:

        1. The variance of each channel.
        2. The mean amplitude of each channel.
        3. The median amplirude of each channel.
        4. The mean value of the 1st derivative (how fast, on average
                                                 the data changes)
        5. The median value of the 1st derivative
        6. The peak of the power-spectral density for each lead. 

     For info on how these values are derived see the .pdf file that accompanies this repository.   

     !!!! This function runs in parallel and uses all threads. To change the 
     number of threads, see the variable "n_jobs" @config.py
