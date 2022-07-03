# global variable declaration
PYTHON = python3

main:
	$(PYTHON) 00_get_patient_info.py         # Read the raw data and extract features from the raw signal. 
	$(PYTHON) 01_get_cohort_statistics.py    # Read all the header metadata and construct a dataframe with information for all patients. Extract the differenct classes (e.g: 'healthy control') and store in a pickle file.       
	$(PYTHON) 02_eda.py                      # Using the metadata extracted @01_, perform explatory data analysis. Save images at the "images" dir.
	$(PYTHON) 03_data_preprocessing.py       # Preprocess the time series (smoothing with Gaussian kernal and Standarization). The time series are then saved as a numpy array per patient and record at the "preprocessed" dir.
	$(PYTHON) 04_modelling.py                # Perform binary classification for each ELECTRODE and for each available pathologies against the "healthy control" sub-cohort.
	$(PYTHON) 05_plot_model_results.py       # Plot the results of the modelling analysis as a HEATMAP.  