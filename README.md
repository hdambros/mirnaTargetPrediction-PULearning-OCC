# miRNA target prediction with PU Learning and One-Class Classification #

This code uses machine learning in the form of two PU Learning techiniques (Two-Step and PU Bagging) and two One-Class Classification techiniques (One-Class SVM and Isolation Forest) for predicting functional and non-functional targets in the problem of miRNA target Prediction. Two supervised methods (Random Forest and SVM) are also used for comparison to the results obtained with PU Learning and OCC.

### Python libraries needed ###
  - python=3.7
  - pandas
  - numpy
  - matplotlib
  - scikit-learn
  - imbalanced-learn
  
  ### Dataset ###
  The dataset used can be found at https://drive.google.com/open?id=1SPVYiqNMeOiwFasTUHtiFDx81ji_xCYb
  
  Only the .att file is needed and it should be added to a "datasets" folder inside the main folder.
  
  ### Files generated ###
  All files generated on each run will be stored inside a "executions" folder.
  
  ### Startup File ###
  The main script file that should be run is the "tarbasePU.py" file.
