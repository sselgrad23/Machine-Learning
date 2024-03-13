First we load data from X_train, y_train, and X_test into Pandas DataFrames, create copies and NumPy arrays of the data, and extract labels into a list. Both X_train and X_test are preprocessed in the same way: For each sequence from the NumPy array we created from X_train or X_test, we remove NaN values, compute autocorrelation, average, peak-to-peak distance, and identify the top frequency components using FFT. We also pad the sequences so that they are the same length. For each preprocessed ECG signal, we call 'ecg' from biosppy.signals, which returns the time axis, filtered signal, R-peaks, template time axis, templates, heart rate time axis, and instantaneous heart rate. We normalise all templates and then compute the mean and median for each normalised template. Next we identify the R-peak in each preprocessed heartbeat and extract the P, Q, S, and T points. We extract template features (including the PR, QRS, and ST intervals), RR interval features, and whole wave features and create a data array consisting of these features. Our model uses three base classifiers: XGBClassifier, LGBMClassifier, and GradientBoostingClassifier. We use Nevergrad to find the optimal hyperparameters for all base classifiers. We create a voting classifier with the optimally initialised base classifiers and use soft voting. We train the ensemble model on the y_train and preprocessed X_train data. Finally, we make predictions on X_test and output the predictions to a csv file.