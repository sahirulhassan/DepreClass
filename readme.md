# DepreClass – Depression Risk Classifier

A desktop application built with Python and PyQt6 that predicts a student's risk of depression based on academic,
lifestyle, and personal factors using logistic regression.

---

## Features

- PyQt6-based GUI
- Custom Logistic regression model with pre-trained scalers and mappings
- One-click prediction based on user input
- Packaged into a standalone `.exe` using PyInstaller

---

## Requirements

Install the following Python libraries before running or building the project:

- pandas
- joblib
- scikit-learn
- PyQt6

You can install all of them via:

```
pip install pandas joblib scikit-learn PyQt6
```

Note: Ensure you have Python installed
Note: If you are using the Anaconda distribution, you may not need to install pandas and scikit-learn as they are
included by default.


Run the application using:
```

python DepreClass.py

```

# Model Training
Open the `Group7.ipynb` in Jupyter Notebook to see the model training process.