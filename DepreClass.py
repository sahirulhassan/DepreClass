import sys
import pandas as pd
import joblib
from PyQt6 import QtWidgets

from ui_depreclass import Ui_DepreClass


class DepreClass(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_DepreClass()
        self.ui.setupUi(self)
        self.numeric_scaler = joblib.load('Scalers/numericScaler.pkl')
        self.sleep_mapping = joblib.load('Mappings/sleep_map.pkl')
        self.diet_mapping = joblib.load('Mappings/diet_map.pkl')
        self.scaler_0_5 = joblib.load('Scalers/scaler_0_5.pkl')
        self.scaler_1_3 = joblib.load('Scalers/scaler_1_3.pkl')
        self.scaler_1_4 = joblib.load('Scalers/scaler_1_4.pkl')
        self.model = joblib.load('depression_model.pkl')
        self.ui.submitBtn.clicked.connect(self.handleSubmit)

    def handleSubmit(self):
        name = self.ui.nameField.text().strip()
        roll = self.ui.rollField.text().strip()
        age = self.ui.ageSpinner.value()
        cgpa = self.ui.cgpaSpinner.value()
        study_hours = self.ui.studyHrsSpinner.value()
        sleep_duration = self.ui.sleepDurationCombo.currentText()
        diet = self.ui.dietCombo.currentText()
        academic_pressure = self.ui.academicPressureSlider.value()
        financial_stress = self.ui.financialStressSlider.value()
        study_satisfaction = self.ui.studySatisfactionSlider.value()
        suicidal_thoughts = self.ui.suicidalThoughtsButtonGroup.checkedButton()
        fam_history = self.ui.famHistoryButtonGroup.checkedButton()
        # check if any of the fields are empty, we need all filled
        if not (name and roll and sleep_duration and suicidal_thoughts and fam_history):
            QtWidgets.QMessageBox.warning(self, "Input Error", "Please fill all fields.")
            return

        suicidal_thoughts = 1 if suicidal_thoughts.text() == "Yes" else 0
        fam_history = 1 if fam_history.text() == "Yes" else 0

        data = {
            "Academic Pressure": [academic_pressure],
            "Financial Stress": [financial_stress],
            "Sleep Duration": [sleep_duration],
            "Work Hours": [study_hours],
            "Suicidal Thoughts": [suicidal_thoughts],
            "Age": [age],
            "Dietary Habits": [diet],
            "Family History": [fam_history],
            "Name": [name],
            "Roll No": [roll],
            "CGPA": [cgpa],
            "Study Satisfaction": [study_satisfaction]
        }
        df = pd.DataFrame(data)

        # Transform 'Age' and 'Work Hours' (Study Hours)
        df[['Age', 'CGPA', 'Work Hours']] = self.numeric_scaler.transform(df[['Age', 'CGPA', 'Work Hours']])

        # Map and transform 'Sleep Duration'
        df['Sleep Duration'] = df['Sleep Duration'].map(self.sleep_mapping)
        df[['Sleep Duration']] = self.scaler_1_4.transform(df[['Sleep Duration']])

        # Map and transform 'Dietary Habits'
        df['Dietary Habits'] = df['Dietary Habits'].map(self.diet_mapping)
        df[['Dietary Habits']] = self.scaler_1_3.transform(df[['Dietary Habits']])

        # Transform 'Academic Pressure' and 'Financial Stress'
        df[['Academic Pressure', 'Financial Stress', 'Study Satisfaction']] = self.scaler_0_5.transform(
            df[['Academic Pressure', 'Financial Stress', 'Study Satisfaction']]
        )

        # Calculate 'Total Stress'
        df['Total_Stress'] = df['Academic Pressure'] + df['Financial Stress']

        # Construct the features numpy array (in desired order)
        features = df[['Suicidal Thoughts', 'Work Hours', 'Sleep Duration', 'Total_Stress',
       'Financial Stress', 'Academic Pressure', 'Age', 'Dietary Habits',
       'Family History']]
        print("Checkpoint 1")
        prediction = self.model.predict(features)
        print("Checkpoint 2")
        if prediction[0] == 1:
            result_text = "At risk of depression"
        else:
            result_text = "Not at risk"

        QtWidgets.QMessageBox.information(self, "Prediction Result", result_text)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = DepreClass()
    window.show()
    sys.exit(app.exec())
