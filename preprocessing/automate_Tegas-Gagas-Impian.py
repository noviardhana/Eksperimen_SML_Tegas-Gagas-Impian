import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

def handle_outliers_iqr(dataframe, column):
    Q1 = dataframe[column].quantile(0.25)
    Q3 = dataframe[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    dataframe[column] = dataframe[column].clip(lower=lower_bound, upper=upper_bound)
    return dataframe

def preprocess_data():
    # Menentukan path secara dinamis berdasarkan lokasi file script ini
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir) # Naik satu level ke root
    
    file_path = os.path.join(root_dir, 'healthy_diet_calorie_intake.csv')
    output_path = os.path.join(script_dir, 'healthy_diet_calorie_intake_preprocessing.csv')
    
    print(f"Membaca data dari: {file_path}")
    df = pd.read_csv(file_path)
    
    intake_cols = ['Carbohydrate_Intake_g', 'Protein_Intake_g', 'Fat_Intake_g']
    valid_cols = [col for col in intake_cols if col in df.columns]
    
    df_clean = df.copy()
    if valid_cols:
        for col in valid_cols:
            df_clean = df_clean[df_clean[col] >= 0]
            
    numerical_cols = df_clean.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if 'Person_ID' in numerical_cols:
        numerical_cols.remove('Person_ID')
        
    for col in numerical_cols:
        df_clean = handle_outliers_iqr(df_clean, col)
        
    scaler = StandardScaler()
    df_scaled = df_clean.copy()
    df_scaled[numerical_cols] = scaler.fit_transform(df_scaled[numerical_cols])
    
    df_scaled = df_scaled.drop(columns=['Person_ID'], errors='ignore')
    
    target_mapping = {'Obese': 0, 'Underweight': 1, 'Overweight': 2, 'Healthy': 3}
    if 'Health_Status' in df_scaled.columns:
        df_scaled['Health_Status'] = df_scaled['Health_Status'].map(target_mapping)
            
    activity_mapping = {'Sedentary': 0, 'Lightly Active': 1, 'Moderately Active': 2, 'Very Active': 3, 'Athlete': 4}
    if 'Activity_Level' in df_scaled.columns:
        df_scaled['Activity_Level'] = df_scaled['Activity_Level'].map(activity_mapping)
            
    encode_cols = [col for col in ['Gender', 'Diet_Type'] if col in df_scaled.columns]
    df_final = pd.get_dummies(df_scaled, columns=encode_cols, drop_first=True)
    
    bool_cols = df_final.select_dtypes(include=['bool']).columns
    df_final[bool_cols] = df_final[bool_cols].astype(int)
    
    df_final.to_csv(output_path, index=False)
    print(f"Data berhasil diekspor menjadi '{output_path}'!")

if __name__ == '__main__':
    preprocess_data()

#Trigger: python automate_Tegas-Gagas-Impian.py

