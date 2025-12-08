from torch.utils.data import Dataset 
from tqdm import tqdm
import pandas as pd 
import pyrootutils 
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
import torch

class DiabetesDataset(Dataset): 

    def __init__(self, data : pd.DataFrame): 
        self.data = data 
        self.data = self._preprocess()



    def __len__(self): 

        return len(self.data)
    
    def __getitem__(self, idx): 
        sample = self.data.iloc[idx]
        y = torch.tensor(sample['diagnosed_diabetes'], dtype=torch.float32)
        x = sample.drop('diagnosed_diabetes').values
        x = torch.tensor(x, dtype=torch.float32)
        return x, y


    def _preprocess(self): 

        df_encoded = self.data.copy()
        education_mapping = {
            'No formal': 0,
            'Highschool': 1,
            'Graduate': 2,
            'Postgraduate': 3
        }
        df_encoded['education_level'] = self.data['education_level'].map(education_mapping)


        income_mapping = {
            'Low': 0,
            'Lower-Middle': 1,
            'Middle': 2,
            'Upper-Middle': 3,
            'High': 4
        }
        df_encoded['income_level'] = self.data['income_level'].map(income_mapping)

        smoking_mapping = {
            'Never': 0,
            'Former': 1,
            'Current': 2
        }
        df_encoded['smoking_status'] = self.data['smoking_status'].map(smoking_mapping)

        df_encoded = pd.get_dummies(df_encoded, columns=['gender'], prefix='gender', drop_first=False)

        # employment_status - nominal, ít categories (4 values)
        df_encoded = pd.get_dummies(df_encoded, columns=['employment_status'], prefix='employment', drop_first=False)

        # ethnicity - nominal, ít categories (5 values)
        df_encoded = pd.get_dummies(df_encoded, columns=['ethnicity'], prefix='ethnicity', drop_first=False)

        return df_encoded



if __name__ == "__main__":


    pyrootutils.set_root(__file__, indicator=".project-root", pythonpath=True)
    path = pyrootutils.find_root(__file__, indicator=".project-root") / "data" / "raw" / "diabetes.csv"

    data = pd.read_csv(path)

    dataset = DiabetesDataset(data)
    first_sample =  dataset[0]

