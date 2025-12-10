from torch.utils.data import DataLoader
import lightning as pl 
import pandas as pd 
from src.data.component.dataset import DiabetesDataset
import warnings 

warnings.filterwarnings("ignore")

class DiabetesDataModule(pl.LightningDataModule): 

    def __init__(self, train_dir: str, batch_size=32, num_workers=4): 
        super(DiabetesDataModule, self).__init__()
        self.save_hyperparameters(logger = False)

    

    def setup(self, stage= None): 

        data_df = pd.read_csv(self.hparams.train_dir)

        train_size = int(0.9 * len(data_df))
        val_size = int(0.1 * len(data_df))

        train_df = data_df[:train_size]
        val_df = data_df[train_size:]

        if stage == 'fit' or stage is None: 
            self.train_dataset = DiabetesDataset(train_df)
            self.val_dataset = DiabetesDataset(val_df)
        
        if stage == 'test' or stage is None:
            self.test_dataset = DiabetesDataset(val_df)

    def train_dataloader(self): 
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=self.hparams.num_workers)


    def val_dataloader(self): 
        return DataLoader(self.val_dataset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.num_workers)

    def test_dataloader(self): 
        return DataLoader(self.test_dataset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.num_workers)



if __name__ == "__main__": 
    pass



