from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
import pandas as pd
import requests
import zipfile
import urllib
import os 
import logging as log

class dataset(Dataset):
    def __init__(self,name, transform=None,download=True,train=True):
        self.path = None
        if download:
            self.download_base()
        self.basecsv = self.path+'base.csv'
        self.dataframe = pd.read_csv(self.basecsv)
        self.data_type = 'Train' if train else 'Test' 
        log.info('Handle {} database'.format(self.data_type))
        self.imgs = self.dataframe[self.dataframe.images.str.match(self.data_type+'.*') == True].reset_index(drop=True)
        self.transform = transform
        self.classes = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']
        
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self,idx):   
        img = Image.open(self.path+'imgs/'+self.imgs.loc[idx,'images'], 'r')
        label = self.imgs.loc[idx,'labels']

        if self.transform:
            img = self.transform(img)

        return img, label

    def show(self,idx):
        img = Image.open(self.path+'imgs/'+self.imgs.loc[idx,'images'], 'r')
        plt.title('Sample #{}'.format(idx))
        plt.axis('off')
        plt.imshow(img)
        plt.show()
    
    def download_base(self):
        self.path = 'base/'
        if not os.path.exists('base'):
            url = "https://web.isen-ouest.fr/filez_web_root/asfq/download"
            log.info('Downloading database at : {}'.format(url))
            file = 'base.zip'    
            r = requests.get(url, stream=True, allow_redirects=True)
            total_size = int(r.headers.get('content-length'))
            initial_pos = 0
            with open(file,'wb') as f: 
                with tqdm(total=total_size, unit='it', unit_scale=True, desc=file,initial=initial_pos, ascii=True) as pbar:   
                    for ch in r.iter_content(chunk_size=1024):                            
                        if ch:
                            f.write(ch) 
                            pbar.update(len(ch))
            log.info('Unzip {} at : {}'.format(file,self.path))
            zip_ref = zipfile.ZipFile(file, 'r')
            zip_ref.extractall(self.path)
            zip_ref.close() 
            log.info('Removing {}'.format(file))  
            os.remove(file)
        else:
            log.info('database exist')


