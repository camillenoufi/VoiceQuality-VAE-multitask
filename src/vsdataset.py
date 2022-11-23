import os.path as osp

import torch
import pandas as pd
import torchaudio
from tqdm import tqdm
from torch.utils.data import Dataset
import crepe
import numpy as np

# generic sofa dataset
class VocalSetDataset(Dataset):
    def __init__(self, 
                dataset_path,
                measurements_filename, 
                transform=None, 
                augm_transform=None, 
                sr=44100, 
                duration=1,
                device='cpu'
                ):
        self.dataset_path = dataset_path
        self.measurements_filename = measurements_filename
        self.transform = transform
        self.augm_transform = augm_transform
        self.sr = sr
        self.duration = int(sr * duration)
        self.vq_measurements = self._list_of_vq_measurements()
        self.auxiliary_features_lengths = None
        self.feat_stats = {}
        self.df = self._load_data()
        print('VSD initialized')

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx: int):
        # load audio
        item = self.df.iloc[idx]
        name = self.df_names[idx]
        filepath = osp.join(self.dataset_path, f'{name}.wav')
        sample, sr = torchaudio.load(filepath)
        if sr != self.sr:
            sample = torchaudio.functional.resample(sample, sr, self.sr)
        assert sample.shape[0] == 1 #assert mono file
        paddings = (0, self.duration - sample.shape[1]) #pad file with 0s if shorter
        sample_clean = sample_noisy = torch.nn.functional.pad(sample, paddings)
        
        # apply transformations and augmentations, if applicable
        if self.transform:
            sample_clean = self.transform(sample_clean)
        if self.augm_transform:
            sample_noisy = self.augm_transform(sample_noisy)
        else:
            sample_noisy = self.transform(sample_noisy)
        
        # get labels
        vqm = item[self.vq_measurements].to_dict()  #dict containing 12 VQM values
        frequency = [0] #crepe.predict(sample, sr, viterbi=True, model-capacity='small', step-size=100) #get 10 frequency values per second of audio
        metadata = self._get_sample_labels(name)
        label = (vqm, frequency, metadata)
        self.auxiliary_features_lengths = (len(self.vq_measurements), len(frequency), len(metadata))
        return sample_noisy, sample_clean, label

    def _get_data_frame(self):
        return self.df
    
    def _load_data(self):
        df = self._read_measurements_file()
        assert list(df.columns) == self.vq_measurements
        self.vq_measurements = self.vq_measurements[1:] #chop off filename
        #normalize features
        df = self._convert_to_float(df)
        self.df_names = df['Filename']
        df = self._normalize_features(df)
        ###### TO - DO:
        # convert skewed data to normal distributions: Box-Cox method
        # create an inverse transform as well
        # probably do this within normalize_features()
        return df

    def _read_measurements_file(self):
        #read in measurements excel file produced by VoiceLab
        xls = pd.ExcelFile(osp.join(self.dataset_path, self.measurements_filename))
        df = pd.read_excel(xls, 'Summary') #all measurements in Summary sheet
        df = df[df.columns.intersection(self.vq_measurements)]
        return df

    def _convert_to_float(self,df):
        # reformat to have all valid values in measurement cols
        for i,col in enumerate(df.columns):
            if i==0:
                pass
            else:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.replace(np.nan, 0, regex=True)
        # if self.normtype == 'mean':
        #     df = df.replace(np.nan, 0, regex=True)
        # else:
        #     df = df.replace(np.nan, 0.5, regex=True) #minmax normalization
        return df

    def _normalize_features(self, df):
        # apply minmax-normalization to columns
        df = df.iloc[:,1:] #drop filename col
        self.feat_stats['mean'] = df.mean()
        self.feat_stats['std'] = df.std()
        self.feat_stats['min'] = df.min()
        self.feat_stats['max'] = df.max()
        # if self.normtype=='mean': #default is minmax
        print('Applying mean-normalization to all features')
        normalized_df=(df-self.feat_stats['mean'])/self.feat_stats['std']
        # else:
        #     print('Applying minmax-normalization to all features')
        #     normalized_df=(df-self.feat_stats['min'])/(self.feat_stats['max']-self.feat_stats['min'])
        return normalized_df


    def _unnormalize_features(self):
        # if self.normtype == 'mean':
        df = self.df*self.feat_stats['std'] + self.feat_stats['mean']
        # elif self.normtype == 'minmax':
        #     df = self.df*(self.feat_stats['max'] - self.feat_stats['min']) + self.feat_stats['min']
        return df

    # Measurements of interest specific to voice quality / timbre
    def _list_of_vq_measurements(self):
        return ['Filename',
                 'subharmonic-to-harmonic ratio',
                 'Subharmonic Mean Pitch',
                 'Harmonics to Noise Ratio',
                 'Local Jitter',
                 #'Local Absolute Jitter',
                 #'RAP Jitter',
                 #'ppq5 Jitter',
                 #'ddp Jitter', #'PCA Result', <- causing error
                 #'local_shimmer',
                 'localdb_shimmer',
                 #'apq3_shimmer',
                 #'aqpq5_shimmer',
                 #'apq11_shimmer',
                 #'dda_shimmer',
                 #'PCA Result.1',
                 'cpp',
                 'Spectral Tilt',
                 'Centre of Gravity',
                 'Standard Deviation',
                 #'Kurtosis',
                 'Skewness',
                 'Band Energy Difference',
                 'Band Density Difference']

    ##### The following methods get information about the dataset
    def _get_sample_name(self, index):
        return self.df_names.iloc[index] #index of row

    def _get_sample_measurements(self, index):
        row = self.df.iloc[index].tolist()
        return torch.tensor(row)
    
    def get_statistics(self):
        all_data = []
        for i in tqdm(range(self.__len__())):
            _, sample, _ = self.__getitem__(i)
            all_data.append(sample)
        all_data = torch.stack(all_data)
        return all_data.mean(), all_data.std()

    def get_average_datapoint(self):
        all_data = []
        for i in tqdm(range(self.__len__())):
            _, sample, _ = self.__getitem__(i)
            all_data.append(sample)
        all_data = torch.stack(all_data)
        return all_data.mean(0), all_data.std(0)

    ##### The following methods create Corresponding Labels from VocalSet Filename Metadata
    def _removeItem(self,item, labels):
        if item in labels: labels.remove(item)
        return labels

    def _replaceItem(self,item1,item2,labels):
        if(item1 in labels):
            labels[labels.index(item1)] = item2
        return labels

    def _combine(self,item1, item2, labels):
        if((item1 in labels) and (item2 in labels)):
            combo = item1+item2
            labels[labels.index(item1)] = combo
            labels.remove(item2)
        return labels

    def _splitItem1(self,labels):
        labels.insert(0,labels[0][0])
        labels[1] = labels[1][1:]
        return labels

    def _cleanUpLabels(self,labels):
        labels = self._removeItem('c',labels)
        labels = self._removeItem('f',labels) #do before gender, singer separation!
        labels = self._replaceItem('u(1)','u',labels)
        labels = self._replaceItem('a(1)','a',labels)
        labels = self._replaceItem('arepggios','arpeggios',labels)
        labels = self._combine('fast','forte',labels)
        labels = self._combine('fast','piano',labels)
        labels = self._combine('slow','piano',labels)
        labels = self._combine('slow','forte',labels)
        labels = self._combine('lip','trill',labels)
        labels = self._combine('vocal','fry',labels)
        labels = self._splitItem1(labels)
        return labels

    def _parseLabels(self,filename):
        # info,ext = os.path.splitext(filename)
        # if ext=='.csv':
        #     info = os.path.splitext(info)[0] #remove crepe .f0 tag
        filename = filename[:-3]
        lbl = filename.split("_") #known delimiter
        return self._cleanUpLabels(lbl)

    def _get_sample_labels(self,name):
        lbls = self._parseLabels(name)
        #print(lbls)
        gender = lbls[0]
        singer = lbls[1]
        phrase = lbls[2]
        technique = lbls[3]
        try:
            vowel = lbls[4]
        except:
            vowel = 'N'
#         print("labels generated")
        return gender, singer, phrase, technique, vowel



####################### Old code for Nsynth: #############################################
# def get_n_classes(self):
#     return len(self.df['instrument'].unique())

# def load_data(self):
#     filepath_cache = osp.join(self.dataset_path, 'examples_cache.pkl')
#     if osp.exists(filepath_cache):
        #print(f'Loading cached data: {filepath_cache}')
    #     self.df = pd.read_csv(filepath_cache)
    # else:
    #     print('filepath not found')
        # filepath = osp.join(self.dataset_path, 'examples.json')
        # #print(f'Caching data: {filepath}')
        # _df = pd.read_json(filepath).T
        # _df.to_pickle(filepath_cache)
    # filter data
    # if self.frequencyes:
    #     _df = _df[_df['frequency'].isin(self.frequencyes)]
    # if self.velocities:
    #     _df = _df[_df['velocity'].isin(self.velocities)]
    # if self.instrument_sources:
    #     _df = _df[_df['instrument_source'].isin(self.instrument_sources)]
    # if self.instrument_families:
    #     _df = _df[_df['instrument_family'].isin(self.instrument_families)]
    # _df['instrument'] = _df['instrument_source_str'].str.cat(_df['instrument_family_str'], sep=' ')
    # self.onehot = pd.get_dummies(_df['instrument']).to_numpy()
    # self.df = _df
    # print(f'Data: {df.shape}')