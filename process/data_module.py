import torch, os
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, RandomSampler
from keras_preprocessing.sequence import pad_sequences

NUM_WORKERS = 4

# define RNN dataset
class DataSet(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx): 
        if len(self.sequences[idx]) == 4:
            sequence, label, _, static = self.sequences[idx]
            return {
                "sequence": torch.Tensor(np.array(sequence)),
                "static":  torch.Tensor(np.array(static)),
                "label": torch.tensor(label).long(),
            }
        else:
            sequence, label, _, = self.sequences[idx]
            return {
                "sequence": torch.Tensor(np.array(sequence)),
                "label": torch.tensor(label).long(),
            }

class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_sequences=None,
        test_sequences=None,
        batch_size=256,
        pad_type="post",
        pad_seq=False,
        max_seq_len=None
    ):
        super().__init__()
        self.pad_type = pad_type
        self.pad_seq = pad_seq
        self.train_sequences = train_sequences
        self.test_sequences = test_sequences
        self.max_seq_len = self.__get_max_seq_len(max_seq_len)
        self.batch_size = int(batch_size)
        self.train_mask = None
        self.test_mask = None
        self.test_sampler = None

    def __get_max_seq_len(self, max_seq_len):
        if max_seq_len:
            return max_seq_len
        else:
            return max([len(x[0]) for x in self.train_sequences])

    def pad_sequences_lens(self, sequences, train_split = False):
        seqences_pad = [torch.tensor(np.array(x[0])) for x in sequences]
        mask = [torch.tensor([0] * len(x[0]) + [1] * (self.max_seq_len - len(x[0]))) for x in sequences]
        if train_split:
            self.train_mask = mask
        else:
            self.test_mask = mask
        return [
            (np.array(ts), sequences[i][1])
            for i, ts in enumerate(
                pad_sequences(
                    seqences_pad,
                    maxlen=self.max_seq_len,
                    dtype="float32",
                    padding=self.pad_type,
                    truncating=self.pad_type,
                    value=0.0,
                )
            )
        ]

    def setup(self, stage=None):
        if self.pad_seq:
            self.train_sequences = self.pad_sequences_lens(self.train_sequences, train_split = True)
            self.test_sequences = self.pad_sequences_lens(self.test_sequences)
        self.train_dataset = DataSet(self.train_sequences)
        self.test_dataset = DataSet(self.test_sequences)
        self.test_sampler = RandomSampler(self.test_dataset, replacement=True, num_samples=len(self.test_dataset))

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=NUM_WORKERS,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=NUM_WORKERS,
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=NUM_WORKERS,
        )

    def test_dataloader_sampler(self):
        return DataLoader(
            self.test_dataset, batch_size=len(self.test_dataset), shuffle=False, num_workers=NUM_WORKERS, sampler=self.test_sampler
        )
