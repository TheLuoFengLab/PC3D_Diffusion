import os
import pickle
import torch
import numpy as np

class Dataset(torch.utils.data.Dataset):

    def __init__(self, files_or_paths):
        super().__init__()

        print("Scanning files...")
        data_files = []
        for path in files_or_paths:
            if os.path.isdir(path):
                for p, _, files in os.walk(path):
                    if not files: continue
                    for filename in files:
                        if filename.endswith(".pkl"):
                            data_files.append(os.path.join(p, filename))
            else:
                data_files.append(path)
        data, target = [], []
        for file in data_files:
            with open(file, "rb") as f:
                samples, coeff = pickle.load(f)
            data.extend(samples)
            target.append(coeff)
        self.data = data
        self.target = np.concatenate(target)

        print("Loaded data items {}".format(len(self.data)))


    def __getitem__(self, i):
        data = self.data[i]
        return data, self.target[i]
    
    def __len__(self):
        return len(self.data)
    
    @staticmethod
    def collate_fn(batch):
        seq_len = [len(item) for item, _ in batch]
        max_len = max(seq_len)
        data = []
        target = []
        for x, y in batch:
            data.append(np.pad(x, ((0, (max_len-len(x))), (0, 0))))
            target.append(y)
        return torch.as_tensor(np.stack(data, 0), dtype=torch.float32), \
            torch.as_tensor(seq_len, dtype=torch.long), \
            torch.as_tensor(np.stack(target, 0), dtype=torch.float32)


if __name__ == "__main__":
    dataset = Dataset(["data/fiber_sd_230l10d_coeff_025.pkl"])
