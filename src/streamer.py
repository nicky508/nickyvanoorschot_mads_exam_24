from pathlib import Path
from typing import Iterator

import torch
from mads_datasets.base import BaseDatastreamer
from pydantic import BaseModel

class VAEstreamer(BaseDatastreamer):
    def stream(self) -> Iterator:
        while True:
            if self.index > (self.size - self.batchsize):
                self.reset_index()
            batch = self.batchloop()
            # we throw away the Y
            X_, _ = zip(*batch)  # noqa N806
            X = torch.stack(X_)  # noqa N806
            
            # batch_size, sequence_length, feature_dim = X.size()
            # X = X.view(X.size(0), -1) 
            # print(X.shape)
            # and yield X, X
            yield X, X