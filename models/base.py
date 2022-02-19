import torch
from .types_ import *
from torch import nn
from abc import abstractmethod
from pytorch_lightning.core.lightning import LightningModule


class BaseVAE(nn.Module):
    
    def __init__(self) -> None:
        super(BaseVAE, self).__init__()

    def encode(self, input: Tensor) -> List[Tensor]:
        raise NotImplementedError

    def decode(self, input: Tensor) -> Any:
        raise NotImplementedError

    def sample(self, batch_size:int, current_device: int, **kwargs) -> Tensor:
        raise NotImplementedError

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def forward(self, *inputs: Tensor) -> Tensor:
        pass

    @abstractmethod
    def loss_function(self, *inputs: Any, **kwargs) -> Tensor:
        pass
    
   def training_step(self, batch, batch_idx):
        results = self(batch['images'])
        train_loss = self.loss_function(*results, M_N = 0.005)

        self.log_dict({key: val.item() for key, val in train_loss.items()}, sync_dist=True)

        return train_loss['loss']

    def validation_step(self, batch, batch_idx):
        self._shared_eval(batch, batch_idx, "val")

    def _shared_eval(self, batch, batch_idx, prefix):
        with torch.no_grad():
            results = self(batch['A'])
            eval_loss = self.loss_function(*results, M_N = 0.005)
        
        self.log_dict({f"{prefix}_{key}": val.item() for key, val in eval_loss.items()}, sync_dist=True)
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=2e-5, betas=(0.5, 0.999))




