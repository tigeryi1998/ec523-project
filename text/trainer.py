import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from omegaconf import DictConfig
from text.encoder import Encoder, AutoTokenizer
from text.dataset import WikiText
from data2vec import Data2Vec
from utils import AverageMeter, maybe_save_checkpoint

class TextTrainer:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.device = torch.device(self.cfg.device if torch.cuda.is_available() else 'cpu')
        self.init_components()

    def init_components(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model.encoder_checkpoint)
        self.encoder = Encoder(cfg=self.cfg).to(self.device)
        self.model = Data2Vec(encoder=self.encoder, cfg=self.cfg).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), self.cfg.optimizer.lr)
        self.criterion = nn.SmoothL1Loss(reduction='sum').to(self.device)
        self.setup_data_loaders()
        self.tensorboard = SummaryWriter(log_dir=self.cfg.train.log_dir)
        self.loss_tracker = AverageMeter('loss')

    def setup_data_loaders(self):
        self.train_dataset = WikiText(self.cfg, 'train', self.tokenizer)
        self.test_dataset = WikiText(self.cfg, 'test', self.tokenizer)
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.cfg.train.batch_size, collate_fn=self.train_dataset.collate_fn)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.cfg.train.eval_batch_size, collate_fn=self.test_dataset.collate_fn)

    def process_batch(self, batch, train=True):
        src, trg, mask = [x.to(self.device) for x in batch]
        x, y = self.model(src, trg, mask)
        loss = self.criterion(x, y) / x.size(0)
        if train:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return loss.item()

    def run_epoch(self, loader, train=True):
        self.model.train() if train else self.model.eval()
        self.loss_tracker.reset()
        method = self.process_batch if train else self.test_step
        desc = 'Training' if train else 'Evaluating'
        
        with tqdm(loader, unit="batch", desc=f'{desc}...', leave=False) as iterator:
            for batch in iterator:
                loss = method(batch, train=train)
                self.loss_tracker.update(loss)
                iterator.set_postfix(loss=self.loss_tracker.avg)

        return self.loss_tracker.avg

    def train(self):
        for epoch in range(1, self.cfg.train.num_epochs + 1):
            print(f"Epoch {epoch}/{self.cfg.train.num_epochs}")
            train_loss = self.run_epoch(self.train_loader, train=True)
            val_loss = self.run_epoch(self.test_loader, train=False)
            
            self.tensorboard.add_scalar('train_loss', train_loss, epoch)
            self.tensorboard.add_scalar('val_loss', val_loss, epoch)
            
            maybe_save_checkpoint(self.model, self.optimizer, self.ckpt_dir, epoch, self.save_ckpt_freq)
