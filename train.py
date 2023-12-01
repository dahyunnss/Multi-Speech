import os
# CUDA 환경 설정
# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
#os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import gc
import torch.nn as nn
from data import get_batch_loader
from data_loaders import TextLoader
from loss import Loss
from model import Model
from optim import AdamWarmup
from padder import get_padders
from pipelines import get_pipelines
from tokenizer import CharTokenizer
from torch.utils.data import DataLoader
from interfaces import ITrainer
from torch.nn import Module
from pathlib import Path
from typing import Union
from torch import Tensor
from args import (
    get_args,
    get_model_args,
    get_loss_args,
    get_optim_args,
    get_aud_args,
    get_data_args,
    get_trainer_args
)

import torch


class Trainer(ITrainer):
    def __init__(
            self,
            train_loader: DataLoader,
            test_loader: DataLoader,
            model: Module,
            criterion: Module, #lossfunction
            optimizer: object,
            save_dir: Union[str, Path],
            steps_per_ckpt: int,
            epochs: int,
            last_step: int,
            device: str
            ) -> None:
        super().__init__()
        self.save_dir = save_dir
        self.epochs = epochs
        self.device = device
        self.optimizer = optimizer
        self.criterion = criterion
        self.model = model
        self.test_loader = test_loader
        self.train_loader = train_loader
        self.steps_per_ckpt = steps_per_ckpt
        self.last_step = last_step

    def set_train_mode(self):
        self.model = self.model.train()

    def set_test_mode(self):
        self.model = self.model.test()

    def _predict(self, text: Tensor, spk: Tensor, speech: Tensor):  #model_input
        text = text.to(self.device)
        spk = spk.to(self.device)
        speech = speech.to(self.device) # text, spk 입력
        return self.model(
            text, spk, speech #model 입력
        )

    def _train_step( #train 관련
            self,
            speech: Tensor,
            speech_length: Tensor,
            mask: Tensor,
            text: Tensor,
            spk: Tensor
            ):
        mel_results, stop_results, alignments = self._predict(
            text, spk, speech
        )
        self.optimizer.zero_grad()
        loss = self.criterion(
            lengths=speech_length,
            mask=mask,
            stop_pred=stop_results,
            mels_pred=mel_results,
            mels_target=speech,
            alignments=alignments
        )
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train(self):
        total_train_loss = 0
        for item in self.train_loader: #issue
            loss = self._train_step(*item)
            total_train_loss += loss
        return total_train_loss / len(self.train_loader)

    def test(self):
        total_test_loss = 0
        for (speech, speech_length, mask, text, spk) in self.test_loader:
            mel_results, stop_results, alignments = self._predict(
                text, spk, speech
            )
            total_test_loss += self.criterion(
                lengths=speech_length,
                mask=mask,
                stop_pred=stop_results,
                mels_pred=mel_results,
                mels_target=speech,
                alignments=alignments
                ).item()
        return total_test_loss / len(self.test_loader)



    def fit(self):
        # TODO: Add per step exporting here
        # TODO: Add tensor board here
        
        for epoch in range(self.epochs):
            train_loss = self.train() #issue
            test_loss = self.test()
          
            print(
                'epoch={}, training loss: {}, testing loss: {}'.format(
                    epoch, train_loss, test_loss)
                )

            


    def save_ckpt(self, idx: int): #checkpoint 저장
        path = os.path.join(self.save_dir, f'ckpt_{idx}')
        torch.save(self.model, path)
        print(f'checkpoint saved to {path}')


def get_model(args: dict, model_args: dict):
    return Model(
        **model_args
    )


def get_optim(args: dict, opt_args: dict, model: Module):
    return AdamWarmup(parameters=model.parameters(), **opt_args)


def get_criterion(args: dict, criterion_args: dict):
    return Loss(**criterion_args)


def get_tokenizer(args):
    # TODO: refactor this code
    tokenizer = CharTokenizer()
    tokenizer_path = args.tokenizer_path
    if args.tokenizer_path is not None:
        tokenizer.load_tokenizer(tokenizer_path)
        return tokenizer
    data = TextLoader(args.train_path).load().split('\n')
    # print(type(data[-1]))
    
    data = list(map(lambda x: x.split(args.sep)[2] if len(x.split(args.sep)) >= 3 else 'N/A', data)) #문장 자르기

    tokenizer.add_pad_token().add_eos_token()
    tokenizer.set_tokenizer(data)

    tokenizer_path = os.path.join(args.checkpoint_dir, 'tokenizer.json')
    tokenizer.save_tokenizer(tokenizer_path)
    print(f'tokenizer saved to {tokenizer_path}')
    return tokenizer

#issue
def get_trainer(args: dict): 
    # TODO: refactor this code
    tokenizer = get_tokenizer(args) 
    vocab_size = tokenizer.vocab_size
    data = TextLoader(args.train_path).load().split('\n')
    
    n_speakers = len(set(map(lambda x: x.split(args.sep)[0], data)))

    device = args.device
    model_args = get_model_args(
        args,
        vocab_size,
        tokenizer.special_tokens.pad_id,
        n_speakers
        )
    loss_args = get_loss_args(args)
    optim_args = get_optim_args(args)
    aud_args = get_aud_args(args)
    data_args = get_data_args(args)
    trainer_args = get_trainer_args(args)
    model = get_model(args, model_args)
    model = nn.DataParallel(model, device_ids = [0,1,2])
    model = model.to(device)
    model.cuda()
    optim = get_optim(args, optim_args, model)
    criterion = get_criterion(args, loss_args)
    text_padder, aud_padder = get_padders(0, tokenizer.special_tokens.pad_id)
    audio_pipeline, text_pipeline = get_pipelines(tokenizer, aud_args) # 전처리단
    
    train_loader = get_batch_loader(
        TextLoader(args.train_path),
        audio_pipeline,
        text_pipeline,
        aud_padder,
        text_padder,
        **data_args
    )
    test_loader = get_batch_loader(
        TextLoader(args.test_path),
        audio_pipeline,
        text_pipeline,
        aud_padder,
        text_padder,
        **data_args
    )
    return Trainer(
        train_loader=train_loader,
        test_loader=test_loader,
        model=model,
        criterion=criterion,
        optimizer=optim,
        last_step=0,
        **trainer_args
    )


if __name__ == '__main__':
    args = get_args()
    gc.collect()
    torch.cuda.empty_cache()
    get_trainer(args).fit()

