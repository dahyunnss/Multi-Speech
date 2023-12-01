from typing import List, Tuple
import torch
import torch.nn as nn
from torch import Tensor
from layers import (
    Encoder,
    Decoder,
    SpeakerModule,
    PredModule,
    DecoderPrenet,
    PositionalEmbedding
    )
from utils import (
    cat_speaker_emb,
    get_positionals
    )


class Model(nn.Module):
    """Implements the MultiSpeech Model archticture

    Args:
        pos_emb_params (dic): The encoder's positional embedding parameters.
        encoder_params (dic): The model's encoder parameters.
        decoder_params (dic): The model's decoder parameters.
        speaker_mod_params (dic): The speaker module parameters.
        prenet_params (dic): The decoder's prenet/bottleneck parameters.
        pred_params (dict): The prediction module parameters.
        n_layers (int): The number of stacked encoders and decoders.
        device (str): The device to map the Tensors to.
    """
    def __init__(
            self,
            pos_emb_params: dict,
            encoder_params: dict,
            decoder_params: dict,
            speaker_mod_params: dict,
            prenet_params: dict,
            pred_params: dict,
            n_layers: int,
            device: str
            ) -> None:
        super().__init__()
        self.device = device
        self.n_layers = n_layers

        self.enc_layers = nn.ModuleList([
            Encoder(**encoder_params).to(device)
            for _ in range(n_layers)
            ])
        self.dec_layers = nn.ModuleList([
            Decoder(**decoder_params).to(device)
            for _ in range(n_layers)
        ])

        # Decoder의 Prenet 레이어
        self.dec_prenet = DecoderPrenet(
            **prenet_params
        ).to(device)

        
        self.pos_emb = PositionalEmbedding( # 입력 데이터의 위치 정보를 학습 가능한 위치 임베딩과 결합
            **pos_emb_params
            ).to(device)
        self.speaker_mod = SpeakerModule( # 발화자 정보를 처리하는 Speaker Module
            **speaker_mod_params
        ).to(device)
        self.pred_mod = PredModule( # Mel 스펙트로그램 및 Stop 예측을 수행하는 Prediction Module
            **pred_params
        ).to(device)

    def forward(
            self,
            x: Tensor,
            speakers: Tensor,
            y: Tensor,
            ) -> Tuple[Tensor, Tensor, List[Tensor]]:
        """Passes the input to the model's layers.

        Args:
            x (Tensor): The encoded text input of shpae [B, M].
            speakers (Tensor): The corresoponding speaker embeddings of shape
            [B, 1].
            y (Tensor): The target tensor of shape [B, T, n_mdels].

        Returns:
            Tuple[Tensor, Tensor, List[Tensor]]: A tuple contains the mel,
            stop prediction results and the list of all alignments.
        """
        # TODO: Add Teacher forcing
        # TODO: Add Prediction function
        enc_inp = self.pos_emb(x) # 입력 데이터에 위치 임베딩(pos_emb) 추가
        speaker_emb = self.speaker_mod(speakers) # 발화자 정보를 처리하여 Speaker Embedding을 생성
        prenet_out = self.dec_prenet(y) # Decoder의 Prenet을 통해 입력 데이터를 전처리

        dec_input = cat_speaker_emb(speaker_emb, prenet_out) # 발화자 Embedding과 Prenet 출력을 결합하여 Decoder의 입력 데이터를 생성
        
        
        # Decoder 입력 데이터에 위치 정보 추가
        batch_size, max_len, d_model = dec_input.shape
        dec_input = dec_input + get_positionals(
            max_len, d_model
            ).to(self.device)
        
        # 학습 중에 생성되는 Mel 스펙트로그램, Stop 예측 및 Alignment 결과를 저장하기 위한 변수 초기화
        center = torch.ones(batch_size)
        mel_results = None
        stop_results = None
        alignments = [None for _ in range(self.n_layers)]

        # Decoder의 각 타임 스텝에 대해 반복
        for i in range(1, max_len):
            dec_input_sliced = dec_input[:, :i, :]
            iterator = enumerate(zip(self.enc_layers, self.dec_layers))

            # 각 레이어에서 Encoder 및 Decoder를 적용하여 각각의 결과 및 Alignment 저장
            for j, (enc_layer, dec_layer) in iterator:
                enc_inp = enc_layer(enc_inp)
                dec_input_sliced, att, temp_center = dec_layer(
                    x=dec_input_sliced,
                    encoder_values=cat_speaker_emb(speaker_emb, enc_inp),
                    center=center
                    )
                alignments[j] = att[:, -1:, :] if alignments[j] is None else \
                    torch.cat([alignments[j], att[:, -1:, :]], dim=1)
            center = temp_center

            # Mel 스펙트로그램 및 Stop 예측 결과 저장
            mels, stop_props = self.pred_mod(dec_input_sliced[:, -1:, :])
            mel_results = mels if mel_results is None else \
                torch.cat([mel_results, mels], dim=1)
            stop_results = stop_props if stop_results is None else \
                torch.cat([stop_results, stop_props], dim=1)
        return mel_results, stop_results, alignments
