from typing import Tuple
from interfaces import IPadder
from torch import Tensor
import torch


class TextPadder(IPadder):
    def __init__(
            self,
            pad_id: int
            ) -> None:
        super().__init__()
        self.pad_id = pad_id

    def pad(self, x: Tensor, max_len: int) -> Tensor:
        length = x.shape[0] # 입력 텐서 길이

        # print(max_len) #159
        # print(length) #159
        
        # 패딩을 위한 텐서 생성, 부족한 길이(max_len - length)만큼 pad_id로 채움
        pad = torch.ones(max_len - length, dtype=torch.int) * self.pad_id
        return torch.cat([x, pad], dim=0)


class AudPadder(IPadder):
    def __init__(
            self,
            pad_val: int,
            ) -> None:
        super().__init__()
        self.pad_val = pad_val


    # def pad(self, x: Tensor, max_len: int) -> Tensor:
    #     length = x.shape[0] # torch.Size([1, 80, 699])
    #     dim = x.shape[1:] # # [80, 699]
    #     # length, dim = x.shape
        
    #     # print(type(max_len))
    #     # print(type(length))
    #     # print(type(dim))
    #     # print(max_len)
    #     print(length)
    #     print(dim)

    #     # max_len = int(max_len)
    #     # length = int(length)

    #     # pad = torch.ones(max_len - length, dim, dtype=torch.int) * self.pad_val # max_len, length
    #     pad_shape = [max_len - length] + list(dim)  # 패딩을 위한 모양을 생성
    #     pad = torch.ones(pad_shape, dtype=torch.int) * self.pad_val # max_len, length
        
    
    def pad(self, x: Tensor, max_len: int) -> Tensor:
        length = x.shape[0] # 데이터의 길이
        dim = x.shape[1] # 데이터의 차원
 
        # 패딩된 모양(pad_shape) 계산 >> (max_len - length)은 패딩할 길이
        pad_shape = [max_len - length] + list(dim)

        # 패딩을 생성하고 pad_val 값으로 채움
        pad = torch.ones(pad_shape, dtype=torch.int)*self.pad_val # 텐서 생성
        
        return torch.cat([x,pad], dim=0)

        

def get_padders(
        aud_pad_val: int, text_pad_val: int
        ) -> Tuple[IPadder, IPadder]:
    return (
        TextPadder(text_pad_val),
        AudPadder(aud_pad_val)
        )
