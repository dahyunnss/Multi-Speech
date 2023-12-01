import torch
from typing import Union
from torch.utils.data import Dataset, DataLoader
from interfaces import IDataLoader, IPipeline, IPadder


SPK_ID = 0
PATH_ID = 1
TEXT_ID = 2
DURATION_ID = 3


class Data(Dataset):
    def __init__(
            self,
            data_loader: IDataLoader,
            aud_pipeline: IPipeline,
            text_pipeline: IPipeline,
            aud_padder: IPadder,
            text_padder: IPadder,
            batch_size: int,
            sep: str
            ) -> None:
        super().__init__()
        self.sep = sep
        self.aud_pipeline = aud_pipeline
        self.text_pipeline = text_pipeline
        self.aud_padder = aud_padder
        self.text_padder = text_padder
        self.batch_size = batch_size
        self.data = self.process(data_loader)
        self.max_speech_lens = []
        self.max_text_lens = []
        self.n_batches = len(self.data) // self.batch_size
        if len(self.data) % batch_size > 0:
            self.n_batches += 1
        self.__set_max_text_lens()
        self.__set_max_speech_lens()

    def process(self, data_loader: IDataLoader):
        data = data_loader.load().split('\n')
        p_data = []

        for item in data:
            elements = item.split(self.sep)
            if len(elements) > DURATION_ID:
                p_data.append(elements)
        p_data = sorted(p_data, key=lambda x: x[DURATION_ID], reverse=True)
        
        return p_data
    
    def __set_max_speech_lens(self):
        for i, item in enumerate(self.data):
            idx = i // self.batch_size
            length = float(item[DURATION_ID])
          
            if idx >= len(self.max_speech_lens):
                self.max_speech_lens.append(length)
            else:
                self.max_speech_lens[idx] = max(length, self.max_speech_lens[idx])

    def __set_max_text_lens(self):
        for i, item in enumerate(self.data):
            idx = i // self.batch_size
            length = len(item[TEXT_ID])
          
            if idx >= len(self.max_text_lens):
                self.max_text_lens.append(length)
            else:
                self.max_text_lens[idx] = max(length, self.max_text_lens[idx])
            

    def __len__(self) -> int:
        return len(self.data)

    def _get_max_len(self, idx: int) -> Union[None, int]:
         # 현재 데이터 인덱스에 해당하는 데이터가 속하는 버킷(bucket)을 결정 >> idx를 사용하여 해당 데이터가 속하는 버킷(데이터 범주)결정

        bucket_id = idx // self.batch_size #버킷 구간 #bucket_id : 0 >> 0번째 batch
       
        # 최대 음성 길이 정보가 없다면 None을 반환
        # 최대 텍스트 길이 정보를 반환 (self.max_text_lens[bucket_id] + 1)
        if bucket_id >= len(self.max_speech_lens): # bucket_id가 max_speech_len 리스트 길이를 넘어간 경우
            return None, self.max_text_lens[bucket_id] + 1
        
        return (  # 그렇지 않다면, 현재 버킷의 최대 음성 길이와 최대 텍스트 길이 정보를 반환
            self.max_speech_lens[bucket_id],
            self.max_text_lens[bucket_id] + 1
            )
        

    def __getitem__(self, idx: int): # 텐서 생성 부분
        [spk_id, file_path, text, _] = self.data[idx] # 데이터에서 발화자 ID, 음성파일경로, 텍스트 추출
        spk_id = int(spk_id)  

        max_speech_len, max_text_len = self._get_max_len(idx)  # 최대 음성 길이와 최대 텍스트 길이 >> 모든 데이터 포인트에서 최대 길이를 계산하는데 사용
        print(max_speech_len) #98.0
        print(max_text_len) #159

        #speech >> 2차원

        text = self.text_pipeline.run(text) # 텍스트 처리(데이터 전처리)
        text = self.text_padder.pad(text, max_text_len)  # 패딩을 추가 >> 모든 텍스트를 동일한 길이로 맞춤(159)

        speech = self.aud_pipeline.run(file_path) # aud_pipeline(mel) >> audio를 tensor로 변환
        # print('s1',speech[0].shape) # s1 torch.size([80, 712])
        # print('s2',speech[1].shape) # s2 torch.size([80, 699])
        # print('s23',speech[1].unsqueeze(0).shape) # s2 torch.size([80, 699])       
        
        speech_length = speech.shape[0] # 음성의 길이
        # print(speech_length) #[2,80,712]

        mask = [True] * speech_length 
        

        # 차원 맞추기
        if max_speech_len is not None:
            mask.extend([False] * int((max_speech_len - speech_length)))
            speech = self.aud_padder.pad(speech, int(max_speech_len))  # 음성을 패딩하여 길이를 맞춤(aud_padder)
            # print(max_speech_len[0])

        else:
            self.max_speech_lens.append(speech_length) 
        mask = torch.BoolTensor(mask) 
        spk_id = torch.LongTensor([spk_id]) 
    
        
        return speech, max_speech_len, mask, text, spk_id 

def get_batch_loader(
        data_loader: IDataLoader,
        aud_pipeline: IPipeline,
        text_pipeline: IPipeline,
        aud_padder: IPadder,
        text_padder: IPadder,
        batch_size: int,
        sep: str
        ):
    return DataLoader(
        Data(
            data_loader=data_loader,
            aud_pipeline=aud_pipeline,
            text_pipeline=text_pipeline,
            aud_padder=aud_padder,
            text_padder=text_padder,
            batch_size=batch_size,
            sep=sep
        ),
        batch_size=batch_size,
        shuffle=False
    )
