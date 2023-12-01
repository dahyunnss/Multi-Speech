import torch
from typing import Union
from torch.utils.data import Dataset, DataLoader
from interfaces import IDataLoader, IPipeline, IPadder
import matplotlib.pyplot as plt


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
        self.max_speech_lens = [1011]
        self.max_text_lens = []
        self.n_batches = len(self.data) // self.batch_size
        if len(self.data) % batch_size > 0:
            self.n_batches += 1
        self.__set_max_text_lens()
        # self.__set_max_speech_lens() #new

    def process(self, data_loader: IDataLoader):
        data = data_loader.load().split('\n')
        p_data = [] 

        for item in data:
            elements = item.split(self.sep) #sep(|) 기준 split 

            if len(elements) > DURATION_ID: # 결측치 제거(데이터 필터링) >> 데이터 길이 3초과하면 append
                p_data.append(elements)
        # for i in p_data:
        #     print(i[3])
        
        p_data = sorted(p_data, key=lambda x: float(x[DURATION_ID]), reverse=True) # 여러개의 리스트들을 DURATION_ID기준으로 정렬
        # print('p==================', p_data[0][3]) #12.629

        return p_data
    
    # new
    # def __set_max_speech_lens(self): # max_speech_len setting
    #     for i, item in enumerate(self.data):
    #         idx = i // self.batch_size
    #         length = float(item[DURATION_ID]) # 현재 record의 음성 길이를 나타냄. DURATION_ID를 사용해 음성 길이 정보를 가져옴
          
    #         if idx >= len(self.max_speech_lens):
    #             self.max_speech_lens.append(length)
    #         else:
    #             self.max_speech_lens[idx] = max(length, self.max_speech_lens[idx])
            

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
        bucket_id = idx // self.batch_size #버킷 구간 #bucket_id : 0 >> 0번째 batch

        # 최대 음성 길이 정보가 없다면 None을 반환
        # 최대 텍스트 길이 정보를 반환 (self.max_text_lens[bucket_id] + 1)
        # print('========msl2=========',self.max_speech_lens)
        if bucket_id >= len(self.max_speech_lens): # bucket_id가 max_speech_len 리스트 길이를 넘어간 경우
            return None, self.max_text_lens[bucket_id] + 1
        # print('========msl2=========',self.max_speech_lens) #[1001]
    
        return (  # 그렇지 않다면, 현재 버킷의 최대 음성 길이와 최대 텍스트 길이 정보를 반환
            self.max_speech_lens[bucket_id],
            self.max_text_lens[bucket_id] + 1
        )
                

    def __getitem__(self, idx: int): # 텐서 생성 부분
        
        [spk_id, file_path, text, _] = self.data[idx] # 데이터에서 발화자 ID, 음성파일경로, 텍스트 추출
        spk_id = int(spk_id)  

        max_speech_len, max_text_len = self._get_max_len(idx)  # 최대 음성 길이와 최대 텍스트 길이 >> 모든 데이터 포인트에서 최대 길이를 계산하는데 사용

        text = self.text_pipeline.run(text) # 텍스트 처리(데이터 전처리)
        text = self.text_padder.pad(text, max_text_len)  # 패딩을 추가 >> 모든 텍스트를 동일한 길이로 맞춤(159)


        # 걍 멜스펙트로그램으로 바꾸는애 
        speech = self.aud_pipeline.run(file_path) # aud_pipeline(mel) >> audio를 tensor로 변환 
        
        speech_length = speech.shape[2] # 음성의 길이

        mask = [True] * speech_length 



        if max_speech_len is not None:
            mask.extend([False] * int((max_speech_len - speech_length)))
            speech = self.aud_padder.pad(speech, int(max_speech_len))  # 음성을 패딩하여 길이를 맞춤(aud_padder) >> [2,80,1006]
    
        else:
            speech = speech[0]
            self.max_speech_lens.append(speech_length) 

        mask = torch.BoolTensor(mask) 
        spk_id = torch.LongTensor([spk_id]) 
        # print('mmmm', speech.size())
        # print('spspspspsp',self.max_speech_lens) #[1011]

        # print(speech.size(), max_speech_len, mask.size(), text.size(), spk_id) #ppt

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
        shuffle=False,
        num_workers=3
    )
