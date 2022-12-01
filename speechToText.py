# 보이스피싱을 찾아내는 인공지능 연구

# 	1. Sound to text  : 음성을 문자로 변환하는 기술
# 	2. Voice phishing 관련 lexicon 만들어야 함
# 	3. Lexicon으로 필터를 만들어서 입력문장에서 비교처리하여 점수화한다. 점수가 일정 기준이상이면 voice phishing 가능성이 있음.


# error 해결방안 google colab 에서 처리하면 잘됨
	




import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch
import librosa
import numpy as np
import soundfile as sf
from scipy.io import wavfile
from IPython.display import Audio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer,Wav2Vec2Processor, Wav2Vec2CTCTokenizer


tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

#file_name = "HSBC_Scam_call.wav"

file_name = "my-audio.wav"


Audio(file_name)

data = wavfile.read(file_name)
framerate = data[0]
sounddata = data[1]
time = np.arange(0,len(sounddata))/framerate
input_audio, _ = librosa.load(file_name, sr=16000)
input_values = tokenizer(input_audio, return_tensors="pt").input_values
logits = model(input_values).logits
predicted_ids = torch.argmax(logits, dim=-1)
transcription = tokenizer.batch_decode(predicted_ids)[0]

transcription
