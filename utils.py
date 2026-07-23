import numpy as np
import librosa as lb
import torch 

def slice_array(audio_array: np.ndarray, chunk_size: int) -> list:
  chunks = []
  for x in range(0, len(audio_array), chunk_size):
    chunk = audio_array[x: x + chunk_size]
    if len(chunk) < chunk_size:
      padding = np.zeros(chunk_size - len(chunk))
      chunk = np.concatenate([chunk, padding])
    chunks.append(chunk)
  return chunks  


def denoise_chunk(noisy_chunk, model, device):
    stft_chunk = lb.stft(noisy_chunk)
    stft_chunk_amp = np.abs(stft_chunk)
    stft_chunk_ph = np.angle(stft_chunk)
    tensor_chunk = torch.from_numpy(stft_chunk_amp).float()
    tensor_chunk = tensor_chunk.unsqueeze(0).unsqueeze(0)
    tensor_chunk = tensor_chunk.to(device)
    prediction = model(tensor_chunk)
    return prediction, stft_chunk_ph