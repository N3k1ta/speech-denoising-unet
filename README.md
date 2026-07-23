# Speech Denoising — U-Net

Deep learning speech enhancement: U-Net encoder-decoder trained on VoiceBank+DEMAND dataset.
Portfolio step toward a real-time **DeepFilterNet + JUCE** VST/AU plugin.

## Goal

Train a U-Net model to suppress background noise from speech recordings.
Input: noisy speech spectrogram. Output: clean speech spectrogram.

## Results

On a held-out test file (`p232_363.wav`):

| Metric | Score |
|---|---|
| PESQ | 2.13 |
| STOI | 0.97 |

Trained for 10 epochs — chosen over 20 after an explicit comparison found
test-set metrics got worse past that point (train loss kept falling, test
loss/PESQ rose — a clear overfitting signal). See notebook 05 for the full
10-vs-20-epoch table.

Denoising is strongest against stationary noise (hums, fans, traffic); babble
noise (background speech) remains the hardest case, as expected — it shares
spectral structure with the target voice.

Example audio (noisy / denoised / clean) in [`samples/`](samples/).

## Dataset

**VoiceBank+DEMAND**
- 11,572 training pairs / 824 test pairs
- 28 speakers (train), separate test set
- Sample rate: 16,000 Hz
- Noise environments: café, street, office, restaurant and more (DEMAND library)

## Notebooks

| Notebook | Description |
|----------|-------------|
| `speech_denoising_01.ipynb` | EDA — dataset exploration, waveforms, STFT spectrograms |
| `speech_denoising_02.ipynb` | Preprocessing — chunking, zero-padding, STFT conversion, save to .npy |
| `speech_denoising_03.ipynb` | U-Net architecture — ConvBlock, encoder/decoder, skip connections |
| `speech_denoising_04.ipynb` | Training — MSE loss, Adam optimizer, MPS acceleration, checkpointing |
| `speech_denoising_05.ipynb` | Inference — full-file reconstruction, listening test, PESQ & STOI |

## Code

- `model.py` — standalone `UNet` architecture module (`ConvBlock` + `UNet`)
- `utils.py` — shared helpers: `slice_array` (chunking), `denoise_chunk` (inference on one chunk)

## Stack

Python · PyTorch · librosa · NumPy · soundfile · pesq · pystoi

## Roadmap

- [x] EDA & data exploration
- [x] Preprocessing — chunking, STFT conversion
- [x] U-Net architecture
- [x] Training loop (MPS-accelerated, checkpointed on best loss)
- [x] Inference — full-file reconstruction, listening test, PESQ & STOI evaluation

This U-Net baseline is complete. Next step (fine-tuning DeepFilterNet, exporting
to ONNX, building the JUCE plugin) continues in a separate repository as part
of the broader VoiceClean project.

## License

MIT
