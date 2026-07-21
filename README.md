# Speech Denoising — U-Net

Deep learning speech enhancement: U-Net encoder-decoder trained on VoiceBank+DEMAND dataset.  
Portfolio step toward a real-time **DeepFilterNet + JUCE** VST/AU plugin.

## Goal

Train a U-Net model to suppress background noise from speech recordings.  
Input: noisy speech spectrogram. Output: clean speech spectrogram.

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

## Model

`model.py` — standalone `UNet` architecture module (`ConvBlock` + `UNet`), imported by
the training and (future) inference notebooks. Encoder 1→32→64→128 channels,
bottleneck, decoder with skip connections back down to a single-channel output.

## Stack

Python · PyTorch · librosa · NumPy · soundfile · pesq · pystoi

## Roadmap

- [x] EDA & data exploration
- [x] Preprocessing — chunking, STFT conversion
- [x] U-Net architecture
- [x] Training loop (MPS-accelerated, checkpointed on best loss)
- [ ] Inference — full-file reconstruction, listening test, PESQ & STOI evaluation
- [ ] Export to ONNX → DeepFilterNet → JUCE VST/AU plugin

## License

MIT
