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
| `speech_denoising_02.ipynb` | Preprocessing — chunking, zero-padding, save to .npy |

## Stack

Python · PyTorch · librosa · NumPy · soundfile · pesq · pystoi

## Roadmap

- [x] EDA & data exploration
- [x] Preprocessing — chunking, normalisation
- [ ] U-Net architecture
- [ ] Training loop
- [ ] Evaluation — PESQ & STOI
- [ ] Export to ONNX → DeepFilterNet → JUCE VST/AU plugin

## License

MIT
