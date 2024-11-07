# Models
## Download Model Checkpoints
The pre-trained model checkpoints are too large to include directly in the repository. Download the following files from Google Drive and place them in the models directory.

### Required Files
- **encoder.pt**
- **synthesizer.pt**
- **vocoder.pt**
1. Go to the [Google Drive folder](https://drive.google.com/drive/folders/1755vsx7Qq3oXLoGVIur__HLmH19TgP9A?usp=sharing) containing the model checkpoints.
2. Download encoder.pt, synthesizer.pt, and vocoder.pt.
3. Create a models directory in the root of this repository and place the downloaded files inside:

```plaintext
📦 narrify_ai
 ┣ 📂embeddings
 ┣ 📂encoder
 ┣ 📂models
 ┃ ┣ 📜encoder.pt
 ┃ ┣ 📜synthesizer.pt
 ┃ ┗ 📜vocoder.pt
 ┣ 📂synthesized_samples
 ┣ 📂synthesizer
 ┣ 📂utils
 ┣ 📂vocoder
 ┣ 📜embedding_extraction.py
 ┣ 📜README.md
 ┗ 📜synthesize_voice.py
```

## Model Descriptions
- **encoder.pt**: The encoder fine-tuned with AI HUB data from the CorentinJ repository's pre-trained model (nickname: transfer_learning_best.pt)
- **synthesizer.pt**: The synthesizer trained from scratch using the KSponSpeech dataset
- **vocoder.pt**: The vocoder fine-tuned from the pre-trained model of the CorentinJ repository using the KSponSpeech dataset
