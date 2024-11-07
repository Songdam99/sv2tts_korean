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

# Usage
### embedding_extraction.py
`embedding_extraction.py` is a script used to extract embeddings from a reference voice file. The extracted embeddings represent the voice features, which can be used for synthesize_voice.py
#### Parameters:
- **--ref_voice_path**: **Path to the reference voice file.** This is the .wav of .m4a file from which the embeddings will be extracted. It should be the voice that you want to use for generating the embedding.
- **--hash**: **Unique identifier for the embedding file.** This is a custom string or ID that will be used to save and identify the embedding file. You can use any alphanumeric string (e.g., "abcd1234"), but it should be unique for each extraction to avoid overwriting files.
```plaintext
python embedding_extraction.py --ref_voice_path "path/to/your/ref-voice.wav" --hash 12345678
```
#### Outputs:
Once you run the command, the embeddings will be extracted from the reference voice file and saved to a file, typically with the name corresponding to the --hash value. (e.g., narrify_ai/embeddings/abcd1234.pkl)

### synthesize_voice.py
`synthesize_voice.py` is a script used to synthesize speech from input text using pre-trained models. This script generates an audio file by converting the given text into speech, using synthesizer and vocoder.
#### Parameters:
- **--text**: **Text to be synthesized into speech.** This argument takes the input text that you want to convert to speech. The text should be provided within quotation marks. The input text must be in Korean, as the current model is specifically trained for Korean text-to-speech conversion.
- **--hash_and_time**: **Unique identifier with timestamp.** This is a unique string used to identify the generated speech file. It helps distinguish between different synthesis sessions. The hash_and_time should follow a specific format like <hash>_<date-time>, where hash is a unique identifier (e.g., 12345678), and the date-time is formatted as DDMMYY-HHMMSS (e.g., 241107-142130).
```plaintext
python synthesize_voice.py --text "여기에 합성할 텍스트가 들어갑니다." --hash_and_time 12345678_241107-142130
```
#### Outputs:
When you run this command, the text will be converted to speech, and an audio file (.wav format) will be generated. The filename will include the unique hash_and_time identifier to make it easy to track and store multiple synthesized audio files.

For example, after running the above command, an audio file named 12345678_241107-142130.wav will be generated, which you can play or further process.
