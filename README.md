** In progress



# Real-Time Korean Voice Cloning
This repository is Korean version of sv2tts. The original model (which was developed by CorentinJ(https://github.com/CorentinJ/Real-Time-Voice-Cloning)) is based on English.
To implement Korean speech on the model, I refer to tail95(https://github.com/tail95/Voice-Cloning) where he made korean preprocessing codes. 
I changed some codes to convert tensorflow model to pytorch model and fixed some errors.

## References
- https://github.com/CorentinJ/Real-Time-Voice-Cloning
- https://github.com/tail95/Voice-Cloning
- https://medium.com/analytics-vidhya/the-intuition-behind-voice-cloning-with-5-seconds-of-audio-5989e9b2e042
- Transfer Learning from Speaker Verification to Multispeaker Text-To-Speech Synthesis (https://arxiv.org/abs/1806.04558)


## Used dataset
-KSponspeech (https://aihub.or.kr/aidata/105)

Make sure that your datasets has text-audio pairs.


## How to train your own model.

1. Install requirements
- pip install -r requirements.txt

2. run 
