import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from synthesizer import audio
from synthesizer.models.tacotron import Tacotron
from synthesizer.synthesizer_dataset import SynthesizerDataset, SynthesizerValidationDataset, collate_synthesizer
from synthesizer.utils import ValueWindow, data_parallel_workaround
from synthesizer.utils.plot import plot_spectrogram
from synthesizer.utils.symbols import symbols
from synthesizer.utils.text import sequence_to_text
from synthesizer.hparams import hparams
from synthesizer.utils.text import text_to_sequence
from vocoder.display import *
from datetime import datetime
import numpy as np
from pathlib import Path
import sys
import time
import dill
import random
import wandb
from tqdm import tqdm
from torchinfo import summary
##



def np_now(x: torch.Tensor): return x.detach().cpu().numpy()

def time_string():
    return datetime.now().strftime("%Y-%m-%d %H:%M")

def synthesize_spectrograms(model, texts, embeds, device, return_alignments=False):
        """
        Synthesizes mel spectrograms from texts and speaker embeddings.

        :param texts: a list of N text prompts to be synthesized
        :param embeddings: a numpy array or list of speaker embeddings of shape (N, 256) 
        :param return_alignments: if True, a matrix representing the alignments between the 
        characters
        and each decoder output step will be returned for each spectrogram
        :return: a list of N melspectrograms as numpy arrays of shape (80, Mi), where Mi is the 
        sequence length of spectrogram i, and possibly the alignments.
        """
        # Print some info about the model when it is loaded            
        tts_k = model.get_step() // 1000

        # simple_table([("Tacotron", str(tts_k) + "k"),
        #             ("r", model.r)])

        # Preprocess text inputs
        inputs = [text_to_sequence(text.strip(), hparams.tts_cleaner_names) for text in texts]
        if not isinstance(embeds, list):
            embeds = [embeds]

        # Batch inputs
        batched_inputs = [inputs[i:i+hparams.synthesis_batch_size]
                             for i in range(0, len(inputs), hparams.synthesis_batch_size)]
        batched_embeds = [embeds[i:i+hparams.synthesis_batch_size]
                             for i in range(0, len(embeds), hparams.synthesis_batch_size)]

        def pad1d(x, max_len, pad_value=0):
            return np.pad(x, (0, max_len - len(x)), mode="constant", constant_values=pad_value)
        
        specs = []
        for i, batch in enumerate(batched_inputs, 1):
            # print(f"\n| Generating {i}/{len(batched_inputs)}")

            # Pad texts so they are all the same length
            text_lens = [len(text) for text in batch]
            max_text_len = max(text_lens)
            chars = [pad1d(text, max_text_len) for text in batch]
            chars = np.stack(chars)

            # Stack speaker embeddings into 2D array for batch processing
            speaker_embeds = np.stack(batched_embeds[i-1])

            # Convert to tensor
            chars = torch.tensor(chars).long().to(device)
            speaker_embeddings = torch.tensor(speaker_embeds).float().to(device)
            # Inference
            _, mels, alignments = model.generate(chars, speaker_embeddings)
            mels = mels.detach().cpu().numpy()
            for m in mels:
                # Trim silence from end of each spectrogram
                while np.max(m[:, -1]) < hparams.tts_stop_threshold:
                    temp = np.max(m[:, -1])
                    m = m[:, :-1]
                specs.append(m)
        # print("\n\nDone.\n")
        return (specs, alignments) if return_alignments else specs


def validate(model, valid_metadata, valid_dataloader, device, mel_images_dir, num_save_mel_images):
    running_loss = 0.0
    model.eval()
    pbar = tqdm(enumerate(valid_dataloader, 1), desc="validation phase")
    with torch.no_grad():
        for i, (texts, mels, embeds, idx) in pbar:
            start_time = time.time()

            # Generate stop tokens for training
            stop = torch.ones(mels.shape[0], mels.shape[2])
            for j, k in enumerate(idx):
                stop[j, :int(valid_metadata[k][4])-1] = 0

            texts = texts.to(device)
            mels = mels.to(device)
            embeds = embeds.to(device)
            stop = stop.to(device)

            # Forward pass
            # Parallelize model onto GPUS using workaround due to python bug
            if device.type == "cuda" and torch.cuda.device_count() > 1:
                m1_hat, m2_hat, attention, stop_pred = data_parallel_workaround(model, texts,
                                                                                mels, embeds)
            else:
                m1_hat, m2_hat, attention, stop_pred = model(texts, mels, embeds, mode='valid')

            # Backward pass
            m1_loss = F.mse_loss(m1_hat, mels) + F.l1_loss(m1_hat, mels)
            m2_loss = F.mse_loss(m2_hat, mels)
            stop_loss = F.binary_cross_entropy(stop_pred, stop)

            loss = m1_loss + m2_loss + stop_loss
            running_loss += loss.item()

            # Update tqdm with the current valid loss
            pbar.set_postfix(valid_loss=running_loss / (i + 1))
    valid_loss = running_loss / len(valid_metadata)
    
    # 합성한 멜 스펙트로그램 저장
    random.seed(42)
    step = model.get_step()

    dataloader_iter = iter(valid_dataloader)
    batch = next(dataloader_iter)

    _, mels, embeds, indices = batch
    texts = "저희는 팀쿡이고 보이스 클로닝 인공지능을 활용한 커스텀 오디오북 제작 프로젝트를 합니다.".split("\n")
    mels = mels.cpu().detach().numpy()
    embeds = embeds.cpu().detach().numpy()  # (b_size, embed_dim)
    indices = np.array(indices)
    # num_save_mel_images = 1
    # assert num_save_mel_images==1, "For now, only one mel spec generation is supported."

    random_indices = np.random.choice(embeds.shape[0], num_save_mel_images, replace=False)
    embeds = embeds[random_indices]   # (num_save_mel_images, embed_dim)
    embeds = np.repeat(embeds, len(texts), axis=1)
    print(f'embeds.shape: {embeds.shape}')
    selected_mels = mels[random_indices]
    selected_indices = indices[random_indices]

    for i, embed in enumerate(embeds):
        specs = synthesize_spectrograms(model, texts, embed, device)   # (80, mel_frames)를 원소로 num_save_mel_images개 갖는 리스트
    
        spec = np.concatenate(specs, axis=1)

        plt.figure(figsize=(10, 4))
        plt.imshow(spec.T, aspect='auto', origin='lower', cmap='viridis')  # 또는 다른 colormap 사용
        plt.title(f'Mel Spectrogram Step {step} Sample {i}')
        plt.xlabel('Time')
        plt.ylabel('Frequency Bin')
        plt.colorbar(format='%+2.0f dB')
        
        # 이미지 파일로 저장
        mel_output_fpath = mel_images_dir.joinpath(f"step_{step}")
        mel_output_fpath.mkdir(exist_ok=True)
        mel_output_fpath = mel_output_fpath.joinpath(f"sample_{selected_indices[i]}.png")
        plt.savefig(mel_output_fpath, bbox_inches='tight')
        plt.close()

        # ground truth mel 저장 (참고용)
        plt.figure(figsize=(10, 4))
        plt.imshow(selected_mels[i], aspect='auto', origin='lower', cmap='viridis')
        plt.title(f'Mel Spectrogram Step {step} Sample {i}')
        plt.xlabel('Time')
        plt.ylabel('Frequency Bin')
        plt.colorbar(format='%+2.0f dB')
        mel_output_fpath = mel_output_fpath.parent.joinpath(f"sample_gt_{selected_indices[i]}.png")
        plt.savefig(mel_output_fpath, bbox_inches='tight')
        plt.close()

    return valid_loss


def train(run_id: str, syn_dir: str, models_dir: str, save_every: int,
         backup_every: int, force_restart:bool, hparams):

    syn_dir = Path(syn_dir)
    models_dir = Path(models_dir)
    models_dir.mkdir(exist_ok=True)

    model_dir = models_dir.joinpath(run_id)
    plot_dir = model_dir.joinpath("plots")
    wav_dir = model_dir.joinpath("wavs")
    mel_output_dir = model_dir.joinpath("mel-spectrograms")
    mel_images_dir = model_dir.joinpath("mel-images")
    meta_folder = model_dir.joinpath("metas")
    model_dir.mkdir(exist_ok=True)
    plot_dir.mkdir(exist_ok=True)
    wav_dir.mkdir(exist_ok=True)
    mel_output_dir.mkdir(exist_ok=True)
    mel_images_dir.mkdir(exist_ok=True)
    meta_folder.mkdir(exist_ok=True)
    
    weights_fpath = model_dir.joinpath(run_id).with_suffix(".pt")
    metadata_fpath = syn_dir.joinpath("train.txt")
    
    print("Checkpoint path: {}".format(weights_fpath))
    print("Loading training data from: {}".format(metadata_fpath))
    print("Using model: Tacotron")
    
    # Book keeping
    step = 0
    time_window = ValueWindow(100)
    loss_window = ValueWindow(100)
    
    
    # From WaveRNN/train_tacotron.py
    if torch.cuda.is_available():
        device = torch.device("cuda")

        for session in hparams.tts_schedule:
            _, _, _, batch_size = session
            if batch_size % torch.cuda.device_count() != 0:
                raise ValueError("`batch_size` must be evenly divisible by n_gpus!")
    else:
        device = torch.device("cpu")
    print("Using device:", device)

    # Instantiate Tacotron Model
    print("\nInitialising Tacotron Model...\n")
    model = Tacotron(embed_dims=hparams.tts_embed_dims,
                     num_chars=len(symbols),
                     encoder_dims=hparams.tts_encoder_dims,
                     decoder_dims=hparams.tts_decoder_dims,
                     n_mels=hparams.num_mels,
                     fft_bins=hparams.num_mels,
                     postnet_dims=hparams.tts_postnet_dims,
                     encoder_K=hparams.tts_encoder_K,
                     lstm_dims=hparams.tts_lstm_dims,
                     postnet_K=hparams.tts_postnet_K,
                     num_highways=hparams.tts_num_highways,
                     dropout=hparams.tts_dropout,
                     stop_threshold=hparams.tts_stop_threshold,
                     speaker_embedding_size=hparams.speaker_embedding_size).to(device)
    max_text_len = len("저희는 팀쿡이고 보이스 클로닝 인공지능을 활용한 커스텀 오디오북 제작 프로젝트를 합니다.")

    # Initialize the optimizer
    optimizer = optim.Adam(model.parameters())

    # Load the weights
    if force_restart or not weights_fpath.exists():
        print("\nStarting the training of Tacotron from scratch\n")
        model.save(weights_fpath)

        # Embeddings metadata
        char_embedding_fpath = meta_folder.joinpath("CharacterEmbeddings.tsv")
        with open(char_embedding_fpath, "w", encoding="utf-8") as f:
            for symbol in symbols:
                if symbol == " ":
                    symbol = "\\s"  # For visual purposes, swap space with \s

                f.write("{}\n".format(symbol))
    else:
        print("\nLoading weights at %s" % weights_fpath)
        model.load(weights_fpath, optimizer)
        print("Tacotron weights loaded from step %d" % model.step)
    
    # Initialize the dataset
    metadata_fpath = syn_dir.joinpath("train.txt")
    mel_dir = syn_dir.joinpath("mels")
    embed_dir = syn_dir.joinpath("embeds")
    dataset = SynthesizerDataset(metadata_fpath, mel_dir, embed_dir, hparams)
    valid_dataset = SynthesizerValidationDataset(metadata_fpath, mel_dir, embed_dir, hparams)
    valid_dataloader = DataLoader(valid_dataset, collate_fn=lambda batch: collate_synthesizer(batch, r, hparams),
                             batch_size=batch_size,
                             num_workers=0,
                             shuffle=False,
                             pin_memory=True)

    best_valid_loss = float("inf")  # 최고 검증 손실 초기화
    num_no_improvement = 0  # 개선되지 않은 횟수 초기화

    for i, session in enumerate(hparams.tts_schedule):
        current_step = model.get_step()

        r, lr, max_step, batch_size = session

        training_steps = max_step - current_step

        # Do we need to change to the next session?
        if current_step >= max_step:
            # Are there no further sessions than the current one?
            if i == len(hparams.tts_schedule) - 1:
                # We have completed training. Save the model and exit
                model.save(weights_fpath, optimizer)
                break
            else:
                # There is a following session, go to it
                continue

        model.r = r

        # Begin the training
        simple_table([(f"Steps with r={r}", str(training_steps // 1000) + "k Steps"),
                      ("Batch Size", batch_size),
                      ("Learning Rate", lr),
                      ("Outputs/Step (r)", model.r)])

        for p in optimizer.param_groups:
            p["lr"] = lr

        data_loader = DataLoader(dataset, collate_fn=lambda batch: collate_synthesizer(batch, r, hparams),
                                 batch_size=batch_size,
                                 num_workers=0, 
                                #  num_wokrers=2, ## cannot use multiprocessing in Windows
                                 shuffle=True,
                                 pin_memory=True)

        total_iters = len(dataset) 
        steps_per_epoch = np.ceil(total_iters / batch_size).astype(np.int32)
        epochs = np.ceil(training_steps / steps_per_epoch).astype(np.int32)

        for epoch in range(1, epochs+1):
            for i, (texts, mels, embeds, idx) in enumerate(data_loader, 1):
                model.train()
                start_time = time.time()

                # Generate stop tokens for training
                stop = torch.ones(mels.shape[0], mels.shape[2])
                for j, k in enumerate(idx):
                    stop[j, :int(dataset.metadata[k][4])-1] = 0

                texts = texts.to(device)
                mels = mels.to(device)
                embeds = embeds.to(device)
                stop = stop.to(device)

                # Forward pass
                # Parallelize model onto GPUS using workaround due to python bug
                if device.type == "cuda" and torch.cuda.device_count() > 1:
                    m1_hat, m2_hat, attention, stop_pred = data_parallel_workaround(model, texts,
                                                                                    mels, embeds)
                else:
                    m1_hat, m2_hat, attention, stop_pred = model(texts, mels, embeds)

                # Backward pass
                m1_loss = F.mse_loss(m1_hat, mels) + F.l1_loss(m1_hat, mels)
                m2_loss = F.mse_loss(m2_hat, mels)
                stop_loss = F.binary_cross_entropy(stop_pred, stop)

                loss = m1_loss + m2_loss + stop_loss
                
                # wandb.log({"training_loss": loss.item()})

                optimizer.zero_grad()
                loss.backward()

                if hparams.tts_clip_grad_norm is not None:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), hparams.tts_clip_grad_norm)
                    if np.isnan(grad_norm.cpu()):
                        print("grad_norm was NaN!")

                optimizer.step()

                time_window.append(time.time() - start_time)
                loss_window.append(loss.item())

                step = model.get_step()
                k = step // 1000

                msg = f"| Epoch: {epoch}/{epochs} ({i}/{steps_per_epoch}) | Loss: {loss_window.average:#.4} | {1./time_window.average:#.2} steps/s | Step: {k}k | "
                stream(msg)

                # Backup or save model as appropriate
                if backup_every != 0 and step % backup_every == 0 : 
                    backup_fpath = Path("{}/{}_{}k.pt".format(str(weights_fpath.parent), run_id, k))
                    model.save(backup_fpath, optimizer)
                    
                    valid_loss = validate(model, valid_dataset.metadata, valid_dataloader, device, mel_images_dir, num_save_mel_images=5)

                    print(f'valid loss : {valid_loss:.3f}')
                    # wandb.log({"validation_loss": valid_loss, "step": step})

                    # 검증 손실이 개선되지 않는 경우 카운트
                    if valid_loss < best_valid_loss:
                        best_valid_loss = valid_loss
                        num_no_improvement = 0  # 개선됨, 카운트 리셋
                    else:
                        num_no_improvement += 1  # 개선되지 않음, 카운트 증가
                        print(f"validation loss is not improved. num_no_improvement: {num_no_improvement}")

                    # 7번 개선되지 않으면 훈련 종료
                    if num_no_improvement >= 7:
                        print("Validation loss has not improved for 7 consecutive checks. Stopping training.")
                        # wandb.log({"training_status": "stopped"})
                        return  # 훈련 종료

                if save_every != 0 and step % save_every == 0 : 
                    # Must save latest optimizer state to ensure that resuming training
                    # doesn't produce artifacts
                    model.save(weights_fpath, optimizer)

                # Evaluate model to generate samples
                epoch_eval = hparams.tts_eval_interval == -1 and i == steps_per_epoch  # If epoch is done
                step_eval = hparams.tts_eval_interval > 0 and step % hparams.tts_eval_interval == 0  # Every N steps
                if epoch_eval or step_eval:
                    for sample_idx in range(hparams.tts_eval_num_samples):
                        # At most, generate samples equal to number in the batch
                        if sample_idx + 1 <= len(texts):
                            # Remove padding from mels using frame length in metadata
                            mel_length = int(dataset.metadata[idx[sample_idx]][4])
                            mel_prediction = np_now(m2_hat[sample_idx]).T[:mel_length]
                            target_spectrogram = np_now(mels[sample_idx]).T[:mel_length]
                            attention_len = mel_length // model.r

                            eval_model(attention=np_now(attention[sample_idx][:, :attention_len]),
                                       mel_prediction=mel_prediction,
                                       target_spectrogram=target_spectrogram,
                                       input_seq=np_now(texts[sample_idx]),
                                       step=step,
                                       plot_dir=plot_dir,
                                       mel_output_dir=mel_output_dir,
                                       wav_dir=wav_dir,
                                       sample_num=sample_idx + 1,
                                       loss=loss,
                                       hparams=hparams)

                # Break out of loop to update training schedule
                if step >= max_step:
                    break

            # Add line break after every epoch
            print("")

def eval_model(attention, mel_prediction, target_spectrogram, input_seq, step,
               plot_dir, mel_output_dir, wav_dir, sample_num, loss, hparams):
    # Save some results for evaluation
    attention_path = str(plot_dir.joinpath("attention_step_{}_sample_{}".format(step, sample_num)))
    save_attention(attention, attention_path)

    # save predicted mel spectrogram to disk (debug)
    mel_output_fpath = mel_output_dir.joinpath("mel-prediction-step-{}_sample_{}.npy".format(step, sample_num))
    np.save(str(mel_output_fpath), mel_prediction, allow_pickle=False)

    # save griffin lim inverted wav for debug (mel -> wav)
    wav = audio.inv_mel_spectrogram(mel_prediction.T, hparams)
    wav_fpath = wav_dir.joinpath("step-{}-wave-from-mel_sample_{}.wav".format(step, sample_num))
    audio.save_wav(wav, str(wav_fpath), sr=hparams.sample_rate)

    # save real and predicted mel-spectrogram plot to disk (control purposes)
    spec_fpath = plot_dir.joinpath("step-{}-mel-spectrogram_sample_{}.png".format(step, sample_num))
    title_str = "{}, {}, step={}, loss={:.5f}".format("Tacotron", time_string(), step, loss)
    plot_spectrogram(mel_prediction, str(spec_fpath), title=title_str,
                     target_spectrogram=target_spectrogram,
                     max_len=target_spectrogram.size // hparams.num_mels)
    print("Input at step {}: {}".format(step, sequence_to_text(input_seq,hparams,True,True)))