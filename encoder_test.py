import torch
import argparse
from pathlib import Path
from tqdm import tqdm
from encoder.params_model import *
from encoder.data_objects.speaker_verification_validation_dataset import SpeakerValidationDataset, SpeakerValidationDataLoader
from encoder.model import SpeakerEncoder
# from encoder.tail95_models import SpeakerEncoder

def sync(device: torch.device):
    # For correct profiling (cuda operations are async)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

def test(model, test_loader, device, loss_device):
    model.eval()  # Validation 모드로 전환
    total_loss = 0
    total_eer = 0
    num_batches = 0

    with torch.no_grad():
        print("Test Loader Info:")
        print(f"Number of batches: {len(test_loader)}")
        for speaker_batch in tqdm(test_loader):
            inputs = torch.from_numpy(speaker_batch.data).to(device)
            print(inputs.shape)
            sync(device)
            embeds = model(inputs)
            embeds_loss = embeds.view((speakers_per_batch, utterances_per_speaker, -1)).to(loss_device)
            loss, eer = model.loss(embeds_loss)
            
            total_loss += loss.item()
            total_eer += eer
            num_batches += 1

    avg_loss = total_loss / num_batches
    avg_eer = total_eer / num_batches
    return avg_loss, avg_eer

if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description="Trains the speaker encoder. You must have run encoder_preprocess.py first.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("test_data_root", type=Path, help= \
        "Path to the test directory of encoder_preprocess.py.")
    parser.add_argument("-m", "--models_dir", type=Path, default="encoder/saved_models/", help=\
        "Path to the output directory that will contain the saved model weights, as well as "
        "backups of those weights and plots generated during training.")
    parser.add_argument("--model_name", type=str, help=\
        "model file name. ex) tail95_encoder.pt => --model_name tail95_encoder")
    parser.add_argument("-s", "--sight_range", type=int, help=\
        "In test speaker folder, the number of folders that dataloader can see.")
    args = parser.parse_args()
    
    test_dataset = SpeakerValidationDataset(args.test_data_root, args.sight_range)
    test_loader = SpeakerValidationDataLoader(
        test_dataset,
        speakers_per_batch,
        utterances_per_speaker,
        num_workers=0,  # CPU 코어 수에 맞게 조정
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")    
    loss_device = torch.device("cuda") 
    print(f"Using loss_device: {loss_device}")    
    
    model = SpeakerEncoder(device, loss_device)
    
    args.model_name = args.model_name + ".pt"
    state_fpath = args.models_dir.joinpath(args.model_name)
    if state_fpath.exists():
        print("Found existing model \"%s\", loading it and resuming training." % args.model_name)
        checkpoint = torch.load(state_fpath)
        init_step = checkpoint["step"]
        print(f"init_step: {init_step}")
        model.load_state_dict(checkpoint["model_state"])
    else:
        print("No model \"%s\" found, starting training from scratch." % args.model_name)
        
    model.eval()
    avg_loss, avg_eer = test(model, test_loader, device, loss_device)
    print(f'avg_test_loss : {avg_loss:.4f}, avg_test_eer : {avg_eer:.4f}')