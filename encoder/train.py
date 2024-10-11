from encoder.visualizations import Visualizations
from encoder.data_objects import SpeakerVerificationDataLoader, SpeakerVerificationDataset
from encoder.data_objects.speaker_verification_validation_dataset import SpeakerValidationDataset, SpeakerValidationDataLoader
from encoder.params_model import *
from encoder.model import SpeakerEncoder
from utils.profiler import Profiler
from pathlib import Path
import torch
from tqdm import tqdm
import time  # time 모듈 추가
import os
import wandb


def sync(device: torch.device):
    # For correct profiling (cuda operations are async)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

def validate(model, validation_loader, device, loss_device):
    model.eval()  # Validation 모드로 전환
    total_loss = 0
    total_eer = 0
    num_batches = 0

    with torch.no_grad():
        print("Validation Loader Info:")
        print(f"Number of batches: {len(validation_loader)}")
        for speaker_batch in validation_loader:
            # print(speaker_batch.data.shape)
            inputs = torch.from_numpy(speaker_batch.data).to(device)
            if inputs.shape[0] != speakers_per_batch * utterances_per_speaker:
                break
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

def train(run_id: str, clean_data_root: Path, validation_data_root: Path, models_dir: Path, 
          umap_every: int, save_every: int, backup_every: int, vis_every: int, 
          force_restart: bool, visdom_server: str, no_visdom: bool):
    
    # Create datasets and dataloaders
    train_dataset = SpeakerVerificationDataset(clean_data_root)
    validation_dataset = SpeakerValidationDataset(validation_data_root)

    train_loader = SpeakerVerificationDataLoader(
        train_dataset,
        speakers_per_batch,
        utterances_per_speaker,
        num_workers=6,  # CPU 코어 수에 맞게 조정
    )

    validation_loader = SpeakerValidationDataLoader(
        validation_dataset,
        speakers_per_batch,
        utterances_per_speaker,
        num_workers=0,  # CPU 코어 수에 맞게 조정
    )

    # Setup the device on which to run the forward pass and the loss
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")    
    loss_device = torch.device("cuda") 
    print(f"Using loss_device: {loss_device}")    
        
    # Create the model and the optimizer
    model = SpeakerEncoder(device, loss_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate_init)
    
    init_step = 1
    
    # Configure file path for the model
    state_fpath = models_dir.joinpath(run_id + ".pt")
    backup_dir = models_dir.joinpath(run_id + "_backups")

    # Load any existing model
    if not force_restart:
        if state_fpath.exists():
            print("Found existing model \"%s\", loading it and resuming training." % run_id)
            checkpoint = torch.load(state_fpath)
            init_step = checkpoint["step"]
            print(f"init_step: {init_step}")
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            optimizer.param_groups[0]["lr"] = learning_rate_init
        else:
            print("No model \"%s\" found, starting training from scratch." % run_id)
    else:
        print("Starting the training from scratch.")
    model.train()
    
    # Initialize the visualization environment
    vis = Visualizations(run_id, vis_every, server=visdom_server, disabled=no_visdom)
    vis.log_dataset(train_dataset)
    vis.log_params()
    device_name = str(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
    vis.log_implementation({"Device": device_name})
        
    # Training loop
    profiler = Profiler(summarize_every=500, disabled=False)

    total_steps = len(train_loader)  # 총 스텝 수
    patience = 7  # 조기 종료를 위한 허용 스텝 수
    num_bad_epochs = 0  # 조기 종료 카운터
    best_eer = float('inf')  # 최고의 EER 초기화
    # best_loss = float('inf')  # 최고의 Loss 초기화

    with tqdm(total=total_steps, desc="Training Progress", unit="step" ,initial=init_step - 1) as pbar:
        start_time = time.time()  # 학습 시작 시간 기록
    
        for step, speaker_batch in enumerate(train_loader, init_step):
            profiler.tick("Blocking, waiting for batch (threaded)")
            
            # Forward pass
            inputs = torch.from_numpy(speaker_batch.data).to(device)
            if inputs.shape[0] != speakers_per_batch * utterances_per_speaker:
                break
            sync(device)
            embeds = model(inputs)
            sync(device)
            embeds_loss = embeds.view((speakers_per_batch, utterances_per_speaker, -1)).to(loss_device)
            loss, eer = model.loss(embeds_loss)
            sync(loss_device)
            profiler.tick("Loss")

            # Backward pass
            model.zero_grad()
            # print(f"Loss before backward: {loss.item()}")
            loss.backward()

            # Print the loss gradients after backward
            for name, param in model.named_parameters():
                if param.grad is None:
                    print(f"Gradient for {name} is None.")

            model.do_gradient_ops()
            optimizer.step()
            profiler.tick("Parameter update")
            
            # Update visualizations
            vis.update(loss.item(), eer, step)
            
            # Draw projections and save them to the backup folder
            if umap_every != 0 and step % umap_every == 0:
                print("Drawing and saving projections (step %d)" % step)
                backup_dir.mkdir(exist_ok=True)
                projection_fpath = backup_dir.joinpath("%s_umap_%06d.png" % (run_id, step))
                embeds = embeds.detach().cpu().numpy()
                vis.draw_projections(embeds, utterances_per_speaker, step, projection_fpath)
                vis.save()

            # Overwrite the latest version of the model
            if save_every != 0 and step % save_every == 0:
                print("Saving the model (step %d)" % step)
                torch.save({
                    "step": step + 1,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                }, state_fpath)
                
            # Make a backup
            if backup_every != 0 and step % backup_every == 0:
                print("Making a backup (step %d)" % step)
                backup_dir.mkdir(exist_ok=True)
                backup_fpath = backup_dir.joinpath("%s_bak_%06d.pt" % (run_id, step))
                torch.save({
                    "step": step + 1,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                }, backup_fpath)
                
            # Log training loss and EER every 7500 steps
            if step % 7500 == 0 and step != init_step:
                wandb.log({'train_loss': loss.item(), 'train_eer': eer}, step=step)
            
           
            # Validation every 7500 steps
            if step % 7500 == 0:
                start_time=time.time()
                val_loss, val_eer = validate(model, validation_loader, device, loss_device)
                end_time=time.time()
                print(f"Validation Time: {end_time - start_time:.2f} seconds")  # 소요 시간 출력

                wandb.log({'val_loss': val_loss, 'val_eer': val_eer}, step=step)
                print(f"Validation Loss: {val_loss}, Validation EER: {val_eer}")

                # 조기 종료 체크
                # if val_loss < best_loss:
                #     best_loss = val_loss
                #     num_bad_epochs = 0  # 성능이 향상되면 카운터 초기화
                # else:
                #     num_bad_epochs += 1
                
                if val_eer < best_eer:
                    best_eer = val_eer
                    num_bad_epochs = 0  # 성능이 향상되면 카운터 초기화
                    print("Saving the best model (step %d)" % step)
                    best_state_fpath=models_dir.joinpath(run_id + "_best.pt")
                    torch.save({
                        "step": step + 1,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                    }, best_state_fpath)
                else:
                    num_bad_epochs += 1

                if num_bad_epochs >= patience:
                    print("Early stopping triggered.")
                    break

                model.train()
            
            
            profiler.tick("Extras (visualizations, saving)")
            pbar.update(1)  # tqdm 진행 표시 업데이트
            
        end_time = time.time()  # 학습 종료 시간 기록
        total_time = end_time - start_time
        print(f"Total training time: {total_time // 60:.0f} minutes {total_time % 60:.2f} seconds")
