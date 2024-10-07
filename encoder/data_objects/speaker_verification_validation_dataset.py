from encoder.data_objects.random_cycler import RandomCycler
from encoder.data_objects.speaker_batch import SpeakerBatch
from encoder.data_objects.speaker import Speaker
from encoder.params_data import partials_n_frames
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

class SpeakerValidationDataset(Dataset):
    def __init__(self, datasets_root: Path, sight_range=450):
        self.root = datasets_root
        self.sight_range = sight_range
        print(f'self.root : {self.root}')
        speaker_dirs = [f for f in self.root.glob("*") if f.is_dir()]
        
        if len(speaker_dirs) == 0:
            raise Exception("No speakers found. Make sure you are pointing to the directory "
                            "containing all preprocessed speaker directories.")
        
        print(f'num speaker : {len(speaker_dirs)}')
        # print(f'expected data total duration : {3.9*len(speaker_dirs)} sec')
        
        # 각 화자에 대한 Speaker 객체를 생성
        self.speakers = [Speaker(speaker_dir, sight_range) for speaker_dir in speaker_dirs]
        self.len_speakers=len(speaker_dirs)
        self.speaker_cycler = RandomCycler(self.speakers)
        
    def __len__(self):
        return self.len_speakers
    
    def __getitem__(self, index):
        return next(self.speaker_cycler)  # 다음 화자 배치를 반환
    
    def get_logs(self):
        log_string = ""
        for log_fpath in self.root.glob("*.txt"):
            with log_fpath.open("r") as log_file:
                log_string += "".join(log_file.readlines())
        return log_string

class SpeakerValidationDataLoader(DataLoader):
    def __init__(self, dataset, speakers_per_batch, utterances_per_speaker, sampler=None, 
                 batch_sampler=None, num_workers=0, pin_memory=False, timeout=0, 
                 worker_init_fn=None):
        self.utterances_per_speaker = utterances_per_speaker

        super().__init__(
            dataset=dataset, 
            batch_size=speakers_per_batch, 
            shuffle=False, 
            sampler=sampler, 
            batch_sampler=batch_sampler, 
            num_workers=num_workers,
            collate_fn=self.collate, 
            pin_memory=pin_memory, 
            drop_last=False, 
            timeout=timeout, 
            worker_init_fn=worker_init_fn
        )

    def collate(self, speakers):
        return SpeakerBatch(speakers, self.utterances_per_speaker, partials_n_frames) 
