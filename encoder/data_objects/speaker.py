from encoder.data_objects.random_cycler import RandomCycler
from encoder.data_objects.utterance import Utterance
from pathlib import Path
import os
import random

# Contains the set of utterances of a single speaker
class Speaker:
    def __init__(self, root: Path, sight_range=450):
        self.root = root
        self.name = root.name
        self.utterances = None
        self.utterance_cycler = None
        self.sight_range = sight_range  # Add sight_range parameter
        
    def _load_utterances(self, num_utterance):
        sources_file_path = self.root.joinpath("_sources.txt")
        
        # Check if _sources.txt exists
        if not sources_file_path.exists():
            print(f"File not found: {sources_file_path}")
            return  # Exit the method if the file does not exist
        
        with sources_file_path.open("r", encoding='utf-8') as sources_file:
            sources = []
            for line in sources_file:
                line = line.strip()
                if line:
                    # Split only on the first comma
                    parts = line.split(",", 1)  # Limit the split to 1 occurrence
                    if len(parts) == 2:  # Ensure there are exactly 2 parts
                        sources.append(parts)
                    else:
                        print(f"Skipping invalid line in {sources_file_path}: {line}")  # Log invalid line
                else:
                    print(f"Skipping empty line in {sources_file_path}")  # Log empty line
        
        if not sources:
            print(f"No valid entries found in {sources_file_path}")
            return  # Exit if no valid sources are found

        sight_range = min(self.sight_range, len(sources))
        if sight_range <= 0:
            print("No valid utterances available.")
            return  # exit if no valid sources are found

        if num_utterance > sight_range:
            print(f"Requested num_utterance ({num_utterance}) is greater than available utterances ({sight_range}). Setting num_utterance to sight_range.")
            num_utterance = sight_range  # Set num_utterance to sight_range if it's larger

        sight_source_idx = random.sample(range(sight_range), num_utterance)
        sight_sources = [sources[idx] for idx in sight_source_idx]
        
        sources = {frames_fname: wave_fpath for frames_fname, wave_fpath in sight_sources}
        self.utterances = [Utterance(self.root.joinpath(f), w) for f, w in sources.items()]
        self.utterance_cycler = RandomCycler(self.utterances)

    def random_partial(self, count, n_frames):
        """
        Samples a batch of <count> unique partial utterances from the disk in a way that all 
        utterances come up at least once every two cycles and in a random order every time.
        
        :param count: The number of partial utterances to sample from the set of utterances from 
        that speaker. Utterances are guaranteed not to be repeated if <count> is not larger than 
        the number of utterances available.
        :param n_frames: The number of frames in the partial utterance.
        :return: A list of tuples (utterance, frames, range) where utterance is an Utterance, 
        frames are the frames of the partial utterances and range is the range of the partial 
        utterance with regard to the complete utterance.
        """
        if self.utterances is None:
            self._load_utterances(count)  # Pass count to load_utterances

        utterances = self.utterance_cycler.sample(count)

        a = [(u,) + u.random_partial(n_frames) for u in utterances]

        return a

