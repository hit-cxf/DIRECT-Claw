from pathlib import Path
from typing import cast
import numpy as np
import librosa
import allin1
from ..utils.path import get_data_dir, get_output_dir

class MusicResult(allin1.typings.AnalysisResult):
    """Wrapper class for music analysis results, providing easier access to attributes and methods."""
    def __init__(self, analysis_result: allin1.typings.AnalysisResult):
        super().__init__(**analysis_result.__dict__)
        
        assert self.path is not None, "AnalysisResult must have a valid path."
        assert self.path.is_file(), f"Audio file does not exist: {self.path}"
        self.y, self.sr = librosa.load(self.path, sr=None)
        
        if (len(self.y) / self.sr) - self.beats[-1] < 0.2:
            del self.beats[-1]  # Remove the last beat as it is too close to the end of the music
        self.beats.append(len(self.y) / self.sr)

        # Remove segments that do not contain any beats
        beat_set = set(self.beats)
        self.segments = [
            seg for seg in self.segments
            if any(seg.start <= b < seg.end for b in beat_set)
        ]
        
    
    def get_rms(self, start_time: float, end_time: float) -> float:
        """Get the RMS (Root Mean Square) energy for a specified time range.
        
        Parameters:
            start_time: Start time (seconds)
            end_time: End time (seconds)
        
        Returns:
            RMS value
        """
        
        # Convert time to sample indices
        start_sample = int(start_time * self.sr)
        end_sample = int(end_time * self.sr)
        
        # Ensure indices are within valid range
        start_sample = max(0, start_sample)
        end_sample = min(len(self.y), end_sample)
        
        if start_sample >= end_sample:
            return 0.0
        
        # Calculate RMS for the time range
        segment = self.y[start_sample:end_sample]
        rms = float(np.sqrt(np.mean(segment ** 2)))
        
        return rms
    
    def get_rms_quantiles(self, lower_quantile: float = 0.1, upper_quantile: float = 0.9) -> tuple[float, float]:
        """Get the quantiles of RMS energy for the entire music.
        
        Uses librosa.feature.rms to calculate the RMS feature for the entire music, then returns the specified quantiles.
        
        Parameters:
            lower_quantile: Lower quantile
            upper_quantile: Upper quantile
        
        Returns:
            (lower_value, upper_value): Lower quantile value and upper quantile value
        """
        # Use librosa to calculate RMS feature
        rms_feature = librosa.feature.rms(y=self.y, frame_length=2048, hop_length=512)[0]  # shape: (n_frames,)
        
        # Calculate quantiles
        lower_value = float(np.quantile(rms_feature, lower_quantile))
        upper_value = float(np.quantile(rms_feature, upper_quantile))
        
        if lower_quantile == upper_quantile:
            upper_value = lower_value + 1e-6  # Avoid division by zero when lower and upper quantiles are equal

        return lower_value, upper_value

def analyze_music(music_path: Path | str, is_absolute: bool = False) -> MusicResult:
    """Analyze music and write the results to a cache file, returning a MusicResult object.
    
    Parameters:
        music_path: Path to the music or video file
        is_absolute: Whether the path is absolute (True) or relative to data_dir (False)
    
    Note: allin1.analyze can directly process video files and will automatically extract audio for analysis.
    """
    music_path = Path(music_path)
    
    if is_absolute:
        data_file = music_path
        cache_file = music_path.with_suffix('.json')
    else:
        data_file = get_data_dir() / music_path
        cache_file = get_output_dir() / music_path.with_suffix('.json')
    
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    
    if cache_file.exists():
        result = allin1.load_result(cache_file)
        if isinstance(result, list):
            # If the analysis returned multiple results, use the first one
            result = result[0]
    else:
        analyzed = allin1.analyze(data_file, out_dir=cache_file.parent)
        result = cast(allin1.typings.AnalysisResult, analyzed[0] if isinstance(analyzed, list) else analyzed)
    
    return MusicResult(result)

def get_music_info(music_prof: MusicResult) -> str:
    beats = list(zip(music_prof.beats, music_prof.beat_positions))
    # Note: Identifying the time signature/meter (e.g., 4/4) 
    beat_count = sorted(music_prof.beat_positions)[int(0.95 * len(music_prof.beat_positions))]
    sections = [
        (section, [(beat, position) for beat, position in beats if section.start <= beat < section.end])
        for section in music_prof.segments
    ]
    
    sections_info = []
    for section, beat_info in sections:
        # Calculate the RMS and volume (dB) for the section
        rms = music_prof.get_rms(section.start, section.end)
        # Convert to dB, reference value is 1.0 (normalized audio)
        # Avoid log(0) by setting a minimum value
        db = 20 * np.log10(max(rms, 1e-10))
        
        sections_info.append(
            f"- {section.label} ({section.start:.2f}s - {section.end:.2f}s), "
            f"{len(beat_info)} beats total, {db:.1f} dB"
        )
    
    music_info = (
        f"This music has a total duration of {music_prof.segments[-1].end}s, a BPM of {music_prof.bpm:.0f}, "
        f"and is in {beat_count}/4 time. It contains {len(music_prof.segments)} sections.\n"
        + "\n".join(sections_info)
    )
    return music_info