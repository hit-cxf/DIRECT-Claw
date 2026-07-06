import torch
import torch.nn.functional as F
import cv2
import numpy as np

class PaddedVideoDecoder:
    """
    Wraps cv2.VideoCapture and pads frames to 16:9.
    Output: torch.Tensor(3, H, W) uint8
    """
    def __init__(self, filepath, pad_value: int = 0, device = "cpu"):
        self.cap = cv2.VideoCapture(str(filepath))
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video file: {filepath}")
        self.pad_value = pad_value
        self.target_aspect = 16 / 9
        self.device = device
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __len__(self):
        return self.total_frames
    
    def __del__(self):
        if hasattr(self, 'cap'):
            self.cap.release()

    def get_width(self) -> int:
        """Get the original video width"""
        return int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    
    def get_height(self) -> int:
        """Get the original video height"""
        return int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    def get_size(self) -> tuple:
        """Get the original video size (width, height)"""
        return (self.get_width(), self.get_height())
    
    def get_padded_size(self) -> tuple:
        """Get the padded video size (width, height)"""
        W, H = self.get_size()
        current_aspect = W / H
        
        # If frame too tall → extend width
        if current_aspect < self.target_aspect:
            new_W = int(H * self.target_aspect)
            new_H = H
        # If frame too wide → extend height
        else:
            new_H = int(W / self.target_aspect)
            new_W = W
        
        return (new_W, new_H)
    
    def get_padded_width(self) -> int:
        """Get the padded video width"""
        return self.get_padded_size()[0]
    
    def get_padded_height(self) -> int:
        """Get the padded video height"""
        return self.get_padded_size()[1]

    def _pad_to_16_9(self, frame: torch.Tensor) -> torch.Tensor:
        """
        Input frame: torch.Tensor(3, H, W), uint8
        Output frame: padded to 16:9
        """
        _, H, W = frame.shape
        new_W, new_H = self.get_padded_size()

        pad_H = new_H - H
        pad_W = new_W - W

        pad_top = pad_H // 2
        pad_bottom = pad_H - pad_top
        pad_left = pad_W // 2
        pad_right = pad_W - pad_left

        padded = F.pad(
            frame,
            pad=(pad_left, pad_right, 
                 pad_top, pad_bottom),
            mode="constant",
            value=self.pad_value
        )
        return padded

    def __getitem__(self, index):
        """
        Get frame(s) at index and return as torch.Tensor(3, H, W) uint8.
        Supports both single index and slice operations.
        """
        if isinstance(index, slice):
            # Handle slice
            start, stop, step = index.indices(self.total_frames)
            return [self._get_single_frame(i) for i in range(start, stop, step)]
        else:
            # Handle single index
            return self._get_single_frame(index)
    
    def _get_single_frame(self, index: int) -> torch.Tensor:
        """
        Get a single frame at index and return as torch.Tensor(3, H, W) uint8.
        """
        if index < 0 or index >= self.total_frames:
            raise IndexError(f"Frame index {index} out of range [0, {self.total_frames})")
        
        # Set frame position
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ret, frame = self.cap.read()
        
        if not ret:
            # Create a black frame with the padded size
            new_W, new_H = self.get_padded_size()
            black_frame = torch.full((3, new_H, new_W), self.pad_value, dtype=torch.uint8)
            return black_frame.to(self.device)
            # raise RuntimeError(f"Failed to read frame at index {index}")
        
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert numpy array (H, W, 3) to torch tensor (3, H, W)
        frame_tensor = torch.from_numpy(frame).permute(2, 0, 1)
        
        # Apply padding
        padded_frame = self._pad_to_16_9(frame_tensor)
        
        return padded_frame.to(self.device)


class KeyframeFrameCache:
    """
    Sequentially decodes one bounded keyframe chunk and keeps only that chunk's
    16:9 RGB uint8 frames in memory. Indices exposed by this cache are global
    keyframe indices, so downstream features can be stitched into the original
    full-video timeline without timestamp shifts.
    """

    def __init__(
        self,
        filepath,
        num_frames: int,
        keyframe_interval: int,
        primary_start_idx: int,
        primary_end_idx: int,
        target_size: tuple[int, int] = (640, 360),
        pad_value: int = 0,
    ):
        from tqdm import tqdm

        if primary_start_idx < 0 or primary_end_idx < primary_start_idx:
            raise ValueError(
                f"Invalid keyframe chunk [{primary_start_idx}, {primary_end_idx})"
            )

        self.filepath = filepath
        self.num_frames = num_frames
        self.keyframe_interval = keyframe_interval
        self.primary_start_idx = primary_start_idx
        self.primary_end_idx = primary_end_idx
        self.cache_start_idx = max(0, primary_start_idx - 1)
        self.cache_end_idx = primary_end_idx
        self.primary_global_indices = list(range(primary_start_idx, primary_end_idx))
        self.cached_global_indices = list(range(self.cache_start_idx, self.cache_end_idx))
        self.keyframe_indices = [idx * keyframe_interval for idx in self.cached_global_indices]
        self.target_width, self.target_height = target_size
        self.pad_value = pad_value
        self._global_to_local = {
            global_idx: local_idx for local_idx, global_idx in enumerate(self.cached_global_indices)
        }
        self.frames = torch.empty(
            (len(self.cached_global_indices), 3, self.target_height, self.target_width),
            dtype=torch.uint8,
        )

        if not self.cached_global_indices:
            return

        cap = cv2.VideoCapture(str(filepath))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {filepath}")

        first_frame_idx = self.keyframe_indices[0]
        last_frame_idx = self.keyframe_indices[-1]
        cap.set(cv2.CAP_PROP_POS_FRAMES, first_frame_idx)

        next_cache_idx = 0
        next_frame_idx = self.keyframe_indices[next_cache_idx]
        frame_idx = first_frame_idx

        desc = f"Caching keyframes {primary_start_idx}:{primary_end_idx}"
        with tqdm(total=len(self.cached_global_indices), desc=desc) as pbar:
            while next_cache_idx < len(self.cached_global_indices) and frame_idx <= last_frame_idx:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx == next_frame_idx:
                    self.frames[next_cache_idx] = self._prepare_frame(frame)
                    next_cache_idx += 1
                    pbar.update(1)
                    if next_cache_idx < len(self.cached_global_indices):
                        next_frame_idx = self.keyframe_indices[next_cache_idx]
                frame_idx += 1

        cap.release()

        if next_cache_idx < len(self.cached_global_indices):
            print(
                f"Warning: decoded {next_cache_idx}/{len(self.cached_global_indices)} "
                "cached keyframes; padding the remainder with black frames."
            )
            black = torch.full(
                (3, self.target_height, self.target_width),
                self.pad_value,
                dtype=torch.uint8,
            )
            for i in range(next_cache_idx, len(self.cached_global_indices)):
                self.frames[i] = black

    def _prepare_frame(self, frame) -> torch.Tensor:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        target_aspect = 16 / 9
        current_aspect = w / h

        if current_aspect < target_aspect:
            new_w = int(h * target_aspect)
            pad_total = new_w - w
            left = pad_total // 2
            right = pad_total - left
            frame = cv2.copyMakeBorder(
                frame, 0, 0, left, right, cv2.BORDER_CONSTANT,
                value=(self.pad_value, self.pad_value, self.pad_value),
            )
        else:
            new_h = int(w / target_aspect)
            pad_total = new_h - h
            top = pad_total // 2
            bottom = pad_total - top
            frame = cv2.copyMakeBorder(
                frame, top, bottom, 0, 0, cv2.BORDER_CONSTANT,
                value=(self.pad_value, self.pad_value, self.pad_value),
            )

        frame = cv2.resize(frame, (self.target_width, self.target_height), interpolation=cv2.INTER_AREA)
        return torch.from_numpy(frame).permute(2, 0, 1).contiguous()

    def __len__(self):
        return len(self.primary_global_indices)

    def get(self, global_keyframe_idx: int) -> torch.Tensor:
        return self.frames[self._global_to_local[global_keyframe_idx]]

    def __getitem__(self, global_keyframe_idx: int) -> torch.Tensor:
        return self.get(global_keyframe_idx)

