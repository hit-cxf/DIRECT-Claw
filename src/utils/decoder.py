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
