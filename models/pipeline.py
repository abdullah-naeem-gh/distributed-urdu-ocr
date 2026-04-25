import cv2
import numpy as np
import torch
from pathlib import Path
from typing import Union, Tuple, Optional

from models.restoration_model import load_restoration_model
from models.ocr_model import load_ocr_model
from models.vocab import Vocabulary
from preprocessing import standardize_and_pad

class UrduOCRPipeline:
    """
    End-to-End inference pipeline for Urdu OCR.
    Chains Noisy Input -> SMP Restoration Model -> Clean Image -> Conv-Transformer -> Text Output.
    """
    def __init__(
        self,
        restoration_ckpt: str,
        ocr_ckpt: str,
        vocab_path: str,
        device: Optional[str] = None
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.vocab = Vocabulary.load(vocab_path)
        
        # Load Phase 2 model
        self.restoration_model = load_restoration_model(restoration_ckpt, self.device)
        
        # Load Phase 3 model
        self.ocr_model = load_ocr_model(ocr_ckpt, self.vocab.size, self.device, d_model=256)

    def preprocess_image(self, image_input: Union[str, np.ndarray]) -> torch.Tensor:
        """
        Preprocess image to standard tensor shape (1, 1, 128, 2048).
        """
        if isinstance(image_input, (str, Path)):
            img = cv2.imread(str(image_input), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f"Could not load image at {image_input}")
        elif isinstance(image_input, np.ndarray):
            if len(image_input.shape) == 3:
                img = cv2.cvtColor(image_input, cv2.COLOR_BGR2GRAY)
            else:
                img = image_input.copy()
        else:
            raise TypeError("image_input must be a path string or numpy array")

        # Standardize and pad to 128x2048
        processed_img = standardize_and_pad(img, target_height=128, target_width=2048, pad_value=255)
        
        # Normalize to [0, 1]
        processed_img = processed_img.astype(np.float32) / 255.0
        
        # Create tensor (1, 1, H, W)
        tensor = torch.from_numpy(processed_img).unsqueeze(0).unsqueeze(0).to(self.device)
        return tensor

    @torch.no_grad()
    def predict(self, image_input: Union[str, np.ndarray], beam_width: int = 5) -> Tuple[np.ndarray, str]:
        """
        Run the full inference pipeline on an image.
        Returns:
            restored_image (np.ndarray): The restored/cleaned image as an 8-bit grayscale array.
            recognized_text (str): The decoded text string.
        """
        # 1. Preprocess
        input_tensor = self.preprocess_image(image_input)
        
        # 2. Restore Image
        self.restoration_model.eval()
        restored_tensor = self.restoration_model(input_tensor)
        
        # 3. OCR on Restored Image
        self.ocr_model.eval()
        beam_results = self.ocr_model.beam_search_decode(
            restored_tensor, 
            beam_width=beam_width, 
            max_len=200, 
            alpha=0.7
        )
        
        # Best prediction (top of the beam)
        best_tokens, _ = beam_results[0]
        recognized_text = self.vocab.decode(best_tokens, strip_special=True)
        
        # Process restored tensor to return as numpy array for visualization
        restored_img = restored_tensor.squeeze().cpu().numpy()
        restored_img = np.clip(restored_img * 255.0, 0, 255).astype(np.uint8)
        
        return restored_img, recognized_text
