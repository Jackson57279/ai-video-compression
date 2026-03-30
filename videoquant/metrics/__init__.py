"""VideoQuant Quality Metrics Implementation

Implements video quality evaluation metrics for quantization validation:
- FID (Frechet Inception Distance): Measures video quality vs reference
- CLIPSIM (CLIP Similarity): Measures text-video alignment
- Temporal Consistency: Measures frame-to-frame smoothness

Validates:
- VAL-QTY-001: FID computation accurate to reference implementation
- VAL-QTY-002: CLIPSIM captures text-video alignment
- VAL-QTY-003: Temporal consistency measures smoothness
"""

import torch
import numpy as np
from typing import List, Tuple, Optional, Union, Dict, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class MetricResult:
    """Result container for a quality metric computation."""
    value: float
    metric_name: str
    confidence_interval: Optional[Tuple[float, float]] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            'value': float(self.value),
            'metric_name': self.metric_name,
            'confidence_interval': self.confidence_interval,
            'metadata': self.metadata,
        }


class VideoMetric(ABC):
    """Abstract base class for video quality metrics."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Metric name."""
        pass
    
    @abstractmethod
    def compute(
        self,
        videos: Union[torch.Tensor, List[torch.Tensor]],
        references: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        text_prompts: Optional[List[str]] = None,
    ) -> MetricResult:
        """Compute metric for given videos.
        
        Args:
            videos: Generated video tensor(s) [B, F, C, H, W] or list of [F, C, H, W]
            references: Reference video tensor(s) for comparison (for FID)
            text_prompts: Text prompts for CLIP-based metrics
            
        Returns:
            MetricResult with computed value and metadata
        """
        pass
    
    def compute_batch(
        self,
        videos: List[torch.Tensor],
        references: Optional[List[torch.Tensor]] = None,
        text_prompts: Optional[List[str]] = None,
    ) -> List[MetricResult]:
        """Compute metric for a batch of videos."""
        return [self.compute(v, references, text_prompts) for v in videos]


def _preprocess_video_frames(
    video: torch.Tensor,
    target_size: Tuple[int, int] = (299, 299),
) -> torch.Tensor:
    """Preprocess video frames for feature extraction.
    
    Args:
        video: Video tensor [F, C, H, W] or [B, F, C, H, W]
        target_size: Target size for resizing (H, W)
        
    Returns:
        Preprocessed frames [F, 3, 299, 299] or [B, F, 3, 299, 299]
    """
    import torch.nn.functional as F
    
    # Ensure float tensor in [0, 1] range
    if video.dtype == torch.uint8:
        video = video.float() / 255.0
    
    # Normalize to [-1, 1] for Inception
    video = video * 2.0 - 1.0
    
    # Resize if needed
    if video.dim() == 4:  # [F, C, H, W]
        if video.shape[-2:] != target_size:
            video = F.interpolate(
                video, size=target_size, mode='bilinear', align_corners=False
            )
    elif video.dim() == 5:  # [B, F, C, H, W]
        B, F, C, H, W = video.shape
        video = video.reshape(B * F, C, H, W)
        if video.shape[-2:] != target_size:
            video = F.interpolate(
                video, size=target_size, mode='bilinear', align_corners=False
            )
        video = video.reshape(B, F, C, *target_size)
    
    return video


def _extract_video_features(
    video: torch.Tensor,
    feature_extractor: torch.nn.Module,
    batch_size: int = 8,
) -> torch.Tensor:
    """Extract features from video frames.
    
    Args:
        video: Video tensor [F, C, H, W] or [B, F, C, H, W]
        feature_extractor: Neural network for feature extraction
        batch_size: Batch size for feature extraction
        
    Returns:
        Features tensor [F, D] or [B, F, D]
    """
    if video.dim() == 5:  # [B, F, C, H, W]
        B, F, C, H, W = video.shape
        frames = video.reshape(B * F, C, H, W)
    else:  # [F, C, H, W]
        frames = video
    
    features_list = []
    
    with torch.no_grad():
        for i in range(0, frames.shape[0], batch_size):
            batch = frames[i:i + batch_size]
            feats = feature_extractor(batch)
            if isinstance(feats, tuple):
                feats = feats[0]
            # Global average pooling if spatial features
            if feats.dim() == 4:  # [B, C, H, W]
                feats = feats.mean(dim=[2, 3])
            features_list.append(feats)
    
    features = torch.cat(features_list, dim=0)
    
    if video.dim() == 5:
        features = features.reshape(B, F, -1)
    
    return features


class FIDMetric(VideoMetric):
    """Frechet Inception Distance (FID) for video quality assessment.
    
    FID measures the distance between feature distributions of generated
    and reference videos using InceptionV3 features.
    
    Lower FID = better quality (more similar to reference).
    Typical range: 0-300 (0 is identical to reference).
    
    Implementation based on:
    - Heusel et al. "GANs Trained by a Two Time-Scale Update Rule"
    """
    
    def __init__(
        self,
        feature_extractor: Optional[torch.nn.Module] = None,
        batch_size: int = 8,
        device: Optional[str] = None,
    ):
        """Initialize FID metric.
        
        Args:
            feature_extractor: Optional custom feature extractor
            batch_size: Batch size for feature extraction
            device: Device for computation
        """
        self._feature_extractor = feature_extractor
        self.batch_size = batch_size
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self._inception_model: Optional[torch.nn.Module] = None
    
    @property
    def name(self) -> str:
        return "fid"
    
    def _get_inception_model(self) -> torch.nn.Module:
        """Lazy-load InceptionV3 model."""
        if self._inception_model is None:
            try:
                from torchvision.models import inception_v3, Inception_V3_Weights
                self._inception_model = inception_v3(
                    weights=Inception_V3_Weights.IMAGENET1K_V1,
                    transform_input=False,
                )
                self._inception_model.fc = torch.nn.Identity()  # Remove classifier
                self._inception_model = self._inception_model.to(self.device)
                self._inception_model.eval()
            except ImportError:
                raise ImportError(
                    "torchvision is required for FID computation. "
                    "Install with: pip install torchvision"
                )
        return self._inception_model
    
    def _extract_features(self, video: torch.Tensor) -> torch.Tensor:
        """Extract Inception features from video."""
        # Preprocess to InceptionV3 input format
        video = _preprocess_video_frames(video, target_size=(299, 299))
        
        # Get feature extractor
        if self._feature_extractor is not None:
            extractor = self._feature_extractor
        else:
            extractor = self._get_inception_model()
        
        # Extract features
        video = video.to(self.device)
        
        if video.dim() == 5:  # [B, F, C, H, W]
            B, F, C, H, W = video.shape
            video = video.reshape(B * F, C, H, W)
        
        features = _extract_video_features(
            video, extractor, batch_size=self.batch_size
        )
        
        return features.cpu()
    
    def _compute_statistics(
        self,
        features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute mean and covariance of features."""
        mu = features.mean(dim=0)
        sigma = torch.cov(features.T)
        return mu, sigma
    
    def _compute_fid(
        self,
        mu1: torch.Tensor,
        sigma1: torch.Tensor,
        mu2: torch.Tensor,
        sigma2: torch.Tensor,
        eps: float = 1e-6,
    ) -> float:
        """Compute Frechet distance between two Gaussians.
        
        FID = ||mu1 - mu2||^2 + Tr(sigma1 + sigma2 - 2*sqrt(sigma1*sigma2))
        """
        # Move to numpy for stable computation
        mu1_np = mu1.cpu().numpy()
        mu2_np = mu2.cpu().numpy()
        sigma1_np = sigma1.cpu().numpy()
        sigma2_np = sigma2.cpu().numpy()
        
        # Mean difference
        diff = mu1_np - mu2_np
        
        # Product of covariances
        covmean, _ = self._sqrtm_psd(sigma1_np @ sigma2_np)
        
        # FID formula
        fid = np.dot(diff, diff) + np.trace(sigma1_np + sigma2_np - 2 * covmean)
        
        return float(fid)
    
    def _sqrtm_psd(
        self,
        matrix: np.ndarray,
        eps: float = 1e-6,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute matrix square root for positive semi-definite matrix."""
        # Eigenvalue decomposition
        eigvals, eigvecs = np.linalg.eigh(matrix)
        
        # Ensure positive eigenvalues
        eigvals = np.maximum(eigvals, eps)
        
        # Compute square root
        sqrt_eigvals = np.sqrt(eigvals)
        sqrtm = eigvecs @ np.diag(sqrt_eigvals) @ eigvecs.T
        
        return sqrtm, eigvecs @ np.diag(eigvals) @ eigvecs.T
    
    def compute(
        self,
        videos: Union[torch.Tensor, List[torch.Tensor]],
        references: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        text_prompts: Optional[List[str]] = None,
    ) -> MetricResult:
        """Compute FID between generated and reference videos.
        
        Args:
            videos: Generated videos [B, F, C, H, W] or list of [F, C, H, W]
            references: Reference videos for comparison
            
        Returns:
            MetricResult with FID value (lower is better)
            
        Raises:
            ValueError: If references are not provided
        """
        if references is None:
            raise ValueError("FID requires reference videos for comparison")
        
        # Convert lists to tensors
        if isinstance(videos, list):
            videos = torch.stack(videos)
        if isinstance(references, list):
            references = torch.stack(references)
        
        # Extract features
        gen_features = self._extract_features(videos)
        ref_features = self._extract_features(references)
        
        # Compute statistics
        mu_gen, sigma_gen = self._compute_statistics(gen_features)
        mu_ref, sigma_ref = self._compute_statistics(ref_features)
        
        # Compute FID
        fid_value = self._compute_fid(mu_gen, sigma_gen, mu_ref, sigma_ref)
        
        # Ensure FID is non-negative
        fid_value = max(0.0, fid_value)
        
        return MetricResult(
            value=fid_value,
            metric_name=self.name,
            metadata={
                'mu_gen_mean': float(mu_gen.mean()),
                'mu_ref_mean': float(mu_ref.mean()),
                'n_gen_samples': gen_features.shape[0],
                'n_ref_samples': ref_features.shape[0],
                'feature_dim': gen_features.shape[1],
            }
        )


class CLIPSIMMetric(VideoMetric):
    """CLIP text-video similarity metric.
    
    Measures alignment between text prompts and generated video frames
    using CLIP embeddings. Higher is better.
    
    Implementation based on:
    - Radford et al. "Learning Transferable Visual Models from Natural Language Supervision"
    """
    
    def __init__(
        self,
        model_name: str = "openai/clip-vit-large-patch14",
        batch_size: int = 8,
        device: Optional[str] = None,
        aggregation: str = "mean",
    ):
        """Initialize CLIPSIM metric.
        
        Args:
            model_name: CLIP model name from HuggingFace
            batch_size: Batch size for encoding
            device: Device for computation
            aggregation: How to aggregate frame scores ("mean", "max", "first")
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.aggregation = aggregation
        self._clip_model = None
        self._clip_processor = None
    
    @property
    def name(self) -> str:
        return "clipsim"
    
    def _get_clip_model(self):
        """Lazy-load CLIP model and processor."""
        if self._clip_model is None:
            try:
                from transformers import CLIPModel, CLIPProcessor
                self._clip_model = CLIPModel.from_pretrained(self.model_name)
                self._clip_processor = CLIPProcessor.from_pretrained(self.model_name)
                self._clip_model = self._clip_model.to(self.device)
                self._clip_model.eval()
            except ImportError:
                raise ImportError(
                    "transformers is required for CLIPSIM computation. "
                    "Install with: pip install transformers"
                )
        return self._clip_model, self._clip_processor
    
    def _encode_text(self, texts: List[str]) -> torch.Tensor:
        """Encode text prompts with CLIP."""
        model, processor = self._get_clip_model()
        
        # Process text
        inputs = processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            text_features = model.get_text_features(**inputs)
        
        # Handle BaseModelOutputWithPooling (newer transformers) vs tensor (older)
        if hasattr(text_features, 'pooler_output'):
            text_features = text_features.pooler_output
        elif hasattr(text_features, 'last_hidden_state'):
            text_features = text_features.last_hidden_state.mean(dim=1)
        
        # Normalize
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return text_features.cpu()
    
    def _encode_video_frames(
        self,
        video: torch.Tensor,
    ) -> torch.Tensor:
        """Encode video frames with CLIP.
        
        Args:
            video: Video tensor [F, C, H, W] or [B, F, C, H, W]
            
        Returns:
            Frame features [F, D] or [B, F, D]
        """
        model, processor = self._get_clip_model()
        
        # Handle batch dimension
        if video.dim() == 5:  # [B, F, C, H, W]
            B, F, C, H, W = video.shape
            frames = video.reshape(B * F, C, H, W)
            is_batched = True
        else:  # [F, C, H, W]
            frames = video
            is_batched = False
        
        # Convert to PIL-like format [0, 255] uint8 for processor
        if frames.dtype == torch.float32 or frames.dtype == torch.float64:
            if frames.max() <= 1.0:
                frames = (frames * 255).clamp(0, 255).byte()
        
        all_features = []
        
        with torch.no_grad():
            for i in range(0, frames.shape[0], self.batch_size):
                batch = frames[i:i + self.batch_size]
                
                # Convert to list of PIL images
                pil_images = []
                for frame in batch:
                    # Convert [C, H, W] to PIL
                    frame_np = frame.cpu().permute(1, 2, 0).numpy()
                    from PIL import Image
                    pil_images.append(Image.fromarray(frame_np))
                
                # Process with CLIP
                inputs = processor(images=pil_images, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Get image features - handle both new and old transformers API
                image_features = model.get_image_features(**inputs)
                
                # Handle BaseModelOutputWithPooling (newer transformers) vs tensor (older)
                if hasattr(image_features, 'pooler_output'):
                    image_features = image_features.pooler_output
                elif hasattr(image_features, 'last_hidden_state'):
                    image_features = image_features.last_hidden_state.mean(dim=1)
                
                # Normalize
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                all_features.append(image_features.cpu())
        
        features = torch.cat(all_features, dim=0)
        
        if is_batched:
            features = features.reshape(B, F, -1)
        
        return features
    
    def compute(
        self,
        videos: Union[torch.Tensor, List[torch.Tensor]],
        references: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        text_prompts: Optional[List[str]] = None,
    ) -> MetricResult:
        """Compute CLIP similarity between videos and text prompts.
        
        Args:
            videos: Video tensor(s) [B, F, C, H, W] or list of [F, C, H, W]
            text_prompts: Text prompts for comparison (required)
            
        Returns:
            MetricResult with CLIPSIM value [0, 1] (higher is better)
            
        Raises:
            ValueError: If text_prompts is not provided
        """
        if text_prompts is None:
            raise ValueError("CLIPSIM requires text prompts")
        
        # Convert list to tensor if needed
        if isinstance(videos, list):
            videos = torch.stack(videos)
        
        # Ensure batch dimension
        if videos.dim() == 4:  # [F, C, H, W]
            videos = videos.unsqueeze(0)  # [1, F, C, H, W]
        
        B, F, C, H, W = videos.shape
        
        # Encode video frames
        video_features = self._encode_video_frames(videos)  # [B, F, D]
        
        # Encode text (one prompt per video)
        if len(text_prompts) < B:
            # Repeat prompts if not enough
            text_prompts = text_prompts * ((B // len(text_prompts)) + 1)
        text_prompts = text_prompts[:B]
        
        text_features = self._encode_text(text_prompts)  # [B, D]
        
        # Compute similarity per frame
        similarities = []
        for i in range(B):
            frame_feats = video_features[i]  # [F, D]
            text_feat = text_features[i:i+1]  # [1, D]
            
            # Cosine similarity between each frame and text
            # Ensure features are normalized first
            frame_feats = frame_feats / (frame_feats.norm(dim=-1, keepdim=True) + 1e-10)
            text_feat = text_feat / (text_feat.norm(dim=-1, keepdim=True) + 1e-10)
            
            sim = torch.matmul(frame_feats, text_feat.T).squeeze()  # [F]
            
            # Clip to [-1, 1] range (numerical stability)
            sim = torch.clamp(sim, -1.0, 1.0)
            
            # Aggregate frame scores
            if self.aggregation == "mean":
                video_sim = sim.mean()
            elif self.aggregation == "max":
                video_sim = sim.max()
            elif self.aggregation == "first":
                video_sim = sim[0]
            else:
                video_sim = sim.mean()
            
            similarities.append(video_sim.item())
        
        # Average across batch
        # Raw cosine similarity is in [-1, 1], convert to [0, 1]
        clipsim_value_raw = np.mean(similarities)
        clipsim_value = (clipsim_value_raw + 1.0) / 2.0
        
        return MetricResult(
            value=clipsim_value,
            metric_name=self.name,
            metadata={
                'aggregation': self.aggregation,
                'n_frames': F,
                'n_videos': B,
                'per_video_scores': similarities,
                'model': self.model_name,
            }
        )


class TemporalConsistencyMetric(VideoMetric):
    """Temporal consistency metric for video smoothness.
    
    Measures frame-to-frame consistency to detect flickering and
    temporal artifacts. Higher is better (more consistent).
    
    Implements:
    - Frame difference analysis
    - Optical flow consistency (if available)
    - Motion smoothness metrics
    
    Validated against: VAL-QTY-003
    """
    
    def __init__(
        self,
        method: str = "frame_diff",
        flow_method: Optional[str] = None,
        device: Optional[str] = None,
    ):
        """Initialize temporal consistency metric.
        
        Args:
            method: Consistency method ("frame_diff", "flow", "combined")
            flow_method: Optical flow method (None, "farneback", "deepflow")
            device: Device for computation
        """
        self.method = method
        self.flow_method = flow_method
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    
    @property
    def name(self) -> str:
        return "temporal_consistency"
    
    def _compute_frame_difference_consistency(
        self,
        video: torch.Tensor,
    ) -> float:
        """Compute temporal consistency via frame differences.
        
        Measures how similar consecutive frames are.
        High consistency = smooth motion, low frame-to-frame changes.
        
        Args:
            video: Video tensor [F, C, H, W]
            
        Returns:
            Consistency score [0, 1] (higher is better)
        """
        F, C, H, W = video.shape
        
        if F < 2:
            return 1.0  # Single frame is perfectly consistent
        
        # Normalize to [0, 1]
        if video.max() > 1.0:
            video_norm = video / 255.0
        else:
            video_norm = video
        
        # Compute frame differences
        frame_diffs = []
        for i in range(F - 1):
            frame1 = video_norm[i]
            frame2 = video_norm[i + 1]
            
            # L2 difference
            diff = torch.norm(frame1 - frame2)
            
            # Normalize by frame size
            diff_normalized = diff / (C * H * W)
            frame_diffs.append(diff_normalized.item())
        
        frame_diffs = np.array(frame_diffs)
        
        # Consistency: inverse of mean difference (normalized)
        mean_diff = frame_diffs.mean()
        
        # Convert to consistency score [0, 1]
        # Lower difference = higher consistency
        # Use exponential decay for smoother curve
        consistency = np.exp(-mean_diff * 10)
        
        # Also consider variance - high variance = flickering
        variance_penalty = 1.0 / (1.0 + frame_diffs.std())
        
        # Combine
        final_consistency = 0.7 * consistency + 0.3 * variance_penalty
        
        return float(np.clip(final_consistency, 0.0, 1.0))
    
    def _compute_optical_flow_consistency(
        self,
        video: torch.Tensor,
    ) -> float:
        """Compute temporal consistency via optical flow smoothness.
        
        Uses optical flow between frames to measure motion smoothness.
        
        Args:
            video: Video tensor [F, C, H, W]
            
        Returns:
            Consistency score [0, 1] (higher is better)
            
        Note:
            Returns frame_diff consistency if flow computation fails
        """
        try:
            import cv2
        except ImportError:
            # Fall back to frame difference
            return self._compute_frame_difference_consistency(video)
        
        F, C, H, W = video.shape
        
        if F < 2:
            return 1.0
        
        # Convert to numpy [F, H, W, C] uint8
        if video.dtype == torch.float32:
            if video.max() <= 1.0:
                video_np = (video * 255).cpu().numpy().astype(np.uint8)
            else:
                video_np = video.cpu().numpy().astype(np.uint8)
        else:
            video_np = video.cpu().numpy()
        
        video_np = np.transpose(video_np, (0, 2, 3, 1))  # [F, H, W, C]
        
        # Convert to grayscale if needed
        if C == 3:
            gray_frames = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in video_np]
        else:
            gray_frames = [f[:, :, 0] for f in video_np]
        
        # Compute optical flow between consecutive frames
        flow_magnitudes = []
        flow_variations = []
        
        for i in range(len(gray_frames) - 1):
            flow = cv2.calcOpticalFlowFarneback(
                gray_frames[i], gray_frames[i + 1],
                None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            
            # Flow magnitude
            magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
            flow_magnitudes.append(magnitude.mean())
            
            # Flow variation (smoothness)
            if i > 0:
                flow_var = np.abs(magnitude.mean() - flow_magnitudes[-2])
                flow_variations.append(flow_var)
        
        if not flow_magnitudes:
            return 1.0
        
        # Consistency metrics
        mean_magnitude = np.mean(flow_magnitudes)
        
        # Lower flow = more consistency (less motion)
        # But we also want smooth flow (low variation)
        if flow_variations:
            flow_smoothness = 1.0 / (1.0 + np.mean(flow_variations))
        else:
            flow_smoothness = 1.0
        
        # Normalize flow magnitude (lower is better for static scenes,
        # but we don't penalize smooth motion too much)
        flow_consistency = np.exp(-mean_magnitude / 10.0)
        
        # Combine
        consistency = 0.5 * flow_consistency + 0.5 * flow_smoothness
        
        return float(np.clip(consistency, 0.0, 1.0))
    
    def compute(
        self,
        videos: Union[torch.Tensor, List[torch.Tensor]],
        references: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        text_prompts: Optional[List[str]] = None,
    ) -> MetricResult:
        """Compute temporal consistency for videos.
        
        Args:
            videos: Video tensor(s) [B, F, C, H, W] or list of [F, C, H, W]
            references: Optional reference videos (not used)
            text_prompts: Text prompts (not used)
            
        Returns:
            MetricResult with consistency score [0, 1] (higher is better)
        """
        # Convert to list of single videos
        if isinstance(videos, torch.Tensor):
            if videos.dim() == 5:  # [B, F, C, H, W]
                videos = [videos[i] for i in range(videos.shape[0])]
            else:  # [F, C, H, W]
                videos = [videos]
        
        scores = []
        
        for video in videos:
            if self.method == "frame_diff":
                score = self._compute_frame_difference_consistency(video)
            elif self.method == "flow":
                score = self._compute_optical_flow_consistency(video)
            elif self.method == "combined":
                score_diff = self._compute_frame_difference_consistency(video)
                score_flow = self._compute_optical_flow_consistency(video)
                score = 0.5 * score_diff + 0.5 * score_flow
            else:
                score = self._compute_frame_difference_consistency(video)
            
            scores.append(score)
        
        # Average across all videos
        consistency_value = np.mean(scores)
        
        return MetricResult(
            value=consistency_value,
            metric_name=self.name,
            metadata={
                'method': self.method,
                'flow_method': self.flow_method,
                'n_videos': len(videos),
                'per_video_scores': scores,
            }
        )


class VideoMetricsEvaluator:
    """Unified evaluator for all video quality metrics.
    
    Computes FID, CLIPSIM, and temporal consistency in a single pass.
    
    Example:
        >>> evaluator = VideoMetricsEvaluator()
        >>> results = evaluator.evaluate(
        ...     videos=generated_videos,
        ...     references=reference_videos,
        ...     text_prompts=["a cat playing"],
        ... )
        >>> print(f"FID: {results['fid'].value:.2f}")
        >>> print(f"CLIPSIM: {results['clipsim'].value:.4f}")
    """
    
    def __init__(
        self,
        enable_fid: bool = True,
        enable_clipsim: bool = True,
        enable_temporal: bool = True,
        device: Optional[str] = None,
    ):
        """Initialize metrics evaluator.
        
        Args:
            enable_fid: Enable FID metric
            enable_clipsim: Enable CLIPSIM metric
            enable_temporal: Enable temporal consistency
            device: Device for computation
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.metrics: Dict[str, VideoMetric] = {}
        
        if enable_fid:
            self.metrics['fid'] = FIDMetric(device=device)
        if enable_clipsim:
            self.metrics['clipsim'] = CLIPSIMMetric(device=device)
        if enable_temporal:
            self.metrics['temporal_consistency'] = TemporalConsistencyMetric(device=device)
    
    def evaluate(
        self,
        videos: Union[torch.Tensor, List[torch.Tensor]],
        references: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        text_prompts: Optional[List[str]] = None,
    ) -> Dict[str, MetricResult]:
        """Evaluate all enabled metrics.
        
        Args:
            videos: Generated video tensor(s)
            references: Reference videos (for FID)
            text_prompts: Text prompts (for CLIPSIM)
            
        Returns:
            Dictionary mapping metric names to MetricResult objects
        """
        results = {}
        
        for name, metric in self.metrics.items():
            try:
                result = metric.compute(videos, references, text_prompts)
                results[name] = result
            except Exception as e:
                # Store error in metadata
                results[name] = MetricResult(
                    value=float('nan'),
                    metric_name=name,
                    metadata={'error': str(e)}
                )
        
        return results
    
    def compute_metric_preservation(
        self,
        baseline_results: Dict[str, MetricResult],
        quantized_results: Dict[str, MetricResult],
    ) -> Dict[str, float]:
        """Compute metric preservation percentages.
        
        Compares quantized results against baseline and reports
        percentage preservation (target: >99%).
        
        Args:
            baseline_results: Results from FP16 baseline
            quantized_results: Results from quantized model
            
        Returns:
            Dictionary mapping metric names to preservation percentage
        """
        preservation = {}
        
        for name in baseline_results.keys():
            if name not in quantized_results:
                continue
            
            baseline_val = baseline_results[name].value
            quantized_val = quantized_results[name].value
            
            # Handle directionality
            # For FID: lower is better, so we compare differently
            if name == 'fid':
                # FID: want quantized <= 1.01 * baseline
                # Preservation = baseline / max(quantized, eps)
                if quantized_val > 0:
                    preservation[name] = (baseline_val / quantized_val) * 100
                else:
                    preservation[name] = 100.0
            else:
                # Higher is better metrics
                if baseline_val > 0:
                    preservation[name] = (quantized_val / baseline_val) * 100
                else:
                    preservation[name] = 100.0 if quantized_val == 0 else 0.0
        
        return preservation


def compute_fid(
    generated_videos: Union[torch.Tensor, List[torch.Tensor]],
    reference_videos: Union[torch.Tensor, List[torch.Tensor]],
    device: Optional[str] = None,
) -> float:
    """Convenience function to compute FID.
    
    Args:
        generated_videos: Generated video tensor(s)
        reference_videos: Reference video tensor(s)
        device: Device for computation
        
    Returns:
        FID score (lower is better)
    """
    metric = FIDMetric(device=device)
    result = metric.compute(generated_videos, reference_videos)
    return result.value


def compute_clipsim(
    videos: Union[torch.Tensor, List[torch.Tensor]],
    text_prompts: List[str],
    device: Optional[str] = None,
) -> float:
    """Convenience function to compute CLIPSIM.
    
    Args:
        videos: Video tensor(s)
        text_prompts: Text prompts
        device: Device for computation
        
    Returns:
        CLIPSIM score [0, 1] (higher is better)
    """
    metric = CLIPSIMMetric(device=device)
    result = metric.compute(videos, text_prompts=text_prompts)
    return result.value


def compute_temporal_consistency(
    videos: Union[torch.Tensor, List[torch.Tensor]],
    method: str = "frame_diff",
    device: Optional[str] = None,
) -> float:
    """Convenience function to compute temporal consistency.
    
    Args:
        videos: Video tensor(s)
        method: Consistency method ("frame_diff", "flow", "combined")
        device: Device for computation
        
    Returns:
        Temporal consistency score [0, 1] (higher is better)
    """
    metric = TemporalConsistencyMetric(method=method, device=device)
    result = metric.compute(videos)
    return result.value
