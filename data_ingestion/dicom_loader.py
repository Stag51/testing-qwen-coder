"""
DICOM and Medical Image Data Ingestion Module
Handles loading, preprocessing, and feature extraction from medical images
"""
import pydicom
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
from loguru import logger
import SimpleITK as sitk


class DICOMProcessor:
    """
    Process DICOM files for multi-modal diagnostic model
    Supports CT, MRI, and other radiological modalities
    """
    
    def __init__(
        self,
        target_size: Tuple[int, int, int] = (128, 128, 128),
        normalize: bool = True,
        window_center: Optional[float] = None,
        window_width: Optional[float] = None
    ):
        self.target_size = target_size
        self.normalize = normalize
        self.window_center = window_center
        self.window_width = window_width
        
        logger.info(f"DICOMProcessor initialized with target size {target_size}")
    
    def load_dicom_series(self, dicom_dir: Union[str, Path]) -> np.ndarray:
        """
        Load a complete DICOM series (e.g., all slices of a CT scan)
        
        Args:
            dicom_dir: Directory containing DICOM files
            
        Returns:
            3D numpy array of the volume
        """
        dicom_dir = Path(dicom_dir)
        
        # Use SimpleITK for robust series loading
        reader = sitk.ImageSeriesReader()
        series_ids = reader.GetGDCMSeriesIDs(str(dicom_dir))
        
        if not series_ids:
            raise ValueError(f"No DICOM series found in {dicom_dir}")
        
        series_file_names = reader.GetGDCMSeriesFileNames(str(dicom_dir), series_ids[0])
        reader.SetFileNames(series_file_names)
        
        # Load the image
        image = reader.Execute()
        
        # Convert to numpy array
        volume = sitk.GetArrayFromImage(image)
        
        logger.info(f"Loaded DICOM series: {len(series_file_names)} slices, shape {volume.shape}")
        return volume
    
    def load_single_dicom(self, dicom_path: Union[str, Path]) -> pydicom.Dataset:
        """Load a single DICOM file"""
        return pydicom.dcmread(dicom_path)
    
    def apply_windowing(self, volume: np.ndarray) -> np.ndarray:
        """
        Apply CT windowing (Hounsfield units)
        
        Args:
            volume: Input volume in Hounsfield units
            window_center: Center of the window
            window_width: Width of the window
            
        Returns:
            Windowed volume
        """
        if self.window_center is None or self.window_width is None:
            return volume
        
        min_val = self.window_center - self.window_width / 2
        max_val = self.window_center + self.window_width / 2
        
        windowed = np.clip(volume, min_val, max_val)
        windowed = (windowed - min_val) / (max_val - min_val)
        
        return windowed
    
    def resize_volume(self, volume: np.ndarray) -> np.ndarray:
        """Resize volume to target dimensions"""
        import scipy.ndimage
        
        zoom_factors = [
            self.target_size[i] / volume.shape[i]
            for i in range(min(3, len(volume.shape)))
        ]
        
        resized = scipy.ndimage.zoom(volume, zoom_factors, order=1)
        return resized
    
    def normalize_volume(self, volume: np.ndarray) -> np.ndarray:
        """Normalize volume to [0, 1] range"""
        min_val = volume.min()
        max_val = volume.max()
        
        if max_val - min_val < 1e-6:
            return np.zeros_like(volume)
        
        normalized = (volume - min_val) / (max_val - min_val)
        return normalized
    
    def preprocess(self, volume: np.ndarray) -> np.ndarray:
        """
        Complete preprocessing pipeline
        
        Args:
            volume: Raw volume array
            
        Returns:
            Preprocessed volume ready for model input
        """
        # Apply windowing if specified
        if self.window_center is not None:
            volume = self.apply_windowing(volume)
        
        # Resize to target dimensions
        volume = self.resize_volume(volume)
        
        # Normalize
        if self.normalize:
            volume = self.normalize_volume(volume)
        
        return volume
    
    def process_to_tensor(
        self,
        dicom_source: Union[str, Path],
        add_batch_dim: bool = True
    ) -> torch.Tensor:
        """
        Process DICOM data to PyTorch tensor
        
        Args:
            dicom_source: Path to DICOM file or directory
            add_batch_dim: Whether to add batch dimension
            
        Returns:
            Tensor of shape (C, D, H, W) or (B, C, D, H, W)
        """
        if Path(dicom_source).is_dir():
            volume = self.load_dicom_series(dicom_source)
        else:
            # Single file - might be part of a series
            volume = self.load_dicom_series(Path(dicom_source).parent)
        
        # Preprocess
        processed = self.preprocess(volume)
        
        # Add channel dimension
        if len(processed.shape) == 3:
            processed = processed[np.newaxis, ...]
        
        # Convert to tensor
        tensor = torch.FloatTensor(processed)
        
        if add_batch_dim:
            tensor = tensor.unsqueeze(0)
        
        return tensor
    
    def extract_metadata(self, dicom_path: Union[str, Path]) -> Dict:
        """
        Extract relevant metadata from DICOM file
        
        Args:
            dicom_path: Path to DICOM file
            
        Returns:
            Dictionary of metadata
        """
        ds = self.load_single_dicom(dicom_path)
        
        metadata = {
            'patient_id': getattr(ds, 'PatientID', 'Unknown'),
            'study_date': getattr(ds, 'StudyDate', 'Unknown'),
            'modality': getattr(ds, 'Modality', 'Unknown'),
            'body_part': getattr(ds, 'BodyPartExamined', 'Unknown'),
            'slice_thickness': getattr(ds, 'SliceThickness', None),
            'pixel_spacing': getattr(ds, 'PixelSpacing', None),
            'image_orientation': getattr(ds, 'ImageOrientationPatient', None),
        }
        
        return metadata


class GenomicDataProcessor:
    """
    Process genomic sequence data for diagnostic model
    Handles DNA/RNA sequences, VCF files, and expression data
    """
    
    def __init__(
        self,
        max_seq_len: int = 1000,
        vocab: Optional[Dict[str, int]] = None
    ):
        self.max_seq_len = max_seq_len
        
        # Default DNA vocabulary
        self.vocab = vocab or {
            'A': 0,
            'C': 1,
            'G': 2,
            'T': 3,
            'N': 4,  # Unknown base
            '<PAD>': 5,
            '<START>': 6,
            '<END>': 7
        }
        
        logger.info(f"GenomicDataProcessor initialized with vocab size {len(self.vocab)}")
    
    def encode_sequence(self, sequence: str) -> List[int]:
        """
        Encode DNA/RNA sequence to integer indices
        
        Args:
            sequence: Nucleotide sequence string
            
        Returns:
            List of integer indices
        """
        encoded = []
        
        # Add start token
        encoded.append(self.vocab.get('<START>', 0))
        
        # Encode each base
        for base in sequence.upper():
            encoded.append(self.vocab.get(base, self.vocab.get('N', 4)))
        
        # Add end token
        encoded.append(self.vocab.get('<END>', 0))
        
        return encoded
    
    def pad_sequence(self, encoded: List[int]) -> List[int]:
        """Pad or truncate sequence to max length"""
        pad_token = self.vocab.get('<PAD>', 5)
        
        if len(encoded) >= self.max_seq_len:
            # Truncate
            return encoded[:self.max_seq_len]
        else:
            # Pad
            padding = [pad_token] * (self.max_seq_len - len(encoded))
            return encoded + padding
    
    def process_sequence(self, sequence: str) -> torch.Tensor:
        """
        Process raw sequence to tensor
        
        Args:
            sequence: Raw nucleotide sequence
            
        Returns:
            Tensor of shape (max_seq_len,)
        """
        encoded = self.encode_sequence(sequence)
        padded = self.pad_sequence(encoded)
        
        return torch.LongTensor(padded)
    
    def process_fasta(self, fasta_path: Union[str, Path]) -> torch.Tensor:
        """
        Process FASTA file to tensor
        
        Args:
            fasta_path: Path to FASTA file
            
        Returns:
            Tensor of encoded sequence
        """
        from Bio import SeqIO
        
        records = list(SeqIO.parse(fasta_path, "fasta"))
        
        if not records:
            raise ValueError(f"No sequences found in {fasta_path}")
        
        # Take first sequence (or implement multi-sequence handling)
        sequence = str(records[0].seq)
        
        return self.process_sequence(sequence)
    
    def process_vcf_variants(
        self,
        vcf_path: Union[str, Path],
        gene_list: Optional[List[str]] = None
    ) -> Dict[str, List[str]]:
        """
        Extract variants from VCF file
        
        Args:
            vcf_path: Path to VCF file
            gene_list: Optional list of genes to filter
            
        Returns:
            Dictionary of gene -> variants
        """
        import cyvcf2
        
        variants = {}
        
        vcf = cyvcf2.VCF(vcf_path)
        
        for variant in vcf:
            gene = variant.INFO.get('GENE', 'Unknown')
            
            if gene_list is None or gene in gene_list:
                if gene not in variants:
                    variants[gene] = []
                
                variants[gene].append({
                    'chrom': variant.CHROM,
                    'pos': variant.POS,
                    'ref': variant.REF,
                    'alt': variant.ALT[0] if variant.ALT else None,
                    'type': variant.var_type
                })
        
        logger.info(f"Extracted variants for {len(variants)} genes from {vcf_path}")
        return variants


class MultiModalDataLoader:
    """
    Unified data loader for multi-modal diagnostic data
    Combines radiology and genomics data for a patient
    """
    
    def __init__(
        self,
        dicom_processor: Optional[DICOMProcessor] = None,
        genomic_processor: Optional[GenomicDataProcessor] = None
    ):
        self.dicom_processor = dicom_processor or DICOMProcessor()
        self.genomic_processor = genomic_processor or GenomicDataProcessor()
    
    def load_patient_data(
        self,
        patient_id: str,
        dicom_dir: Union[str, Path],
        genomic_data: Union[str, Dict]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load complete multi-modal data for a patient
        
        Args:
            patient_id: Patient identifier
            dicom_dir: Directory with DICOM images
            genomic_data: Path to genomic file or raw sequence
            
        Returns:
            Tuple of (radiology_tensor, genomics_tensor)
        """
        # Load radiology data
        radiology_tensor = self.dicom_processor.process_to_tensor(dicom_dir)
        
        # Load genomics data
        if isinstance(genomic_data, (str, Path)):
            if str(genomic_data).endswith(('.fasta', '.fa')):
                genomics_tensor = self.genomic_processor.process_fasta(genomic_data)
            elif str(genomic_data).endswith('.vcf'):
                # For VCF, we'd need to construct a representation
                raise NotImplementedError("VCF processing needs implementation")
            else:
                # Assume raw sequence file
                with open(genomic_data, 'r') as f:
                    sequence = f.read().strip()
                genomics_tensor = self.genomic_processor.process_sequence(sequence)
        else:
            # Raw sequence string
            genomics_tensor = self.genomic_processor.process_sequence(genomic_data)
        
        logger.info(f"Loaded multi-modal data for patient {patient_id}")
        
        return radiology_tensor, genomics_tensor


if __name__ == "__main__":
    # Example usage
    processor = DICOMProcessor(target_size=(64, 64, 64))
    print("DICOM Processor initialized")
    
    genomic_processor = GenomicDataProcessor(max_seq_len=500)
    test_seq = "ACGTACGTACGTNNNN"
    encoded = genomic_processor.process_sequence(test_seq)
    print(f"Encoded sequence shape: {encoded.shape}")
