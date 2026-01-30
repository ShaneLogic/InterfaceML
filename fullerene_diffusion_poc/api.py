"""
API Interface for Fullerene Diffusion Model

Provides a clean, RESTful-style API for training, generation, and evaluation.
Designed for integration with frontend applications.

Author: InterfaceML Project
Date: 2026-01-30
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import traceback

import torch
import numpy as np
from torch_geometric.data import Data

from model import FullereneDiffusionModel
from diffusion_utils import DiffusionScheduler
from dataset import FullereneDataset
from generate import create_template_graph, sample_structure
from evaluate import (
    parse_xyz_file,
    compute_bond_lengths,
    compute_sphericity,
    compute_rdf,
    compute_adf,
    compute_structure_energy,
    validate_connectivity
)


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FullereneAPI:
    """
    Main API class for fullerene diffusion model.
    
    Provides methods for model loading, structure generation, and evaluation.
    Thread-safe and designed for production use.
    """
    
    def __init__(self, checkpoint_path: Optional[str] = None, device: Optional[str] = None):
        """
        Initialize the API.
        
        Args:
            checkpoint_path: Path to trained model checkpoint
            device: Device to use ('cuda', 'cpu', or None for auto-detection)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.scheduler = None
        self.checkpoint_info = None
        
        if checkpoint_path:
            self.load_model(checkpoint_path)
        
        logger.info(f"FullereneAPI initialized on device: {self.device}")
    
    def load_model(self, checkpoint_path: str) -> Dict:
        """
        Load trained model from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Dictionary with model info (epoch, loss, etc.)
            
        Raises:
            FileNotFoundError: If checkpoint doesn't exist
            RuntimeError: If model loading fails
        """
        try:
            checkpoint_path = Path(checkpoint_path)
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            
            logger.info(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Initialize model
            model_config = checkpoint.get('config', {
                'hidden_dim': 64,
                'num_layers': 3,
                'edge_dim': 0
            })
            
            self.model = FullereneDiffusionModel(
                hidden_dim=model_config['hidden_dim'],
                num_layers=model_config['num_layers'],
                edge_dim=model_config.get('edge_dim', 0)
            ).to(self.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            # Initialize diffusion scheduler
            self.scheduler = DiffusionScheduler(
                num_steps=checkpoint.get('num_steps', 1000),
                beta_schedule=checkpoint.get('beta_schedule', 'cosine')
            )
            
            # Store checkpoint info
            self.checkpoint_info = {
                'epoch': checkpoint.get('epoch', 0),
                'best_val_loss': checkpoint.get('best_val_loss', float('inf')),
                'config': model_config
            }
            
            logger.info(f"Model loaded successfully (epoch {self.checkpoint_info['epoch']})")
            return self.checkpoint_info
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise RuntimeError(f"Model loading failed: {str(e)}")
    
    def generate(
        self,
        num_carbon: int,
        num_samples: int = 1,
        ddim: bool = False,
        return_trajectory: bool = False
    ) -> List[Dict]:
        """
        Generate fullerene structures.
        
        Args:
            num_carbon: Number of carbon atoms (must be even, >= 20)
            num_samples: Number of structures to generate
            ddim: Use DDIM sampling (faster but less diverse)
            return_trajectory: Return full denoising trajectory
            
        Returns:
            List of dictionaries, each containing:
                - positions: numpy array [N, 3]
                - edges: list of (src, dst) tuples
                - metadata: generation parameters
                
        Raises:
            ValueError: If parameters are invalid
            RuntimeError: If model not loaded or generation fails
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        if num_carbon < 20 or num_carbon > 720 or num_carbon % 2 != 0:
            raise ValueError(f"num_carbon must be even and in [20, 720], got {num_carbon}")
        
        if num_samples < 1:
            raise ValueError(f"num_samples must be >= 1, got {num_samples}")
        
        logger.info(f"Generating {num_samples} C{num_carbon} structures (ddim={ddim})")
        
        results = []
        
        try:
            for i in range(num_samples):
                # Create template graph
                template = create_template_graph(num_carbon, self.device)
                
                # Sample structure
                pos_final, trajectory = sample_structure(
                    self.model,
                    self.scheduler,
                    template,
                    ddim=ddim,
                    return_trajectory=return_trajectory
                )
                
                # Convert to numpy
                positions = pos_final.cpu().numpy()
                edges = template.edge_index.t().cpu().numpy().tolist()
                
                result = {
                    'positions': positions,
                    'edges': edges,
                    'metadata': {
                        'num_carbon': num_carbon,
                        'sample_id': i,
                        'ddim': ddim,
                        'device': self.device
                    }
                }
                
                if return_trajectory:
                    result['trajectory'] = [t.cpu().numpy() for t in trajectory]
                
                results.append(result)
                
            logger.info(f"Generated {num_samples} structures successfully")
            return results
            
        except Exception as e:
            logger.error(f"Generation failed: {str(e)}\n{traceback.format_exc()}")
            raise RuntimeError(f"Generation failed: {str(e)}")
    
    def evaluate(
        self,
        positions: np.ndarray,
        edges: List[Tuple[int, int]],
        compute_advanced_metrics: bool = True
    ) -> Dict:
        """
        Evaluate a single structure.
        
        Args:
            positions: Atomic positions [N, 3]
            edges: Bond connectivity as list of (src, dst) tuples
            compute_advanced_metrics: Include RDF, ADF, energy
            
        Returns:
            Dictionary with evaluation metrics
        """
        try:
            metrics = {}
            
            # Basic metrics
            bond_lengths = compute_bond_lengths(positions, edges)
            if len(bond_lengths) > 0:
                metrics['bond_length'] = {
                    'mean': float(bond_lengths.mean()),
                    'std': float(bond_lengths.std()),
                    'min': float(bond_lengths.min()),
                    'max': float(bond_lengths.max())
                }
            
            # Sphericity metrics
            sphere_metrics = compute_sphericity(positions)
            metrics['sphericity'] = {
                'radius_gyration': float(sphere_metrics['radius_gyration']),
                'asphericity': float(sphere_metrics['asphericity']),
                'acylindricity': float(sphere_metrics['acylindricity']),
                'shape_anisotropy': float(sphere_metrics['shape_anisotropy'])
            }
            
            # Connectivity
            conn_metrics = validate_connectivity(edges, len(positions))
            metrics['connectivity'] = {
                'is_valid': bool(conn_metrics['all_degree_3']),
                'mean_degree': float(conn_metrics['mean_degree']),
                'min_degree': int(conn_metrics['min_degree']),
                'max_degree': int(conn_metrics['max_degree'])
            }
            
            # Advanced metrics
            if compute_advanced_metrics:
                # Energy
                energy_metrics = compute_structure_energy(positions, edges)
                metrics['energy'] = {
                    'total': float(energy_metrics['total_energy']),
                    'mean_bond': float(energy_metrics['mean_bond_energy']),
                    'max_bond': float(energy_metrics['max_bond_energy'])
                }
                
                # RDF
                r, g_r = compute_rdf(positions)
                metrics['rdf'] = {
                    'r': r.tolist(),
                    'g_r': g_r.tolist()
                }
                
                # ADF
                angles, adf = compute_adf(positions, edges)
                metrics['adf'] = {
                    'angles': angles.tolist(),
                    'adf': adf.tolist()
                }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Evaluation failed: {str(e)}")
            raise RuntimeError(f"Evaluation failed: {str(e)}")
    
    def save_structure(
        self,
        positions: np.ndarray,
        edges: List[Tuple[int, int]],
        output_path: str,
        format: str = 'xyz'
    ) -> None:
        """
        Save structure to file.
        
        Args:
            positions: Atomic positions [N, 3]
            edges: Bond connectivity
            output_path: Output file path
            format: File format ('xyz', 'json')
            
        Raises:
            ValueError: If format is unsupported
        """
        output_path = Path(output_path)
        
        if format == 'xyz':
            self._save_xyz(positions, edges, output_path)
        elif format == 'json':
            self._save_json(positions, edges, output_path)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Structure saved to {output_path}")
    
    def _save_xyz(self, positions: np.ndarray, edges: List[Tuple[int, int]], path: Path):
        """Save in extended XYZ format."""
        num_atoms = len(positions)
        
        # Build adjacency list
        neighbors = {i: [] for i in range(num_atoms)}
        for src, dst in edges:
            if dst not in neighbors[src]:
                neighbors[src].append(dst)
        
        with open(path, 'w') as f:
            f.write(f"{num_atoms}\n")
            f.write(f"Generated fullerene structure\n")
            
            for i, pos in enumerate(positions):
                x, y, z = pos
                nb = neighbors.get(i, [])
                nb_str = ' '.join(str(n) for n in nb[:3])  # First 3 neighbors
                
                # Pad with -1 if less than 3 neighbors
                while len(nb_str.split()) < 3:
                    nb_str += ' -1'
                
                f.write(f"C {x:.6f} {y:.6f} {z:.6f} {i} {nb_str}\n")
    
    def _save_json(self, positions: np.ndarray, edges: List[Tuple[int, int]], path: Path):
        """Save in JSON format."""
        data = {
            'num_atoms': len(positions),
            'positions': positions.tolist(),
            'edges': edges,
            'element': 'C'
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def get_model_info(self) -> Dict:
        """
        Get information about loaded model.
        
        Returns:
            Dictionary with model configuration and checkpoint info
        """
        if self.model is None:
            return {'status': 'no_model_loaded'}
        
        return {
            'status': 'loaded',
            'device': self.device,
            'checkpoint_info': self.checkpoint_info,
            'num_parameters': sum(p.numel() for p in self.model.parameters())
        }


# Convenience functions for simple use cases

def quick_generate(
    checkpoint_path: str,
    num_carbon: int,
    num_samples: int = 1,
    output_dir: Optional[str] = None
) -> List[Dict]:
    """
    Quick generation without persistent API object.
    
    Args:
        checkpoint_path: Path to model checkpoint
        num_carbon: Number of carbon atoms
        num_samples: Number of structures to generate
        output_dir: Optional directory to save structures
        
    Returns:
        List of generated structures
    """
    api = FullereneAPI(checkpoint_path)
    results = api.generate(num_carbon, num_samples)
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for i, result in enumerate(results):
            output_path = output_dir / f"C{num_carbon}_sample_{i:03d}.xyz"
            api.save_structure(
                result['positions'],
                result['edges'],
                str(output_path)
            )
    
    return results


def quick_evaluate(structure_path: str) -> Dict:
    """
    Quick evaluation of a structure file.
    
    Args:
        structure_path: Path to XYZ file
        
    Returns:
        Evaluation metrics
    """
    positions, edges = parse_xyz_file(Path(structure_path))
    api = FullereneAPI()
    return api.evaluate(positions, edges)


if __name__ == '__main__':
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Fullerene API Test')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--num_carbon', type=int, default=60)
    parser.add_argument('--num_samples', type=int, default=1)
    parser.add_argument('--output_dir', type=str, default='api_test_output')
    args = parser.parse_args()
    
    # Test API
    results = quick_generate(
        args.checkpoint,
        args.num_carbon,
        args.num_samples,
        args.output_dir
    )
    
    print(f"\n✓ Generated {len(results)} structures")
    
    # Evaluate first structure
    if results:
        metrics = quick_evaluate(f"{args.output_dir}/C{args.num_carbon}_sample_000.xyz")
        print("\nEvaluation metrics:")
        print(json.dumps(metrics, indent=2))
