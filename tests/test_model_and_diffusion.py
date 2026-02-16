"""Tests for fullerene_e3gen model and diffusion utilities.

These tests verify the core ML components without requiring GPU or
trained checkpoints — they test architectural invariants (shapes,
equivariance properties, schedule monotonicity).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

# Add fullerene_e3gen to sys.path so we can import directly
_POC_DIR = Path(__file__).resolve().parents[1] / "fullerene_e3gen"
if str(_POC_DIR) not in sys.path:
    sys.path.insert(0, str(_POC_DIR))

from model import FullereneDiffusionModel, GlobalAttentionPool, bond_length_loss, sphericity_loss, _scatter_softmax
from diffusion_utils import DiffusionScheduler


# ---------------------------------------------------------------------------
# Model architecture tests
# ---------------------------------------------------------------------------

class TestFullereneDiffusionModel:
    """Test EGNN model forward pass shapes and properties."""

    @pytest.fixture()
    def small_model(self):
        return FullereneDiffusionModel(
            hidden_dim=16,
            num_layers=2,
            edge_dim=0,
            C_embed_dim=8,
            time_embed_dim=16,
        )

    @pytest.fixture()
    def dummy_graph(self):
        """A small 10-node 'fullerene' with random 3-regular edges."""
        N = 10
        pos = torch.randn(N, 3)
        # Simple cycle + skip edges to approximate 3-regular
        src = list(range(N)) + list(range(N))
        dst = [(i + 1) % N for i in range(N)] + [(i + 2) % N for i in range(N)]
        edge_index = torch.tensor([src + dst, dst + src], dtype=torch.long)
        t = torch.tensor([42], dtype=torch.long)
        C = torch.tensor([N], dtype=torch.long)
        batch = torch.zeros(N, dtype=torch.long)
        return pos, edge_index, t, C, batch

    def test_output_shape(self, small_model, dummy_graph):
        pos, edge_index, t, C, batch = dummy_graph
        out = small_model(pos, edge_index, t, C, batch)
        assert out.shape == pos.shape  # [N, 3]

    def test_output_is_finite(self, small_model, dummy_graph):
        pos, edge_index, t, C, batch = dummy_graph
        out = small_model(pos, edge_index, t, C, batch)
        assert torch.isfinite(out).all()

    def test_batch_dimension(self, small_model):
        """Two graphs batched together."""
        N1, N2 = 8, 12
        pos = torch.randn(N1 + N2, 3)

        # Build separate edge indices and combine
        e1_src = list(range(N1)) + list(range(N1))
        e1_dst = [(i + 1) % N1 for i in range(N1)] + [(i + 2) % N1 for i in range(N1)]
        e2_src = [i + N1 for i in range(N2)] + [i + N1 for i in range(N2)]
        e2_dst = [(i + 1) % N2 + N1 for i in range(N2)] + [(i + 2) % N2 + N1 for i in range(N2)]

        src = e1_src + e1_dst + e2_src + e2_dst
        dst = e1_dst + e1_src + e2_dst + e2_src
        edge_index = torch.tensor([src, dst], dtype=torch.long)

        t = torch.tensor([10, 20], dtype=torch.long)
        C = torch.tensor([N1, N2], dtype=torch.long)
        batch = torch.cat([torch.zeros(N1, dtype=torch.long), torch.ones(N2, dtype=torch.long)])

        out = small_model(pos, edge_index, t, C, batch)
        assert out.shape == (N1 + N2, 3)

    def test_different_timesteps_give_different_outputs(self, small_model, dummy_graph):
        pos, edge_index, _, C, batch = dummy_graph
        torch.manual_seed(0)
        out_t0 = small_model(pos, edge_index, torch.tensor([0]), C, batch)
        out_t500 = small_model(pos, edge_index, torch.tensor([500]), C, batch)
        # Different timesteps should produce different noise predictions.
        # With random initialization, outputs may be very similar, so use a
        # loose tolerance and check relative difference instead.
        diff = (out_t0 - out_t500).abs().max().item()
        assert diff > 0 or not torch.equal(out_t0, out_t500), \
            "Identical outputs for different timesteps"


# ---------------------------------------------------------------------------
# Scatter softmax tests
# ---------------------------------------------------------------------------

class TestScatterSoftmax:
    """Test the per-node scatter softmax implementation."""

    def test_single_node(self):
        w = torch.tensor([1.0, 2.0, 3.0])
        # All edges target node 0
        col = torch.tensor([0, 0, 0])
        result = _scatter_softmax(w, col, num_nodes=1)
        expected = torch.softmax(w, dim=0)
        torch.testing.assert_close(result, expected)

    def test_two_nodes(self):
        w = torch.tensor([1.0, 2.0, 3.0, 4.0])
        col = torch.tensor([0, 0, 1, 1])
        result = _scatter_softmax(w, col, num_nodes=2)

        # Node 0: softmax([1, 2])
        expected_0 = torch.softmax(torch.tensor([1.0, 2.0]), dim=0)
        # Node 1: softmax([3, 4])
        expected_1 = torch.softmax(torch.tensor([3.0, 4.0]), dim=0)

        torch.testing.assert_close(result[:2], expected_0)
        torch.testing.assert_close(result[2:], expected_1)

    def test_sums_to_one_per_node(self):
        w = torch.randn(100)
        col = torch.randint(0, 5, (100,))
        result = _scatter_softmax(w, col, num_nodes=5)

        for node in range(5):
            mask = col == node
            if mask.any():
                assert result[mask].sum().item() == pytest.approx(1.0, abs=1e-5)


# ---------------------------------------------------------------------------
# Loss function tests
# ---------------------------------------------------------------------------

class TestBondLengthLoss:
    """Test bond_length_loss function."""

    def test_perfect_bonds_zero_loss(self):
        # Square with side length 1.42
        pos = torch.tensor([[0.0, 0.0, 0.0], [1.42, 0.0, 0.0],
                            [1.42, 1.42, 0.0], [0.0, 1.42, 0.0]])
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]])
        batch = torch.zeros(4, dtype=torch.long)
        loss = bond_length_loss(pos, edge_index, target_length=1.42)
        assert loss.item() == pytest.approx(0.0, abs=1e-4)

    def test_wrong_bonds_nonzero_loss(self):
        pos = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        edge_index = torch.tensor([[0], [1]])
        batch = torch.zeros(2, dtype=torch.long)
        loss = bond_length_loss(pos, edge_index, target_length=1.42)
        assert loss.item() > 0.0


class TestSphericityLoss:
    """Test sphericity_loss function."""

    def test_sphere_low_loss(self):
        # Points uniformly on a unit sphere
        torch.manual_seed(42)
        N = 60
        pos = torch.randn(N, 3)
        pos = pos / pos.norm(dim=1, keepdim=True)  # Project onto unit sphere
        batch = torch.zeros(N, dtype=torch.long)
        loss = sphericity_loss(pos, batch, target_radius=1.0)
        assert loss.item() < 0.15  # Should be small for a near-perfect sphere

    def test_non_sphere_higher_loss(self):
        # Elongated structure
        pos = torch.zeros(60, 3)
        pos[:, 0] = torch.linspace(-5, 5, 60)  # Line, not sphere
        batch = torch.zeros(60, dtype=torch.long)
        loss = sphericity_loss(pos, batch, target_radius=1.0)
        assert loss.item() > 0.5  # Should be large


# ---------------------------------------------------------------------------
# Diffusion scheduler tests
# ---------------------------------------------------------------------------

class TestDiffusionScheduler:
    """Test DiffusionScheduler properties."""

    @pytest.fixture()
    def scheduler(self):
        return DiffusionScheduler(num_steps=100, beta_schedule="cosine")

    def test_alphas_cumprod_decreasing(self, scheduler):
        assert (scheduler.alphas_cumprod[:-1] >= scheduler.alphas_cumprod[1:]).all()

    def test_alphas_cumprod_range(self, scheduler):
        assert scheduler.alphas_cumprod[0] > 0.9  # Almost no noise at t=0
        assert scheduler.alphas_cumprod[-1] < 0.1  # Almost pure noise at t=T

    def test_sqrt_precomputations(self, scheduler):
        expected = torch.sqrt(scheduler.alphas_cumprod)
        torch.testing.assert_close(scheduler.sqrt_alphas_cumprod, expected)

    def test_posterior_variance_nonneg(self, scheduler):
        assert (scheduler.posterior_variance >= 0).all()

    def test_linear_schedule(self):
        sched = DiffusionScheduler(num_steps=50, beta_schedule="linear")
        assert sched.alphas_cumprod.shape == (50,)
        assert sched.alphas_cumprod[0] > sched.alphas_cumprod[-1]


# ---------------------------------------------------------------------------
# Global attention tests
# ---------------------------------------------------------------------------

class TestGlobalAttentionPool:
    """Test the GlobalAttentionPool module."""

    def test_output_shape(self):
        pool = GlobalAttentionPool(hidden_dim=16)
        h = torch.randn(10, 16)
        batch = torch.zeros(10, dtype=torch.long)
        out = pool(h, batch)
        assert out.shape == h.shape

    def test_multi_batch(self):
        pool = GlobalAttentionPool(hidden_dim=16)
        h = torch.randn(20, 16)
        batch = torch.cat([torch.zeros(8, dtype=torch.long), torch.ones(12, dtype=torch.long)])
        out = pool(h, batch)
        assert out.shape == (20, 16)


class TestGlobalAttentionModel:
    """Test FullereneDiffusionModel with global attention enabled."""

    @pytest.fixture()
    def model_with_ga(self):
        return FullereneDiffusionModel(
            hidden_dim=16,
            num_layers=4,
            edge_dim=0,
            C_embed_dim=8,
            time_embed_dim=16,
            use_global_attention=True,
            global_attention_heads=4,
            use_hierarchical=False,
        )

    @pytest.fixture()
    def dummy_graph(self):
        N = 10
        pos = torch.randn(N, 3)
        src = list(range(N)) + list(range(N))
        dst = [(i + 1) % N for i in range(N)] + [(i + 2) % N for i in range(N)]
        edge_index = torch.tensor([src + dst, dst + src], dtype=torch.long)
        t = torch.tensor([42], dtype=torch.long)
        C = torch.tensor([N], dtype=torch.long)
        batch = torch.zeros(N, dtype=torch.long)
        return pos, edge_index, t, C, batch

    def test_output_shape(self, model_with_ga, dummy_graph):
        pos, edge_index, t, C, batch = dummy_graph
        out = model_with_ga(pos, edge_index, t, C, batch)
        assert out.shape == pos.shape

    def test_rotation_equivariance(self, model_with_ga, dummy_graph):
        """Rotating input pos should rotate output by same rotation."""
        pos, edge_index, t, C, batch = dummy_graph
        torch.manual_seed(123)

        # Random rotation matrix
        theta = torch.tensor(1.23)
        R = torch.tensor([
            [torch.cos(theta), -torch.sin(theta), 0],
            [torch.sin(theta),  torch.cos(theta), 0],
            [0,                 0,                 1],
        ], dtype=torch.float32)

        model_with_ga.eval()
        with torch.no_grad():
            out_orig = model_with_ga(pos, edge_index, t, C, batch)
            out_rotated_input = model_with_ga(pos @ R.T, edge_index, t, C, batch)

        # f(Rx) should equal R f(x)
        expected = out_orig @ R.T
        torch.testing.assert_close(out_rotated_input, expected, atol=1e-4, rtol=1e-4)
