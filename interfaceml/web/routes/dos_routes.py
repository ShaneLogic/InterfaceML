"""
Routes for TDOS/PDOS visualization.
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from flask import Blueprint, current_app, jsonify, request
from werkzeug.utils import secure_filename

from interfaceml.web.dos import (
    read_cp2k_dos_like,
    extract_pdos_meta,
    gaussian_broaden,
)


bp = Blueprint("dos", __name__)


@bp.route('/api/plot-dos', methods=['POST'])
def plot_dos():
    """Plot TDOS and PDOS curves into a single PNG."""
    data = request.json or {}
    tdos_file = data.get('tdos_file')
    pdos_files = data.get('pdos_files', [])

    if not tdos_file or not Path(tdos_file).exists():
        return jsonify({'error': 'Invalid TDOS file'}), 400

    valid_pdos = [Path(p) for p in pdos_files if p and Path(p).exists()]
    if not valid_pdos:
        return jsonify({'error': 'No valid PDOS files provided'}), 400

    try:
        energy_t, tcols, ef_t, _, header_t = read_cp2k_dos_like(Path(tdos_file))
        if tcols.shape[1] < 1:
            return jsonify({'error': 'TDOS file contains no numeric columns'}), 400

        tdos_source = (data.get('tdos_source') or 'auto').lower()
        has_occupation_col = (
            (tcols.shape[1] >= 2)
            and (re.search(r"\bOccupation\b", header_t, flags=re.IGNORECASE) is not None)
        )

        order = np.argsort(energy_t)
        energy_t = energy_t[order]
        tcols = tcols[order]

        if tdos_source not in {'auto', 'density', 'occupation'}:
            return jsonify({'error': 'Invalid tdos_source (use auto|density|occupation)'}), 400

        if tdos_source == 'auto':
            if has_occupation_col:
                occ_max = float(np.nanmax(tcols[:, 1]))
                dens_max = float(np.nanmax(tcols[:, 0]))
                if occ_max > 1.0 and dens_max < 10.0:
                    tdos_source = 'occupation'
                else:
                    tdos_source = 'density'
            else:
                tdos_source = 'density'

        if tdos_source == 'occupation' and not has_occupation_col:
            return jsonify({'error': 'TDOS occupation column not found in header/data'}), 400

        if tdos_source == 'occupation':
            diffs = np.diff(energy_t)
            delta_e = float(np.median(np.abs(diffs[diffs != 0]))) if diffs.size else 0.0
            if delta_e <= 0:
                return jsonify({'error': 'Cannot infer TDOS energy bin width (ΔE)'}), 400

            occupation_col = tcols[:, 1]
            density_col = tcols[:, 0]

            occupied_mask = (occupation_col > 0.5) & (density_col > 1e-6)
            if np.sum(occupied_mask) > 5:
                dos_from_occ = occupation_col[occupied_mask] / delta_e
                scale_factors = dos_from_occ / density_col[occupied_mask]
                scale_factor = float(np.median(scale_factors))
                tdos = density_col * scale_factor
            else:
                tdos = occupation_col / delta_e
        else:
            tdos = tcols[:, 0]

        pdos_series = []
        ef_candidates = [ef_t]
        for pdos_path in valid_pdos:
            energy_p, cols_p, ef_p, _, header_p = read_cp2k_dos_like(pdos_path)
            ef_candidates.append(ef_p)
            if cols_p.size == 0:
                continue
            label, atom_count = extract_pdos_meta(header_p, pdos_path.stem)
            pdos_series.append({
                'energy': energy_p,
                'dos': np.sum(cols_p, axis=1),
                'label': label,
                'atom_count': atom_count,
            })

        if not pdos_series:
            return jsonify({'error': 'PDOS files contain no DOS columns'}), 400

        ef = next((value for value in ef_candidates if value is not None), None)

        if ef is not None:
            x_t = energy_t - ef
            x_label = "Energy (E − E_F) [eV]"
        else:
            x_t = energy_t
            x_label = "Energy [eV]"

        sigma = float(data.get('sigma', 0.12))
        grid_step = float(data.get('grid_step', 0.02))
        normalize = bool(data.get('normalize', False))
        pdos_scale = data.get('pdos_scale', 'per_atom') or 'per_atom'
        tdos_scale = data.get('tdos_scale', 'none') or 'none'
        tdos_total_atoms = data.get('tdos_total_atoms')
        tdos_total_atoms = int(tdos_total_atoms) if tdos_total_atoms not in (None, "") else None
        x_min = data.get('x_min')
        x_max = data.get('x_max')

        x_min = float(x_min) if x_min not in (None, "") else float(np.min(x_t))
        x_max = float(x_max) if x_max not in (None, "") else float(np.max(x_t))
        if x_max <= x_min:
            return jsonify({'error': 'Invalid energy range (x_max must be > x_min)'}), 400

        grid = np.arange(x_min, x_max + grid_step * 0.5, grid_step)
        tdos_values = tdos
        if tdos_scale == 'per_atom':
            if tdos_total_atoms is None:
                pdos_atom_sum = sum(
                    series.get('atom_count') or 0 for series in pdos_series
                )
                if pdos_atom_sum > 0:
                    tdos_total_atoms = pdos_atom_sum
            if tdos_total_atoms:
                tdos_values = tdos_values / float(tdos_total_atoms)

        tdos_interp = np.interp(grid, x_t, tdos_values, left=0.0, right=0.0)
        if sigma > 0:
            radius = int(max(3, round((4.0 * sigma) / grid_step)))
            kernel_x = np.arange(-radius, radius + 1) * grid_step
            kernel = np.exp(-0.5 * (kernel_x / sigma) ** 2)
            kernel = kernel / np.sum(kernel)
            tdos_smooth = np.convolve(tdos_interp, kernel, mode='same')
        else:
            tdos_smooth = tdos_interp
        if normalize and np.max(tdos_smooth) > 0:
            tdos_smooth = tdos_smooth / np.max(tdos_smooth)

        plt.rcParams.update({
            "font.family": "Arial",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 9 if len(pdos_series) > 5 else 10,
        })

        fig_width = 7.6 if len(pdos_series) <= 5 else 8.5
        fig, ax = plt.subplots(figsize=(fig_width, 4.8))
        ax.plot(grid, tdos_smooth, label="TDOS", linewidth=1.8, color="#111827")

        palette = [
            "#2563eb",
            "#f97316",
            "#16a34a",
            "#7c3aed",
            "#dc2626",
            "#0891b2",
            "#ca8a04",
            "#be185d",
            "#4f46e5",
            "#059669",
            "#9333ea",
            "#ea580c",
        ]
        for idx, series in enumerate(pdos_series):
            x_p = series['energy'] - ef if ef is not None else series['energy']
            pdos_values = series['dos']
            if pdos_scale == 'per_atom' and series.get('atom_count'):
                pdos_values = pdos_values / float(series['atom_count'])
            pdos_smooth = gaussian_broaden(x_p, pdos_values, grid, sigma)
            if pdos_scale == 'match_tdos' and np.max(pdos_smooth) > 0:
                pdos_smooth = pdos_smooth * (np.max(tdos_smooth) / np.max(pdos_smooth))
            if normalize and np.max(pdos_smooth) > 0:
                pdos_smooth = pdos_smooth / np.max(pdos_smooth)
            ax.plot(
                grid,
                pdos_smooth,
                label=series['label'],
                linewidth=1.4,
                color=palette[idx % len(palette)],
                alpha=0.9,
            )

        if ef is not None:
            ax.axvline(0.0, linewidth=0.9, linestyle="--", color="#334155")

        title = data.get('title') or "TDOS + PDOS"
        ax.set_title(title)
        ax.set_xlabel(x_label)
        y_label = "DOS (states/eV)"
        if tdos_scale == 'per_atom':
            y_label = "DOS (states/eV/atom)"
        ax.set_ylabel(y_label)
        ax.legend(loc="best", frameon=False)
        ax.grid(alpha=0.25, linestyle="--", linewidth=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlim(x_min, x_max)
        fig.tight_layout()

        output_name = data.get('output_name') or "dos_tdos_pdos_overlay"
        output_filename = secure_filename(f"{output_name}.png")
        output_path = Path(current_app.config['UPLOAD_FOLDER']) / output_filename
        fig.savefig(output_path, dpi=220)
        plt.close(fig)

        tdos_peak = float(np.max(tdos_smooth)) if tdos_smooth.size else 0.0
        pdos_peaks = {
            series['label']: float(np.max(gaussian_broaden(
                series['energy'] - ef if ef else series['energy'],
                series['dos'] / (float(series['atom_count']) if pdos_scale == 'per_atom' and series.get('atom_count') else 1.0),
                grid, sigma
            ))) for series in pdos_series
        }

        return jsonify({
            'status': 'success',
            'ef': ef,
            'tdos_source_used': tdos_source,
            'tdos_total_atoms_used': tdos_total_atoms,
            'tdos_peak': round(tdos_peak, 2),
            'pdos_peaks': {k: round(v, 2) for k, v in pdos_peaks.items()},
            'image_file': str(output_path),
            'image_url': f"/api/download/{output_filename}",
            'download_url': f"/api/download/{output_filename}",
        })

    except Exception as exc:
        return jsonify({'error': str(exc)}), 500
