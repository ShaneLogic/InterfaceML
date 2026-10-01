# InterfaceML

Python tools for constructing atomistic interfaces, preparing constrained DFT inputs, and exploring equivariant generative models for fullerene/perovskite structures. A Flask application exposes structure upload, interface building, layer selection, DOS plotting, and CP2K input preparation.

The repository combines a reusable geometry core with research scripts and model checkpoints. Generated structures are candidates for inspection and relaxation; geometric matching or a learned generator does not establish energetic stability.

## Workflows

| Workflow | What the implementation does |
| --- | --- |
| Coherent crystal interfaces | Uses pymatgen ZSL matching and CoherentInterfaceBuilder to enumerate terminations and in-plane supercells |
| Molecular adsorption | Builds a slab, selects a termination, sizes its in-plane supercell, rotates/places an adsorbate, and adds a separation and vacuum |
| Layer constraints | Detects layers along the interface normal, unwraps the periodic gap, and writes selective dynamics or CP2K atom indices |
| DOS analysis | Reads CP2K-like TDOS/PDOS files, aligns energies, broadens curves, and exports plots |
| AI generation | Loads optional EGNN/DDPM or PaiNN/flow-matching fullerene models and exposes generation/evaluation endpoints |
| DFT preparation | Produces CP2K geometry-optimization inputs, structure files, and Slurm scripts |
| Iterative data preparation | Generates candidates, builds interfaces, prepares DFT jobs, parses completed calculations, and assembles retraining data |

## Installation

Use a source checkout so the research modules and legacy scripts remain available. Python 3.11 or newer is a practical starting point; the package declares Python >=3.9, but compatibility also depends on the installed scientific dependencies.

~~~bash
git clone https://github.com/ShaneLogic/InterfaceML.git
cd InterfaceML
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install pyyaml
~~~

Core dependencies include NumPy, pymatgen, Flask, Flask-CORS, Werkzeug, and Matplotlib. PyYAML is needed for the pipeline/configuration paths and is currently listed under the AI extra rather than the core dependency set.

For model development or distributed task processing:

~~~bash
python -m pip install -e ".[ai,dev]"
# Optional: requires a separately running Redis service
python -m pip install -e ".[async]"
~~~

Choose a PyTorch build suitable for your hardware. Installing the AI extra does not provide a validated model or ensure checkpoint compatibility.

## Web Application

Start the local application from the repository root:

~~~bash
python -m flask --app interfaceml.web.app:app run --host 127.0.0.1 --port 5000
~~~

Open <http://127.0.0.1:5000/app>. The documentation view is at /docs.

The packaged interfaceml-web command also exists, but its implementation enables debug mode and defaults to 0.0.0.0. The application has permissive CORS and file-processing endpoints; the localhost command above is the intended development setup, not a production deployment configuration.

Uploads are limited to 16 MB. Supported extensions include CIF, POSCAR/VASP, XYZ, DOS, and PDOS. Set INTERFACEML_UPLOAD_FOLDER to retain files in a chosen location; otherwise a temporary directory is created.

Useful endpoints:

| Endpoint | Purpose |
| --- | --- |
| GET /api/health | Core and AI availability |
| POST /api/upload | Upload a structure or DOS file |
| POST /api/list-interface-terminations | Enumerate coherent-interface terminations |
| POST /api/build-interface | Construct a coherent interface |
| POST /api/plot-dos | Generate a TDOS/PDOS plot |
| GET /api/ai/models | Model availability and load errors |
| POST /api/ai/generate | Generate fullerene candidates |
| POST /api/ai/generate-interface | Generate/place an adsorbate candidate |
| POST /api/pipeline/build-and-prep | Build an adsorbate interface and package CP2K inputs |
| POST /api/pipeline/dft-prep | Prepare CP2K inputs from an existing structure |

The pipeline endpoints return downloadable archives. Their download registry is process-local, so links do not persist across server restarts. Preparing an archive does not submit a calculation.

## Methods and Usage

### Coherent Crystal Interfaces

The modular builder searches two-dimensional lattice matches through pymatgen. It enumerates selected termination pairs, attaches strain/area/atom-count metadata, and ranks candidates by von Mises strain, then atom count.

In the Web/core workflow, the film is strained to the substrate. Bidirectional ZSL search changes the match search, not the strain-sharing physics. Registry shifts and twists alter a candidate's geometry; they do not find a new commensurate cell or relax atomic forces.

The legacy script exposes a separate strain_target option. From the source checkout, inspect its actual arguments and run it directly:

~~~bash
python build_heterojunctions/interface_builder.py --help

python build_heterojunctions/interface_builder.py \
  --a /absolute/path/to/material_a.cif \
  --b /absolute/path/to/material_b.cif \
  --miller_a 0,0,1 --miller_b 0,0,1 \
  --slab_thickness_a 20 --slab_thickness_b 12 \
  --vacuum 20 --sep 3.2 \
  --tol 0.03 --max_area 800 \
  --strain_target A --use_builder_interface --max_atoms 400
~~~

Replace the input paths with your structures. Lengths are in angstroms, areas in angstrom^2, and tol is a fraction. Outputs are written under structures/heterojunctions, including combined and separately strained bottom/top POSCAR files.

**Current CLI limitation:** interfaceml-build imports a main function that the legacy interface_builder.py does not define. Use the direct script above. The flags --adsorbate_mode, --base, --adsorbate, --distance, and --output are not arguments of that legacy script; use the Python API or Web workflow for adsorption.

### Adsorbate Placement

The Python API returns the substrate slab, adsorbate layer, combined structure, and supercell/termination metadata:

~~~python
from pymatgen.io.vasp import Poscar
from interfaceml.core.io import load_structure
from interfaceml.core.adsorbate import build_adsorbate_interface

base = load_structure("perovskite.cif")
adsorbate = load_structure("fullerene.xyz")

bottom, top, combined, choice = build_adsorbate_interface(
    base,
    adsorbate,
    miller=(0, 0, 1),
    slab_thickness=18.0,
    vacuum=20.0,
    separation=3.2,
    buffer=10.0,
    xy_frac=(0.5, 0.5),
)
Poscar(combined, sort_structure=False).write_file("interface.vasp")
print(choice)
~~~

Supercell sizing estimates the molecular diameter and adds a buffer, subject to a maximum replication limit. It is a placement heuristic, not a guarantee of adequate image separation for every skewed lattice. The molecular-unwrapping step uses a distance-cutoff connectivity graph to repair molecules split across the c boundary.

XYZ input uses the last readable frame. CP2K Tv_1/Tv_2/Tv_3 cell metadata is supported; without cell information, the loader constructs a padded box. Review that inferred cell before periodic calculations.

### Selective Dynamics and Layer Splitting

Fix the bottom two height-clustered layers while allowing the remaining atoms to relax:

~~~bash
python build_heterojunctions/fix_interface_layers.py interface.vasp \
  --by_z_layers --n_fix_layers 2 --debug_layers \
  -o interface_fixed.vasp
~~~

Use --print_only to inspect layer sizes and fixed-atom indices without writing a POSCAR. Printed indices default to one-based CP2K numbering. Whole-molecule inclusion is enabled by default for the relevant organic-element selection.

The geometry core projects atoms onto the normal derived from the in-plane lattice vectors, cuts at the largest periodic gap, and identifies layer boundaries. The smart splitter additionally uses composition and cross-boundary contact heuristics. These methods require inspection for reconstructed, intermixed, or weakly separated layers.

See [layer splitting](docs/LAYER_SPLITTING.md) and [the smart splitting algorithm](docs/SMART_SPLITTING_ALGORITHM.md).

### AI Models

The Web registry supports:

| Model key | Architecture | Sampling objective |
| --- | --- | --- |
| egnn | Scalar message passing with relative-vector coordinate updates | DDPM noise prediction |
| painn_fm | Scalar/vector PaiNN features with radial filters and time conditioning | Conditional flow matching |

The flow-matching implementation interpolates clean coordinates at t=0 to noise at t=1, predicts noise-minus-data velocity, and integrates backward using Euler steps. Training includes configurable geometric losses; these are priors, not DFT energies.

Checkpoint configuration:

| Environment variable | Use |
| --- | --- |
| INTERFACEML_FULLERENE_PATH | Fullerene source-module directory |
| INTERFACEML_FULLERENE_CHECKPOINT | EGNN checkpoint override |
| INTERFACEML_PAINN_FM_CHECKPOINT | PaiNN/flow-matching checkpoint override |
| INTERFACEML_LOCAL_GNN_CHECKPOINT | Optional local interface refiner |

The tracked tree contains archived experiments and a PaiNN/flow-matching checkpoint. The default EGNN path fullerene_e3gen/checkpoints/best_model.pt is not present in the reviewed revision. Consult /api/ai/models for the actual available backends; archived filenames do not establish model quality.

The model directories are source-tree research modules and are not all included in the installed wheel. Use the checkout for training and generation. The [fullerene implementation](fullerene_e3gen), [perovskite implementation](perovskite_e3gen), and earlier [diffusion proof of concept](fullerene_diffusion_poc) contain separate experiments.

### DFT and Retraining Data

Review [active_learning/config.yaml](active_learning/config.yaml), especially checkpoints, compositions, CP2K basis/potential choices, convergence settings, and Slurm resources.

~~~bash
# Generate candidates, build interfaces, and write DFT input files
python -m active_learning.pipeline \
  --config active_learning/config.yaml --round 0

# After you have separately reviewed, submitted, and completed the DFT jobs
python -m active_learning.pipeline \
  --config active_learning/config.yaml --round 0 --phase parse

# Assemble accepted structures and a retraining configuration
python -m active_learning.pipeline \
  --config active_learning/config.yaml --round 0 --phase retrain
~~~

The default run stops after generate, build, and dft_prep. It writes per-round state and CP2K/Slurm artifacts under active_learning_runs. It does not execute CP2K or sbatch. The retrain phase prepares data and configuration; it does not launch model optimization.

Parsing extracts total energies, force information, convergence markers, and optimized geometries. Binding-energy filtering only applies when binding_energy_eV is present; the current parser does not automatically compute isolated-component reference energies. Missing force/binding fields can bypass those filters, so accepted.json is not sufficient evidence of physical validity.

## Repository Map

- [interfaceml/core](interfaceml/core): structure I/O, coherent/adsorbate construction, splitting, and layer constraints.
- [interfaceml/web](interfaceml/web): Flask application, routes, templates, browser code, DOS helpers, and optional tasks.
- [interfaceml/ai](interfaceml/ai): local graph refinement and interface-generation helpers.
- [build_heterojunctions](build_heterojunctions): direct scripts and legacy workflow implementation.
- [active_learning](active_learning): staged generation, DFT preparation, result parsing, and retraining-data preparation.
- [tests](tests): geometry, I/O, coherent-interface identity, Web archive, and model checks.
- [docs](docs): mathematical and workflow notes.

Some older subproject guides describe historical configurations. Function signatures, current configuration files, and the commands above take precedence over obsolete examples.

## Verification

Install the development dependencies and run focused geometry/Web checks:

~~~bash
python -m pip install -e ".[dev]"
python -m pytest \
  tests/test_core_io.py tests/test_core_adsorbate.py \
  tests/test_coherent_interfaces.py tests/test_pipeline_routes.py \
  tests/test_web_health.py
~~~

During the 2026-10-01 documentation review, these checks produced **38 passed and 6 skipped** in a source checkout using Python 3.13. The skips include unavailable fixture-dependent cases. Direct builder/fix script help succeeded, and the interfaceml-build import failure was reproduced.

No model-training run, checkpoint-quality benchmark, CP2K calculation, or cluster submission was performed as part of that review.

## Contact

Xuan-Yan Chen

Email: [xchen565@connect.hkust-gz.edu.cn](mailto:xchen565@connect.hkust-gz.edu.cn)

Bugs and feature requests: [GitHub Issues](https://github.com/ShaneLogic/InterfaceML/issues)

## License

The package metadata declares MIT licensing. A standalone root LICENSE file is not present in the reviewed tree; consult the author for complete licensing terms and the provenance of individual datasets/checkpoints.
