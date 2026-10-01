# Research Containers

This environment targets Linux ARM64 on Apple Silicon, with CPU PyTorch.
The application and JupyterLab share one image and a persistent research directory.
CP2K input generation is included; a CP2K executable and cluster access are separate.

## Start

Set these optional values in a local `.env` file:

```dotenv
INTERFACEML_PORT=8001
JUPYTER_PORT=8888
INTERFACEML_WORKSPACE=./.research
INTERFACEML_UID=1000
```

For a host-owned research directory, use its absolute path and set
`INTERFACEML_UID` to the numeric result of `id -u`. `.env` is excluded from Git
and from the Docker build context. Existing containers from other projects are
not used by this Compose project.

```bash
docker compose --profile research up --build -d
docker compose --profile research ps
```

- InterfaceML: <http://localhost:8001/app>
- JupyterLab: <http://localhost:8888/lab>
- Health: <http://localhost:8001/api/health>

JupyterLab keeps token authentication enabled. Get its local access URL with:

```bash
docker compose exec research jupyter server list
```

The host research directory is mounted at `/workspace` in both services.
Notebooks, uploaded structures, and calculation outputs stored there survive
container recreation. The application upload directory is `/workspace/uploads`.
Download links for DFT archives are currently session-scoped; the ZIP files remain
in the research directory after a web-process restart.

## Work

```bash
docker compose exec research python
docker compose exec research bash
docker compose exec research python -m pip check
docker compose exec -w /app research python -m pytest tests/ -q
```

The environment includes NumPy, SciPy, pandas, pymatgen, PyTorch, PyTorch Geometric,
Matplotlib, Flask, JupyterLab, pytest, and Ruff. The container imports the same
application sources used by the web service through `/app`.

Optional trained models can be placed in the research directory:

```text
models/egnn/best_model.pt
models/painn_fm/best_model.pt
```

Restart the web service after adding a checkpoint. Without checkpoints, structure
building, analysis, and DFT preparation remain available; AI generation reports
that its models are unavailable. CPU containers do not expose Apple Metal/MPS.

## Stop And Restart

```bash
docker compose --profile research stop
docker compose --profile research start
docker compose --profile research down
```

The bind-mounted research directory is retained by these commands.

## Rebuild Dependencies

The Python base image is pinned by digest. Python package versions and hashes are
recorded in `requirements-research.lock`; the PyTorch CPU index supplies its CPU
wheel. Regenerate the lock after changing `requirements.txt` or
`requirements-research.in`:

```bash
uv pip compile requirements-research.in --python-version 3.11 \
  --python-platform aarch64-unknown-linux-gnu --torch-backend cpu \
  --index-strategy unsafe-best-match \
  --default-index https://pypi.org/simple --emit-index-url --generate-hashes \
  --output-file requirements-research.lock
docker compose --profile research build
docker compose --profile research up -d
```

This lock and image target ARM64 CPU. An x86 or NVIDIA GPU deployment requires
its own dependency resolution and runtime validation.

## Scientific Scope

The coherent-interface builder uses pymatgen lattice matching and strains the film
to the substrate. Bidirectional matching changes the comparison direction; it does
not distribute strain between the two materials. A subsequent arbitrary film
rotation is a geometric operation and has not been validated as a commensurate
periodic supercell. Generated CP2K/SLURM files still require material parameters,
cluster settings, and convergence validation appropriate to the calculation.
