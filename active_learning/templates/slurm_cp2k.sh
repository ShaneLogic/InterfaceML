#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
#SBATCH --nodes={nodes}
#SBATCH --ntasks-per-node={ntasks_per_node}
#SBATCH --time={time}
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
{account_line}

echo "Job started at $(date)"
echo "Running on $(hostname)"
echo "Working directory: $(pwd)"

module load {cp2k_module}

srun {cp2k_binary} -i cp2k.inp -o cp2k.out

echo "Job finished at $(date)"
