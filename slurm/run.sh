#!/bin/sh
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=0-08:00:00
#SBATCH --gres=gpu:v100l:2
#SBATCH --output=vistaformer-%N-%j.out
#SBATCH --mail-user=<redacted@gmail.com>
#SBATCH --mail-type=FAIL


CONFIG_PATH=""
DATA_PATH=""
OUTPUT_PATH=""
SRC_PATH=""

while [[ $# -gt 0 ]]; do
  case $1 in
    -d|--data-path) # path to directory containing training data
      DATA_PATH="$2"
      shift # past argument
      shift # past value
      ;;
    -o|--output-directory)
      OUTPUT_PATH="$2" # path to directory to output data
      shift
      shift
      ;;
    -s|--src-directory)
      SRC_PATH="$2" # path to source code containing directory
      shift
      shift
      ;;
    -c|--config-path)
      CONFIG_PATH="$2" # path to config file
      shift
      shift
      ;;
    -*|--*)
      echo "Unknown option $1"
      exit 1
      ;;
    *)
      shift # past argument
      ;;
  esac
done

# Check that filepaths exist, otherwise exit with non-zero exit code...
if [ ! -f $DATA_PATH ]; then
    echo "${DATA_PATH} file does not exist, exiting."
    exit 1
elif [ ! -d $SRC_PATH ]; then
    echo "${SRC_PATH} does not exist, exiting."
    exit 1
elif [ ! -d $OUTPUT_PATH ]; then
    echo "${OUTPUT_PATH} does not exist, exiting."
    exit 1
fi

# Create data and code directories on the Slurm node
mkdir -p ${SLURM_TMPDIR}/src ${SLURM_TMPDIR}/data

echo "Checking available storage on temp node: ${SLURM_TMPDIR}"
df -h $SLURM_TMPDIR

echo "Copying over the compressed files to data directory on SLURM node..."
cp $DATA_PATH ${SLURM_TMPDIR}/data

echo "Decompressing dataset on the target node... Here we go"
pushd ${SLURM_TMPDIR}/data
tar -xf $DATA_PATH
echo "Listing data in the data directory directory..."
ls -al
popd

cp -r ${SRC_PATH}/* ${SLURM_TMPDIR}/src # copy source code to slurm node
echo "Copying config file from ${CONFIG_PATH} to src directory..."
cp $CONFIG_PATH ${SLURM_TMPDIR}/src/model_default.yaml # overwrite default config

# Load modules and run singularity container
module load apptainer/1.2.4
module load cuda

nvidia-smi # output GPU information
echo "Outputting CUDA toolkit version..."
nvcc --version

apptainer exec --nv \
  --bind ${SLURM_TMPDIR}/src:/code \
  --bind ${SLURM_TMPDIR}/data:/data \
  --bind ${OUTPUT_PATH}:/outputs \
  remote_cattn.sif \
  /bin/bash -c 'cd /code && . /ml-env/bin/activate && python -m remote_cattn.train_and_evaluate.train'
