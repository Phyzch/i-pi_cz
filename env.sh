# ENV_BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" #Works for linux but not MAC
ENV_BASE_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "i-pi path adds to PATH, PYTHONPATH, IPI_ROOT:"
echo $ENV_BASE_DIR 


export PATH=$ENV_BASE_DIR/bin:$PATH
export PYTHONPATH=$ENV_BASE_DIR:$PYTHONPATH
export IPI_ROOT=$ENV_BASE_DIR

unset ENV_BASE_DIR
