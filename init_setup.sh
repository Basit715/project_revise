echo [$(date)]: "START"
echo [$(date)] : "CREATING ENV WITH PYTHON 3.9 VERSION"

conda create --prefix ./env python=3.8 -y

echo [$(date)]: "ACTIVATING THE ENVIRONMENT"

source activate ./env

echo [$(date)]: "INSTALLING THE DEV REQUIREMENTS"

pip install -r requirements.txt

echo [$(date)]: "END"

