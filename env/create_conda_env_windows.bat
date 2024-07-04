# worked on 2024-7-3
conda create --name lstm "python<3.10" -y
conda activate lstm
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0 -y
python -m pip install "tensorflow<2.11"
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
pip install PyYAML pandas matplotlib scikit-learn emd