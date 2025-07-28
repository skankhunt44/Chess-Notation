## set up env

python -m venv venv
source venv/bin/activate  # .\venv\Scripts\activate on Windows
pip install torch torchvision opencv-python tqdm
pip install matplotlib

## Train

cd chess_cnn
python chess_square_classifier2.py train \
       --data-dir data \
       --epochs 10 --batch-size 512 \
       --model-out ../assets/model.pt

## Runs inference on a new image (or one from the dataset) and prints an 8 × 8 occupancy matrix:
python chess_square_classifier2.py infer --model ../assets/model.pt --image data/0046.png


## preview after wrap
python chess_square_classifier2.py preview --image data/0046.png --show
