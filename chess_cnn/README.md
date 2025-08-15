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
       --model-out ../assets/model2.pt

## Runs inference on a new image (or one from the dataset) and prints an 8 × 8 occupancy matrix:
python chess_square_classifier2.py infer --model ../assets/model.pt --image data/0046.png


## preview after wrap
python chess_square_classifier2.py preview --image data/0046.png --show


# Preprocess your new session to crops
python chess_square_classifier2.py preprocess \
  --src-dir ../data/session_*/samples \
  --dst-dir data/crops_my_cam \
  --workers 2

# Train (from scratch)
python chess_square_classifier2.py train \
  --data-dir data/crops_my_cam \
  --preprocessed \
  --epochs 10 \
  --batch-size 256 \
  --lr 3e-4 \
  --model-out ../assets/model.pt

# ...or continue from your existing model.pt
python chess_square_classifier2.py train \
  --data-dir data/crops_my_cam \
  --preprocessed \
  --epochs 6 \
  --batch-size 256 \
  --lr 2e-4 \
  --resume ../assets/model.pt \
  --model-out ../assets/model.pt

# Quick sanity check on a saved warp
python chess_square_classifier2.py infer-warp \
  --model ../assets/model.pt \
  --board-img ../data/session_2591762550623958/samples/img_0001.png