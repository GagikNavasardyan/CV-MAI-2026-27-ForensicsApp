# ForensicsApp course starter

ForensicsApp is the semester project shell for the MSc Computer Vision course.  
Students add a small, demonstrable image-forensics feature as the course moves  
from classical image processing to deep learning and interpretability.

The starter already provides:

- a Tkinter desktop interface with tool, image, and results areas;
- opening and exporting common image formats;
- a fit-to-window image preview;
- undo, redo, and reset-to-original behavior;
- a tiny plug-in-style contract for adding course functionality;
- one analysis example and one image-processing example.

## Quick start

Python 3.10 or newer is recommended.

```
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python run.py
```

On some Linux distributions, Tkinter is a separate system package. For example,  
Ubuntu/Debian users may first need `sudo apt install python3-tk`.

The complete student handout is [docs/STUDENT_GUIDE.md](docs/STUDENT_GUIDE.md).

## Useful commands

```
python -m forensics_app
python -m unittest discover -s tests -v
python -m compileall forensics_app
```

## Course roadmap

The shell is intentionally neutral about the algorithms. Possible tool modules  
can follow the syllabus: filtering and edges, HOG, SIFT, bag of visual words,  
face detection/recognition, CNN classification and detection, transformers,  
semantic segmentation, and visualization/interpretability. Model weights and  
datasets should not be committed to the starter repository; document where they  
come from and keep paths configurable.

## CNN classification (Set 8)

1. Open [notebooks/set8_1_train_cnn.ipynb](notebooks/set8_1_train_cnn.ipynb) in Colab with a GPU runtime and run all  
   cells. It trains the CNN and downloads `set8_models.zip`.
2. Unzip it into `models/`. Each model has three files: `<name>.keras` (weights), `<name>.onnx` (what the app runs  
   with OpenCV) and `<name>_passport.json` (dataset, class names, preprocessing, training history).
3. In the app, open an image and choose **Deep learning → CNN classification**.

## Object detection with YOLO (Set 9.1)

1. Install the dependencies again (`python -m pip install -r requirements.txt`); this adds Ultralytics and PyTorch.
2. In the app, open an image and choose **Deep learning → Object detection (YOLO)**. The first run downloads  
   `yolo11n.pt` (YOLO11 nano, trained on the 80 COCO classes) into `models/`.
3. Tune the two thresholds (**Reset to defaults** restores 0.25 and 0.7):
   - **Confidence**: boxes whose best class score is lower are dropped. Higher means fewer but surer boxes.
   - **NMS overlap (IoU)**: one object yields many near-identical boxes, so non-maximum suppression keeps the most  
     confident one and drops boxes of the same class overlapping it by more than this IoU (overlap area / union  
     area). Lower merges more, including close objects; higher leaves duplicates.
4. Pick a view:
   - **Detected objects**: boxes and "label confidence" tags drawn with matplotlib, one `tab20` colour per class.
   - **Ultralytics plot()**: the library's own drawing, converted from BGR to RGB.
   - **plot() without BGR → RGB**: the same array shown as it is. `plot()` returns OpenCV's BGR channel order,  
     so read as RGB the red and blue channels swap: this is why the people on the slide are blue.

## U-Net segmentation (Set 11.1)

1. Unzip `TrayDataset.zip` (Set 10-11 materials) into `datasets/`, so the folders are `datasets/TrayDataset/XTrain`,  
   `yTrain`, `XTest` and `yTest`.
2. Install the ONNX exporter in the `.venv` (only the notebook needs it):  
   `uv pip install --python .venv/Scripts/python.exe onnx onnxscript`
3. Run [notebooks/set11_1_train_unet.ipynb](notebooks/set11_1_train_unet.ipynb) with the `.venv` kernel (about a minute  
   on a laptop GPU). It trains a small U-Net on a binary task, food vs. not food (mask classes from 5 on are food),  
   and saves `tray_food_unet.pt`, `tray_food_unet.onnx` and `tray_food_unet_passport.json` in `models/`.
4. In the app, open a tray image and choose **Deep learning → U-Net segmentation**. It shows the original image, the  
   segmentation mask and the mask overlaid on the image; on a dataset image (`XTest/<name>.jpg`) it also reads the  
   mask (`yTest/<name>.png`) and shows the ground truth and the Dice coefficient. The **Threshold** sets the food  
   probability above which a pixel is food.

## CLIP zero-shot classification (Set 10.1)

1. Install the dependencies again (`python -m pip install -r requirements.txt`); this adds Hugging Face Transformers.
2. In the app, open an image and choose **Deep learning → CLIP zero-shot classification**. The first run downloads  
   the model into the Hugging Face cache (about 600 MB for ViT-B/32, 1.7 GB for ViT-L/14).
3. Write the classes, one per line (any number, any text), and the template: `{}` is replaced by each class, so  
   `a photo of a {}.` turns `cat` into `a photo of a cat.`. CLIP encodes the image and every sentence, and the  
   softmax of their scaled cosine similarities gives the probabilities. Nothing is trained: change the text and the  
   classifier changes. The probabilities always sum to 1, so add an "other"-style class when none may fit.
