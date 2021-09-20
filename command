python3 train_ssd.py --dataset-type=voc --data=data/pinacle --model-dir=models/pinacle --batch-size=1 --workers=0 --epochs=1


python3 onnx_export.py --model-dir=models/pinacle


detectnet --model=models/pinacle/ssd-mobilenet.onnx --labels=models/pinacle/labels.txt --input-blob=input_0 --output-cvg=scores --output-bbox=boxes /dev/video0

