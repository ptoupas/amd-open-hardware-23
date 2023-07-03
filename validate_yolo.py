import cv2
import toml
import sys
import os
import time
import json
import numpy as np
from IPython.display import display, Image, clear_output

sys.path.append(os.path.abspath("data_management"))
sys.path.append(os.path.abspath("onnx_models"))
sys.path.append(os.path.abspath("hardware"))

import data_preprocessing as dprep
import data_postprocessing as dpostp
import onnx_inference as onnx_inf
import validate as val

yolo_cfg = toml.load('yolov5n.toml')
inf_exec = yolo_cfg['inf_exec']
imgsz = yolo_cfg['input_data']['imgsz']
conf_thres = yolo_cfg['predictor']['conf_thres']
iou_thres = yolo_cfg['predictor']['iou_thres']
classes = yolo_cfg['predictor']['classes']
if not classes:
    classes = None
onnx_model_path = yolo_cfg['onnx_model_path']
if inf_exec == "fpga":
    onnx_model_path = f"{onnx_model_path.split('.onnx')[0]}_head.onnx"
visualize = yolo_cfg['visualize']
val_data_path = yolo_cfg['val_data_path']
out_img_path = yolo_cfg['out_img_path']
input_source = yolo_cfg['source']
bitstream_path = yolo_cfg['hardware']['bitstream_path']
weights_lookup_table = yolo_cfg['hardware']['weights_lookup_table']
weights_path = yolo_cfg['hardware']['weights_path']


stride, names, session, output_names = onnx_inf.load_model(onnx_model_path)


if inf_exec == "fpga":
    import fpgaconvnet_driver as hw_part
    #########################
    ### fpgaConvNet Setup ###
    #########################

    # initialise partition
    partition = hw_part.Partition(bitstream_path, 5)

    # add input buffers
    partition.add_input_buffer(0, 0, [320, 320, 3], bp=12)

    # add output buffers
    partition.add_output_buffer(2, 2, [40, 40, 256], bp=9, streams=2)
    partition.add_output_buffer(3, 3, [20, 20, 256], bp=9, streams=2)
    partition.add_output_buffer(4, 4, [10, 10, 256], bp=9, streams=2)

    # create fifos
    partition.add_fifo(0, 0, 2, 40*40*64 , burst=6400)
    partition.add_fifo(1, 1, 1, 20*20*128, burst=6400)

    # # setup hardware
    partition.reset_hardware()
    # p.start_hardware()

    # get the lookup table for the weights
    with open(weights_lookup_table, "r") as f:
        lookup = json.load(f)

    # iterate over the weights
    for layer, idx in lookup.items():

        # allocate weights and load them
        start_time = time.perf_counter()
        partition.reload_weights(idx, f"{weights_path}/{layer}.dat")
        pred_time = (time.perf_counter() - start_time)*1000
        print(f"[{idx}] {layer} loaded! ({pred_time:.2f} ms)")

    # setup hardware
    partition.reset_hardware()
    partition.start_hardware()


val_data = val.get_val_data(val_data_path)

iouv = np.linspace(0.5, 0.95, 10)
niou = iouv.size
seen = 0
stats = []

for img, labels in val_data:

    orig_img = img.copy()

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = dprep.img_preprocess(img, imgsz, stride)

    if inf_exec == "cpu":
        predictions, pred_time = onnx_inf.model_inf(img, session, output_names)
    elif inf_exec == "fpga":
        out0, out1, out2, pred_time = hw_part.run_fpgaconvnet(partition, img[0])
        start_time = time.perf_counter()
        predictions = session.run(output_names, {
            "/model.24/m.0/Conv_output_0": np.expand_dims(out0[:255,:,:], axis=0),
            "/model.24/m.1/Conv_output_0": np.expand_dims(out1[:255,:,:], axis=0),
            "/model.24/m.2/Conv_output_0": np.expand_dims(out2[:255,:,:], axis=0),
        })[0]

    predictions = dpostp.yolo_nms(predictions, conf_thres=0.001, iou_thres=0.6, classes=classes)

    for pred in predictions:
        seen += 1

        nl, npr = labels.shape[0], pred.shape[0]  # number of labels, predictions

        correct = np.zeros((npr, niou), dtype=bool)

        if npr == 0:
            if nl:
                stats.append((correct, *np.zeros((2, 0)), labels[:, 0]))

        predn = pred.copy()
        predn[:, :4] = dpostp.scale_boxes((imgsz, imgsz), predn[:, :4], orig_img.shape).round()

        if nl:
            labelsn = labels.copy()

            correct = val.process_batch(predn, labelsn, iouv)

        stats.append((correct, pred[:, 4], pred[:, 5], labels[:, 0])) # (correct, conf, pcls, tcls)

# Compute metrics
stats = [np.concatenate(x, 0) for x in zip(*stats)]
if len(stats) and stats[0].any():
    tp, fp, p, r, f1, ap, ap_class = val.ap_per_class(*stats, names=names)
    ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
    mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
nt = np.bincount(stats[3].astype(int), minlength=80)  # number of targets per class

# Print results
print("*"*40)
print("Class: {}".format("all"))
print("Images: {}".format(seen))
print("Instances: {}".format(nt.sum()))
print("Precision: {:.4f}".format(mp))
print("Recall: {:.4f}".format(mr))
print("mAP@0.5: {:.4f}".format(map50))
print("mAP@0.5:0.95: {:.4f}".format(map))
print("*"*40)