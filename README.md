# FPGA Object Detection with YOLOv5

This repository provides an FPGA-based solution for executing object detection, focusing specifically on the popular YOLOv5 model architecture. By leveraging the power of Field-Programmable Gate Arrays (FPGAs) and utilizing both the [**fpgaConvNet**](https://github.com/AlexMontgomerie/fpgaconvnet-model) and the [**Xilinx PYNQ**](http://www.pynq.io/) frameworks, this solution enables high-performance and efficient execution of object detection tasks.

## Table of Contents

- [Installation](#installation)
  - [FPGA Board Setup with PYNQ](#setup-fpga-with-pynq)
  - [Python Environment Setup](#python-environment-setup)
- [Usage](#usage)
- [Documentation](#documentation)
- [License](#license)
- [Citation](#citation)



## Installation

### Setup FPGA with PYNQ

1. Download the latest PYNQ image for your FPGA board from [here](http://www.pynq.io/board.html). For this project, we used the `ZCU104` board which is officially supported by PYNQ. If you are using a different board, you may need to build the PYNQ image yourself. See [here](https://github.com/Xilinx/PYNQ/issues/1425#issuecomment-1601772627) how we built the PYNQ image for the `ZCU106` board.
***Important Notice 1:*** *Make sure to download or build the ***v3.0.1*** version of the PYNQ image which is compatible with the Xilinx ***Vivado 2022*** version which is used to generate the final bistream.*
2. Flash the PYNQ image to an SD card.
3. Insert the SD card into the FPGA board and power it on.
2. Follow the instructions [here](https://pynq.readthedocs.io/en/latest/getting_started/zcu104_setup.html) for the initial setup of the `ZCU104` board. (similar procedure is also applicable to other boards)
4. Once you have connected the board to your network connect to it via SSH. The default credentials are `xilinx` for both the username and password.

### Python Environment Setup
Once you have connected to the FPGA board via SSH, clone this repository and run the following commands:

***Important Notice 2:*** *The default environment where the jupyter notebooks are executed is the `pynq-venv` environment. In order to install the necessary dependencies, make sure that you have activated the `pynq-venv` environment before running the following commands. You can activate the `pynq-venv` environment by running the following command:*
```shell
$ source /usr/local/share/pynq-venv/bin/activate
```
After you have activated the `pynq-venv` environment, run the following commands to install the necessary dependencies:
```shell
$ git clone --recurse-submodules https://github.com/ptoupas/amd-open-hardware-23
$ cd amd-open-hardware-23
$ pip install -r requirements.txt
```

## Usage

We provide a set of Jupyter notebooks that demonstrate the usage of the FPGA-based object detection solution. The notebooks are located in the root directory of this repository. The following table provides a brief description of each notebook:

| Notebook | Description |
| --- | --- |
| `comparison.ipynb` | This notebook provides a comparison between the performance of the FPGA-based solution and the CPU-based solution with ONNX. |
| `predict_yolo.ipynb` | This notebook demonstrates how to use the FPGA-based solution to perform object detection through a webcam. It executes the YOLOv5 model on the FPGA and displays the results on the screen in real-time. |
| `validate_yolo.ipynb` | This notebook provides a validation of the FPGA-based solution on the coco128 dataset. It executes the YOLOv5 model on the FPGA and displays the validation results on the screen. |


***Important Notice 3:*** *All of the above notebooks can be configured through various parameters that can be easily modified at the `yolov5n.toml` file located in the root directory of this repository. For more information about the available parameters, please refer to the [documentation](#documentation) section.*

## Documentation

The following table provides a brief description of the available parameters that can be configured through the `yolov5n.toml` file:

| Parameter | Description |
| --- | --- |
| `onnx_model_path` | The path to the ONNX model file used for the onnxruntime inference (cpu) and for executing the head of the YOLOv5 model on the FPGA scenario. |
| `visualize` | Whether to visualize the results of the inference or not. |
| `val_data_path` | The path to the coco128 dataset used for the validation scenario. |
| `inf_exec` | The target device for the inference execution. Can be either `cpu` or `fpga`. |
| `source` | The source of the input data. Can be either `webcam` or a path to a video file (an example video file is provided in the `input_video` directory). |
| `input_data.imgsz` | The size of the input image. |
| `predictor.conf_thres` | The confidence threshold to be used during the detection. |
| `predictor.iou_thres` | The IoU threshold to be used during the detection. |
| `predictor.classes` | The classes to be identified during the detection. Leave empty to identify all classes. |
| `hardware.bitstream_path` | The path to the bitstream file generated by Vivado. |
| `hardware.weights_path` | The path to the weights directory containing the weights of the YOLOv5 model that will be loaded to the FPGA. |

## License

This project is licensed under the GNU GPLv3 License - see the [LICENSE](https://github.com/ptoupas/amd-open-hardware-23/blob/main/LICENSE) file for details.

## Citation
If you find this project useful in your research, please consider citing our work:
```BibTeX
@article{toupas2023harflow3d,
        title={HARFLOW3D: A Latency-Oriented 3D-CNN Accelerator Toolflow for HAR on FPGA Devices},
        author={Toupas, Petros and Montgomerie-Corcoran, Alexander and Bouganis, Christos-Savvas and Tzovaras, Dimitrios},
        doi={10.48550/arXiv.2303.17218},
        journal={arXiv preprint arXiv:2303.17218},
        year={2023}
}

@inproceedings{montgomerie2022samo,
        author={Montgomerie-Corcoran, Alexander and Yu, Zhewen and Bouganis, Christos-Savvas},
        booktitle={2022 32nd International Conference on Field-Programmable Logic and Applications (FPL)},
        title={SAMO: Optimised Mapping of Convolutional Neural Networks to Streaming Architectures},
        year={2022},
        volume={},
        number={},
        pages={418-424},
        doi={10.1109/FPL57034.2022.00069}
}

@article{venieris2019fpgaconvnet,
        author={Venieris, Stylianos I. and Bouganis, Christos-Savvas},
        journal={IEEE Transactions on Neural Networks and Learning Systems},
        title={fpgaConvNet: Mapping Regular and Irregular Convolutional Neural Networks on FPGAs},
        year={2019},
        volume={30},
        number={2},
        pages={326-342},
        doi={10.1109/TNNLS.2018.2844093}
}
```