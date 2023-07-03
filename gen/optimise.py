import onnx
import json
import numpy as np
import onnx_graphsurgeon as gs
import networkx as nx

from fpgaconvnet.parser.Parser import Parser
from fpgaconvnet.tools.layer_enum import LAYER_TYPE

import fpgaconvnet.tools.graphs as graphs

from fpgaconvnet.models.partition import Partition

# create a parser
parser = Parser(backend="chisel", quant_mode="calibrate", convert_gemm_to_conv=False, custom_onnx=False)

# load the calibration data
with open("calibration_data.json", "r") as f:
    cal_data = json.load(f)

# add images to calibration data
cal_data["images"] = {
    "max_val": 2.0,
    "min_val": -2.0,
}

# parse the network and perform all optimisations
model_name = "yolov5n"
net = parser.onnx_to_fpgaconvnet(f"../onnx_models/{model_name}-fpgaconvnet.onnx",
        "../zcu104.toml", False, save_opt_model=True, calibration_data=cal_data)
net.update_partitions()

"""
Partitioning

Break up the network into excutable chunks, with a focus on reducing long residual connections
"""

# set the batch size to 1024
net.batch_size = 1
net.update_partitions()

"""
Optimise

Manual optimisations applied to the hardware graph
"""
# set fine to max for all layers, and increase coarse out
for p in net.partitions:
    for node in p.graph.nodes:
        if p.graph.nodes[node]["type"] == LAYER_TYPE.Convolution:
            p.graph.nodes[node]["hw"].fine = np.prod(p.graph.nodes[node]["hw"].kernel_size)
            p.graph.nodes[node]["hw"].coarse_out = 4

# reduce certain coarse out
coarse_out_min = 2
net.partitions[0].graph.nodes["Conv_6"]["hw"].coarse_out = coarse_out_min
net.partitions[0].graph.nodes["Conv_8"]["hw"].coarse_out = coarse_out_min
net.partitions[0].graph.nodes["Conv_20"]["hw"].coarse_out = coarse_out_min
net.partitions[0].graph.nodes["Conv_22"]["hw"].coarse_out = coarse_out_min
net.partitions[0].graph.nodes["Conv_25"]["hw"].coarse_out = coarse_out_min
net.partitions[0].graph.nodes["Conv_27"]["hw"].coarse_out = coarse_out_min
net.partitions[0].graph.nodes["Conv_39"]["hw"].coarse_out = coarse_out_min
net.partitions[0].graph.nodes["Conv_41"]["hw"].coarse_out = coarse_out_min
net.partitions[0].graph.nodes["Conv_44"]["hw"].coarse_out = coarse_out_min
net.partitions[0].graph.nodes["Conv_46"]["hw"].coarse_out = coarse_out_min
net.partitions[0].graph.nodes["Conv_49"]["hw"].coarse_out = coarse_out_min
net.partitions[0].graph.nodes["Conv_51"]["hw"].coarse_out = coarse_out_min
net.partitions[0].graph.nodes["Conv_63"]["hw"].coarse_out = coarse_out_min
net.partitions[0].graph.nodes["Conv_87"]["hw"].coarse_out = coarse_out_min
net.partitions[0].graph.nodes["Conv_89"]["hw"].coarse_out = coarse_out_min
net.partitions[0].graph.nodes["Conv_89"]["hw"].coarse_out = coarse_out_min
net.partitions[0].graph.nodes["Conv_102"]["hw"].coarse_out = coarse_out_min
net.partitions[0].graph.nodes["Conv_104"]["hw"].coarse_out = coarse_out_min
net.partitions[0].graph.nodes["Conv_111"]["hw"].coarse_out = coarse_out_min
net.partitions[0].graph.nodes["Conv_116"]["hw"].coarse_out = coarse_out_min
net.partitions[0].graph.nodes["Conv_118"]["hw"].coarse_out = coarse_out_min
net.partitions[0].graph.nodes["Conv_125"]["hw"].coarse_out = coarse_out_min
net.partitions[0].graph.nodes["Conv_130"]["hw"].coarse_out = coarse_out_min

# increase coarse out
net.partitions[0].graph.nodes["Conv_14"]["hw"].coarse_out *= 2
net.partitions[0].graph.nodes["Conv_33"]["hw"].coarse_out *= 2
net.partitions[0].graph.nodes["Conv_57"]["hw"].coarse_out *= 2
net.partitions[0].graph.nodes["Conv_71"]["hw"].coarse_out *= 2
net.partitions[0].graph.nodes["Conv_79"]["hw"].coarse_out *= 4 # too big
net.partitions[0].graph.nodes["Conv_85"]["hw"].coarse_out *= 2
net.partitions[0].graph.nodes["Conv_91"]["hw"].coarse_out *= 2
net.partitions[0].graph.nodes["Conv_94"]["hw"].coarse_out *= 2
net.partitions[0].graph.nodes["Conv_100"]["hw"].coarse_out *= 2
net.partitions[0].graph.nodes["Conv_106"]["hw"].coarse_out *= 2
net.partitions[0].graph.nodes["Conv_109"]["hw"].coarse_out *= 2
net.partitions[0].graph.nodes["Conv_123"]["hw"].coarse_out *= 2
net.partitions[0].graph.nodes["Conv_137"]["hw"].coarse_out *= 2

# increase coarse in of final layers
net.partitions[0].graph.nodes["Conv_139"]["hw"].coarse_in *= 8
net.partitions[0].graph.nodes["Conv_140"]["hw"].coarse_in *= 4
net.partitions[0].graph.nodes["Conv_141"]["hw"].coarse_in *= 2

# give correct scales to the
net.partitions[0].graph.nodes["Resize_83"]["hw"].scales = [2, 2, 1]
net.partitions[0].graph.nodes["Resize_98"]["hw"].scales = [2, 2, 1]

# update partitions
net.update_partitions()

# save the configuration file
net.save_all_partitions("config.json")

# get resource and performance estimates
print("(network performance)")
print(f"predicted latency (us): {net.get_latency()*1000000}")
print(f"predicted throughput (img/s): {net.get_throughput()} (batch size={net.batch_size})")
print("")

for (i, p) in enumerate(net.partitions):
    print(f"(partition index {i})")
    print(f"predicted throughput: {p.get_cycle()}")
    print(f"predicted resource usage: {p.get_resource_usage()}")
    for node in nx.topological_sort(p.graph):
        print(f"{node}:\t {p.graph.nodes[node]['hw'].latency()}")


# add additional information to configuration
with open(f"config.json", "r") as f:
    config = json.load(f)

# fix partition 0 for hardware

## correct input and output nodes
config["partition"][0]["input_nodes"] = [
        "images",
        "/model.6/cv3/act/Mul_output_0",
        "/model.4/cv3/act/Mul_output_0",
        # "/model.10/act/Mul_output_0",
    ]
config["partition"][0]["output_nodes"] = [
        "/model.4/cv3/act/Mul_output_0",
        "/model.6/cv3/act/Mul_output_0",
        "/model.24/m.2/Conv_output_0",
        "/model.24/m.1/Conv_output_0",
        "/model.24/m.0/Conv_output_0",
    ]

# set first convolution to use distributed weights
config["partition"][0]["layers"][0]["parameters"]["weights_ram_style"] = "distributed"

# use URAM for large convolution layers
for i, layer in enumerate(config["partition"][0]["layers"]):
    if layer["type"] == "CONVOLUTION":

        # get channels, filters, coarse, kernel size
        channels    = config["partition"][0]["layers"][i]["parameters"]["channels_in"]
        filters     = config["partition"][0]["layers"][i]["parameters"]["filters"]
        coarse_in   = config["partition"][0]["layers"][i]["parameters"]["coarse_in"]
        coarse_out  = config["partition"][0]["layers"][i]["parameters"]["coarse_out"]
        kernel_size = config["partition"][0]["layers"][i]["parameters"]["kernel_size"]

        # get the depth of the weights
        depth = channels*filters//(coarse_in*coarse_out)

        # set to uram if wide and deep
        if depth > 3000:
            config["partition"][0]["layers"][i]["parameters"]["weights_ram_style"] = "ultra"

        # set to distrubuted if narrow and shallow
        if np.prod(kernel_size) == 1 and depth < 1000:
            config["partition"][0]["layers"][i]["parameters"]["weights_ram_style"] = "distributed"

# increase buffer depths for branches
for i, layer in enumerate(config["partition"][0]["layers"]):

    # Concat layers
    if layer["type"] == "CONCAT":
        config["partition"][0]["layers"][i]["streams_in"][0]["buffer_depth"] = 32
        config["partition"][0]["layers"][i]["streams_in"][1]["buffer_depth"] = 2000

    # increase add layer buffer depths
    if layer["type"] == "ELTWISE":
        config["partition"][0]["layers"][i]["streams_in"][0]["buffer_depth"] = 2000
        config["partition"][0]["layers"][i]["streams_in"][1]["buffer_depth"] = 32

    if layer["name"] == "Concat_56":
        config["partition"][0]["layers"][i]["streams_in"][1]["buffer_depth"] = 5000

    if layer["name"] == "Concat_32":
        config["partition"][0]["layers"][i]["streams_in"][1]["buffer_depth"] = 3500

    if layer["name"] == "Concat_78":
        config["partition"][0]["layers"][i]["streams_in"][0]["buffer_depth"] = 9000
        config["partition"][0]["layers"][i]["streams_in"][1]["buffer_depth"] = 6500
        config["partition"][0]["layers"][i]["streams_in"][2]["buffer_depth"] = 3500
        config["partition"][0]["layers"][i]["streams_in"][3]["buffer_depth"] = 1500

    if layer["name"] == "Concat_127":
        config["partition"][0]["layers"][i]["streams_in"][1]["buffer_depth"] = 7500

# remove long branches
for i, layer in enumerate(config["partition"][0]["layers"]):

    ## split layers
    if layer["name"] == "HardSwish_34_split":
        for j, stream_out in enumerate(config["partition"][0]["layers"][i]["streams_out"]):
            if stream_out["node"] == "Concat_99":
                config["partition"][0]["layers"][i]["streams_out"][j]["node"] = "HardSwish_34_split"
    if layer["name"] == "HardSwish_58_split":
        for j, stream_out in enumerate(config["partition"][0]["layers"][i]["streams_out"]):
            if stream_out["node"] == "Concat_84":
                config["partition"][0]["layers"][i]["streams_out"][j]["node"] = "HardSwish_58_split"

    ## concat layers
    if layer["name"] == "Concat_84":
        config["partition"][0]["layers"][i]["streams_in"][1]["node"] = "Concat_84"
        config["partition"][0]["layers"][i]["streams_in"][1]["buffer_depth"] = 256
    if layer["name"] == "Concat_99":
        config["partition"][0]["layers"][i]["streams_in"][1]["node"] = "Concat_99"
        config["partition"][0]["layers"][i]["streams_in"][1]["buffer_depth"] = 256

# set specific binary points
for i, layer in enumerate(config["partition"][0]["layers"]):

    # convolution in layer
    if layer["name"] in ["Conv_0"]:
        config["partition"][0]["layers"][i]["parameters"]["input_t"]["binary_point"] = 12
        config["partition"][0]["layers"][i]["parameters"]["acc_t"]["width"] = 48

    # find final convolution layers
    if layer["name"] in ["Conv_139", "Conv_140", "Conv_141"]:
        config["partition"][0]["layers"][i]["parameters"]["output_t"]["binary_point"] = 10

    # also change last squeeze layers
    if layer["name"] in ["squeeze_Conv_139", "squeeze_Conv_140", "squeeze_Conv_141"]:
        config["partition"][0]["layers"][i]["parameters"]["data_t"]["binary_point"] = 10

# set certain layers to URAM
for i, layer in enumerate(config["partition"][0]["layers"]):

    # find first convolution layers
    if layer["name"] in ["Conv_2", "Conv_8"]:
        config["partition"][0]["layers"][i]["parameters"]["weights_ram_style"] = "distributed"

# save the updated config
with open(f"config.json", "w") as f:
    json.dump(config, f, indent=2)

