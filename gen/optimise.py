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
# parser = Parser(backend="chisel", quant_mode="auto", convert_gemm_to_conv=False, custom_onnx=False)
parser = Parser(backend="chisel", quant_mode="calibrate", convert_gemm_to_conv=False, custom_onnx=False)

# # give specific optimiser passes
# # TODO

# load the calibration data
with open("calibration_data.json", "r") as f:
    cal_data = json.load(f)

# add images to calibration data
cal_data["images"] = {
    "max_val": 2.0,
    "min_val": -2.0,
}

# # get the outputs
# cal_data["output_0"] = cal_data["/model.24/m.0/Conv_output_0"]
# cal_data["output_1"] = cal_data["/model.24/m.1/Conv_output_0"]
# cal_data["output_2"] = cal_data["/model.24/m.2/Conv_output_0"]

# parse the network and perform all optimisations
model_name = "yolov5n-imgsz-320"
net = parser.onnx_to_fpgaconvnet(f"../models/{model_name}-fpgaconvnet.onnx",
        "../zcu104.toml", False, save_opt_model=True, calibration_data=cal_data)
# net = parser.onnx_to_fpgaconvnet(f"../models/{model_name}-fpgaconvnet.onnx",
#         "../zcu104.toml", False, save_opt_model=True)
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
            p.graph.nodes[node]["hw"].coarse_out = 2

# increase coarse out
net.partitions[0].graph.nodes["Conv_14"]["hw"].coarse_out *= 2
net.partitions[0].graph.nodes["Conv_33"]["hw"].coarse_out *= 2
net.partitions[0].graph.nodes["Conv_57"]["hw"].coarse_out *= 2
net.partitions[0].graph.nodes["Conv_71"]["hw"].coarse_out *= 2
net.partitions[0].graph.nodes["Conv_79"]["hw"].coarse_out *= 4
net.partitions[0].graph.nodes["Conv_85"]["hw"].coarse_out *= 2
net.partitions[0].graph.nodes["Conv_91"]["hw"].coarse_out *= 2
net.partitions[0].graph.nodes["Conv_94"]["hw"].coarse_out *= 2
net.partitions[0].graph.nodes["Conv_100"]["hw"].coarse_out *= 2
net.partitions[0].graph.nodes["Conv_106"]["hw"].coarse_out *= 2
net.partitions[0].graph.nodes["Conv_109"]["hw"].coarse_out *= 2
net.partitions[0].graph.nodes["Conv_123"]["hw"].coarse_out *= 2
net.partitions[0].graph.nodes["Conv_137"]["hw"].coarse_out *= 2

# # set some layers to coarse out of 1
# net.partitions[0].graph.nodes["Conv_116"]["hw"].coarse_out = 1
# net.partitions[0].graph.nodes["Conv_130"]["hw"].coarse_out = 1
# net.partitions[0].graph.nodes["Conv_141"]["hw"].coarse_out = 1

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
        "/model.6/cv3/act/Div_output_0",
        "/model.4/cv3/act/Div_output_0",
        "/model.10/act/Div_output_0",
    ]
config["partition"][0]["output_nodes"] = [
        "/model.4/cv3/act/Div_output_0",
        "/model.6/cv3/act/Div_output_0",
        "/model.10/act/Div_output_0",
        "output_0",
        "output_1",
        "output_2",
    ]

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
    if layer["name"] == "HardSwish_82_split":
        for j, stream_out in enumerate(config["partition"][0]["layers"][i]["streams_out"]):
            if stream_out["node"] == "Concat_127":
                config["partition"][0]["layers"][i]["streams_out"][j]["node"] = "HardSwish_82_split"

    ## concat layers
    if layer["name"] == "Concat_84":
        config["partition"][0]["layers"][i]["streams_in"][1]["node"] = "Concat_84"
        config["partition"][0]["layers"][i]["streams_in"][1]["buffer_depth"] = 0
    if layer["name"] == "Concat_99":
        config["partition"][0]["layers"][i]["streams_in"][1]["node"] = "Concat_99"
        config["partition"][0]["layers"][i]["streams_in"][1]["buffer_depth"] = 0
    if layer["name"] == "Concat_127":
        config["partition"][0]["layers"][i]["streams_in"][1]["node"] = "Concat_127"
        config["partition"][0]["layers"][i]["streams_in"][1]["buffer_depth"] = 0


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
        # if np.prod(kernel_size) == 9 and depth > 3000:
        if depth > 3000:
            config["partition"][0]["layers"][i]["parameters"]["weights_ram_style"] = "ultra"

        # set to distrubuted if narrow and shallow
        if np.prod(kernel_size) == 1 and depth < 1000:
            config["partition"][0]["layers"][i]["parameters"]["weights_ram_style"] = "distributed"

# increase buffer depths for branches
for i, layer in enumerate(config["partition"][0]["layers"]):

    # Concat layers
    if layer["type"] == "CONCAT":
        config["partition"][0]["layers"][i]["streams_in"][0]["buffer_depth"] = 64
        config["partition"][0]["layers"][i]["streams_in"][1]["buffer_depth"] = 5000

    # increase add layer buffer depths
    if layer["type"] == "ELTWISE":
        config["partition"][0]["layers"][i]["streams_in"][0]["buffer_depth"] = 5000
        config["partition"][0]["layers"][i]["streams_in"][1]["buffer_depth"] = 64

    if layer["name"] == "Concat_113":
        config["partition"][0]["layers"][i]["streams_in"][1]["buffer_depth"] = 7500

    if layer["name"] == "Concat_78":
        config["partition"][0]["layers"][i]["streams_in"][0]["buffer_depth"] = 10000
        config["partition"][0]["layers"][i]["streams_in"][1]["buffer_depth"] = 7500
        config["partition"][0]["layers"][i]["streams_in"][2]["buffer_depth"] = 5000
        config["partition"][0]["layers"][i]["streams_in"][3]["buffer_depth"] = 2500

# # ## increase buffer depths
# # config["partition"][1]["layers"][concat_32_idx]["streams_in"][0]["buffer_depth"] = 64
# # config["partition"][1]["layers"][concat_32_idx]["streams_in"][1]["buffer_depth"] = 4000

# # config["partition"][1]["layers"][add_24_idx]["streams_in"][0]["buffer_depth"] = config["partition"][1]["layers"][add_24_idx]["streams_in"][0]["buffer_depth"]*2
# # config["partition"][1]["layers"][add_24_idx]["streams_in"][1]["buffer_depth"] = config["partition"][1]["layers"][add_24_idx]["streams_in"][1]["buffer_depth"]*2
# # config["partition"][1]["layers"][add_29_idx]["streams_in"][0]["buffer_depth"] = config["partition"][1]["layers"][add_29_idx]["streams_in"][0]["buffer_depth"]*2
# # config["partition"][1]["layers"][add_29_idx]["streams_in"][1]["buffer_depth"] = config["partition"][1]["layers"][add_29_idx]["streams_in"][1]["buffer_depth"]*2



# # config["partition"][1]["layers"][add_24_idx]["streams_in"][0]["buffer_depth"] = config["partition"][1]["layers"][add_24_idx]["streams_in"][0]["buffer_depth"]*2
# # config["partition"][1]["layers"][add_24_idx]["streams_in"][1]["buffer_depth"] = config["partition"][1]["layers"][add_24_idx]["streams_in"][1]["buffer_depth"]*2
# # config["partition"][1]["layers"][add_29_idx]["streams_in"][0]["buffer_depth"] = config["partition"][1]["layers"][add_29_idx]["streams_in"][0]["buffer_depth"]*2
# # config["partition"][1]["layers"][add_29_idx]["streams_in"][1]["buffer_depth"] = config["partition"][1]["layers"][add_29_idx]["streams_in"][1]["buffer_depth"]*2



# # add specific convolution storage types
# for i, layer in enumerate(config["partition"][0]["layers"]):
#     if layer["name"] == "Conv_139":
#         config["partition"][0]["layers"][i]["parameters"]["weights_ram_style"] = "distributed"
#     if layer["name"] == "Conv_120":
#         config["partition"][0]["layers"][i]["parameters"]["weights_ram_style"] = "distributed"


# save the updated config
with open(f"config.json", "w") as f:
    json.dump(config, f, indent=2)

# ## give the buffers more depth
# for i, layer in enumerate(config["partition"][0]["layers"]):
#     if layer["name"] == "Add_10":
#         add_10_idx = i
#     if layer["name"] == "Concat_13":
#         concat_13_idx = i
#     if layer["name"] == "Add_24":
#         add_24_idx = i
#     if layer["name"] == "Add_29":
#         add_29_idx = i
#     if layer["name"] == "Concat_32":
#         concat_32_idx = i
#     if layer["name"] == "HardSwish_34_split":
#         hardswish_34_split_idx = i
#     if layer["name"] == "Add_43":
#         add_43_idx = i
#     if layer["name"] == "Add_48":
#         add_48_idx = i
#     if layer["name"] == "Add_53":
#         add_53_idx = i
#     if layer["name"] == "Concat_56":
#         concat_56_idx = i
#     if layer["name"] == "HardSwish_58_split":
#         hardswish_58_split_idx = i
#     if layer["name"] == "Add_67":
#         add_67_idx = i
#     if layer["name"] == "Concat_70":
#         concat_70_idx = i
#     if layer["name"] == "Concat_78":
#         concat_78_idx = i
#     if layer["name"] == "Concat_84":
#         concat_84_idx = i
#     if layer["name"] == "HardSwish_82_split":
#         hardswish_82_split_idx = i
#     if layer["name"] == "Resize_83":
#         resize_83_idx = i

# ## assign concat streams in
# config["partition"][4]["layers"][concat_93_idx]["streams_in"] = [
#     {
#       "name": "concat_93_left",
#       "coarse": 1,
#       "buffer_depth": 16,
#       "node": "HardSwish_90"
#     },
#     {
#       "name": "concat_93_right",
#       "coarse": 1,
#       "buffer_depth": 2000,
#       "node": "HardSwish_92"
#     },
# ]



# # if config["partition"][0]["layers"][concat_13_idx]["streams_in"][0]["node"] == "HardSwish_12":
# #     print("WARNING: concat inputs wrong")
# #     config["partition"][0]["layers"][concat_13_idx]["streams_in"][0], config["partition"][0]["layers"][concat_13_idx]["streams_in"][1] = config["partition"][0]["layers"][concat_13_idx]["streams_in"][1], config["partition"][0]["layers"][concat_13_idx]["streams_in"][0]


# # # increase concat buffer depth
# # config["partition"][0]["layers"][concat_13_idx]["streams_in"][0]["buffer_depth"] = 64
# # config["partition"][0]["layers"][concat_13_idx]["streams_in"][1]["buffer_depth"] = 4000

# # # increase eltwise depths
# # config["partition"][0]["layers"][add_10_idx]["streams_in"][0]["buffer_depth"] = config["partition"][0]["layers"][add_10_idx]["streams_in"][0]["buffer_depth"]*2
# # config["partition"][0]["layers"][add_10_idx]["streams_in"][1]["buffer_depth"] = config["partition"][0]["layers"][add_10_idx]["streams_in"][1]["buffer_depth"]*2

# # # set Conv_0 weights to distributed
# # config["partition"][0]["layers"][conv_0_idx]["parameters"]["weights_ram_style"] = "distributed"
# # config["partition"][0]["layers"][conv_0_idx]["parameters"]["acc_t"]["width"] = 40
# # config["partition"][0]["layers"][conv_0_idx]["parameters"]["output_t"]["binary_point"] = 8

# # # fix partition 1 for hardware

# # ## correct input and output nodes
# # config["partition"][1]["input_nodes"] = ["/model.3/conv/Conv_output_0"]
# # config["partition"][1]["output_nodes"] = [
# #         "/model.4/cv3/act/Div_output_0",
# #         "/model.5/act/Div_output_0",
# #     ]

# # ## give the buffers more depth
# # for i, layer in enumerate(config["partition"][1]["layers"]):
# #     if layer["name"] == "Add_24":
# #         add_24_idx = i
# #     if layer["name"] == "Add_29":
# #         add_29_idx = i
# #     if layer["name"] == "Concat_32":
# #         concat_32_idx = i
# #     if layer["name"] == "HardSwish_34_split":
# #         hardswish_34_split_idx = i

# # ## get the correct concat order
# # if config["partition"][1]["layers"][concat_32_idx]["streams_in"][0]["node"] == "HardSwish_31":
# #     print("WARNING: concat inputs wrong")
# #     config["partition"][1]["layers"][concat_32_idx]["streams_in"][0], config["partition"][1]["layers"][concat_32_idx]["streams_in"][1] = config["partition"][1]["layers"][concat_32_idx]["streams_in"][1], config["partition"][1]["layers"][concat_32_idx]["streams_in"][0]

# # ## increase buffer depths
# # config["partition"][1]["layers"][concat_32_idx]["streams_in"][0]["buffer_depth"] = 64
# # config["partition"][1]["layers"][concat_32_idx]["streams_in"][1]["buffer_depth"] = 4000

# # config["partition"][1]["layers"][add_24_idx]["streams_in"][0]["buffer_depth"] = config["partition"][1]["layers"][add_24_idx]["streams_in"][0]["buffer_depth"]*2
# # config["partition"][1]["layers"][add_24_idx]["streams_in"][1]["buffer_depth"] = config["partition"][1]["layers"][add_24_idx]["streams_in"][1]["buffer_depth"]*2
# # config["partition"][1]["layers"][add_29_idx]["streams_in"][0]["buffer_depth"] = config["partition"][1]["layers"][add_29_idx]["streams_in"][0]["buffer_depth"]*2
# # config["partition"][1]["layers"][add_29_idx]["streams_in"][1]["buffer_depth"] = config["partition"][1]["layers"][add_29_idx]["streams_in"][1]["buffer_depth"]*2

# # ## add output for hardswish
# # config["partition"][1]["layers"][hardswish_34_split_idx]["streams_out"].append({
# #     "name": "HardSwish_34_out",
# #     "coarse": 1,
# #     "node": "HardSwish_34_split",
# # })

# # # fix partition 2 for hardware

# # ## correct input and output nodes
# # config["partition"][2]["input_nodes"] = [
# #         "/model.5/act/Div_output_0",
# #         "/model.5/act/Div_output_0",

# #     ]
# # config["partition"][2]["output_nodes"] = [
# #         "/model.6/cv3/act/Div_output_0",
# #         "/model.7/act/Div_output_0",
# #     ]

# # ## give the buffers more depth
# # for i, layer in enumerate(config["partition"][2]["layers"]):
# #     if layer["name"] == "Add_43":
# #         add_43_idx = i
# #     if layer["name"] == "Add_48":
# #         add_48_idx = i
# #     if layer["name"] == "Add_53":
# #         add_53_idx = i
# #     if layer["name"] == "Concat_56":
# #         concat_56_idx = i
# #     if layer["name"] == "HardSwish_58_split":
# #         hardswish_58_split_idx = i
# #     if layer["name"] == "Conv_59":
# #         conv_59_idx = i
# #     if layer["name"] == "Conv_41":
# #         conv_41_idx = i
# #     if layer["name"] == "Conv_46":
# #         conv_46_idx = i

# # ## get the correct concat order
# # config["partition"][2]["layers"][concat_56_idx]["streams_in"] = [
# #     {
# #       "name": "concat_56_left",
# #       "coarse": 1,
# #       "buffer_depth": 16,
# #       "node": "Add_53"
# #     },
# #     {
# #       "name": "concat_99_right",
# #       "coarse": 1,
# #       "buffer_depth": 16,
# #       "node": "HardSwish_55"
# #     },
# # ]

# # ## adjust the eltwise depths
# # config["partition"][2]["layers"][add_43_idx]["streams_in"] = [
# #     {
# #       "name": "add_43_left",
# #       "coarse": 1,
# #       "buffer_depth": 3000,
# #       "node": "HardSwish_38_split"
# #     },
# #     {
# #       "name": "add_43_right",
# #       "coarse": 1,
# #       "buffer_depth": 16,
# #       "node": "HardSwish_42"
# #     },
# # ]

# # config["partition"][2]["layers"][add_48_idx]["streams_in"] = [
# #     {
# #       "name": "add_48_left",
# #       "coarse": 1,
# #       "buffer_depth": 3000,
# #       "node": "Add_43_split"
# #     },
# #     {
# #       "name": "add_48_right",
# #       "coarse": 1,
# #       "buffer_depth": 16,
# #       "node": "HardSwish_47"
# #     },
# # ]

# # config["partition"][2]["layers"][add_53_idx]["streams_in"] = [
# #     {
# #       "name": "add_53_left",
# #       "coarse": 1,
# #       "buffer_depth": 3000,
# #       "node": "Add_48_split"
# #     },
# #     {
# #       "name": "add_53_right",
# #       "coarse": 1,
# #       "buffer_depth": 16,
# #       "node": "HardSwish_52"
# #     },
# # ]

# # ## add output for hardswish
# # config["partition"][2]["layers"][hardswish_58_split_idx]["streams_out"].append({
# #     "name": "HardSwish_58_out",
# #     "coarse": 1,
# #     "node": "HardSwish_58_split",
# # })

# # ## set memory type for convolution layers
# # config["partition"][2]["layers"][conv_41_idx]["parameters"]["weights_ram_style"] = "ultra"
# # config["partition"][2]["layers"][conv_46_idx]["parameters"]["weights_ram_style"] = "ultra"
# # config["partition"][2]["layers"][conv_59_idx]["parameters"]["weights_ram_style"] = "ultra"

# # # fix partition 3 for hardware

# # ## correct input and output nodes
# # config["partition"][3]["input_nodes"] = [
# #         "/model.7/act/Div_output_0",
# #         "/model.6/cv3/act/Div_output_0",

# #     ]
# # config["partition"][3]["output_nodes"] = [
# #         "/model.10/act/Div_output_0",
# #         "/model.12/Concat_output_0",
# #     ]

# # ## give the buffers more depth
# # for i, layer in enumerate(config["partition"][3]["layers"]):
# #     if layer["name"] == "Add_67":
# #         add_67_idx = i
# #     if layer["name"] == "Concat_70":
# #         concat_70_idx = i
# #     if layer["name"] == "Concat_78":
# #         concat_78_idx = i
# #     if layer["name"] == "Concat_84":
# #         concat_84_idx = i
# #     if layer["name"] == "HardSwish_82_split":
# #         hardswish_82_split_idx = i
# #     if layer["name"] == "Conv_65":
# #         conv_65_idx = i
# #     if layer["name"] == "Conv_71":
# #         conv_71_idx = i
# #     if layer["name"] == "Conv_79":
# #         conv_79_idx = i
# #     if layer["name"] == "Resize_83":
# #         resize_83_idx = i

# # ## assign concat streams in
# # config["partition"][3]["layers"][concat_70_idx]["streams_in"] = [
# #     {
# #       "name": "concat_70_left",
# #       "coarse": 1,
# #       "buffer_depth": 16,
# #       "node": "Add_67"
# #     },
# #     {
# #       "name": "concat_70_right",
# #       "coarse": 1,
# #       "buffer_depth": 3000,
# #       "node": "HardSwish_69"
# #     },
# # ]

# # config["partition"][3]["layers"][concat_78_idx]["streams_in"] = [
# #     {
# #       "name": "concat_78_0",
# #       "coarse": 1,
# #       "buffer_depth": 10000,
# #       "node": "HardSwish_74_split"
# #     },
# #     {
# #       "name": "concat_78_1",
# #       "coarse": 1,
# #       "buffer_depth": 7500,
# #       "node": "MaxPool_75_split"
# #     },
# #     {
# #       "name": "concat_78_2",
# #       "coarse": 1,
# #       "buffer_depth": 5000,
# #       "node": "MaxPool_76_split"
# #     },
# #     {
# #       "name": "concat_78_3",
# #       "coarse": 1,
# #       "buffer_depth": 2500,
# #       "node": "MaxPool_77"
# #     },
# # ]

# # config["partition"][3]["layers"][concat_84_idx]["streams_in"] = [
# #     {
# #       "name": "concat_84_right",
# #       "coarse": 1,
# #       "buffer_depth": 128,
# #       "node": "Resize_83"
# #     },
# #     {
# #       "name": "input_l",
# #       "coarse": 1,
# #       "buffer_depth": 128,
# #       "node": "Concat_84"
# #     },
# # ]

# # ## increase buffer depths
# # config["partition"][3]["layers"][add_67_idx]["streams_in"][0]["buffer_depth"] = config["partition"][3]["layers"][add_67_idx]["streams_in"][0]["buffer_depth"]*2
# # config["partition"][3]["layers"][add_67_idx]["streams_in"][1]["buffer_depth"] = config["partition"][3]["layers"][add_67_idx]["streams_in"][1]["buffer_depth"]*2

# # ## add output for hardswish
# # config["partition"][3]["layers"][hardswish_82_split_idx]["streams_out"].append({
# #     "name": "HardSwish_82_out",
# #     "coarse": 1,
# #     "node": "HardSwish_82_split",
# # })

# # ## set memory type for convolution layers
# # config["partition"][3]["layers"][conv_65_idx]["parameters"]["weights_ram_style"] = "ultra"
# # config["partition"][3]["layers"][conv_71_idx]["parameters"]["weights_ram_style"] = "ultra"
# # config["partition"][3]["layers"][conv_79_idx]["parameters"]["weights_ram_style"] = "ultra"

# # ## set proper scales for resize
# # config["partition"][3]["layers"][resize_83_idx]["parameters"]["scale"] = [2, 2, 1]

# # # fix partition 4 for hardware

# # ## correct input and output nodes
# # config["partition"][4]["input_nodes"] = [
# #         "/model.12/Concat_output_0",
# #         "/model.4/cv3/act/Div_output_0",

# #     ]
# # config["partition"][4]["output_nodes"] = [
# #         "output_0",
# #         "/model.19/Concat_output_0",
# #     ]

# # ## give the buffers more depth
# # for i, layer in enumerate(config["partition"][4]["layers"]):
# #     if layer["name"] == "Concat_93":
# #         concat_93_idx = i
# #     if layer["name"] == "Concat_99":
# #         concat_99_idx = i
# #     if layer["name"] == "Concat_108":
# #         concat_108_idx = i
# #     if layer["name"] == "Concat_113":
# #         concat_113_idx = i
# #     if layer["name"] == "HardSwish_97_split":
# #         hardswish_97_split_idx = i
# #     if layer["name"] == "Conv_89":
# #         conv_89_idx = i
# #     if layer["name"] == "Conv_94":
# #         conv_94_idx = i
# #     if layer["name"] == "Resize_98":
# #         resize_98_idx = i

# # ## assign concat streams in
# # config["partition"][4]["layers"][concat_93_idx]["streams_in"] = [
# #     {
# #       "name": "concat_93_left",
# #       "coarse": 1,
# #       "buffer_depth": 16,
# #       "node": "HardSwish_90"
# #     },
# #     {
# #       "name": "concat_93_right",
# #       "coarse": 1,
# #       "buffer_depth": 2000,
# #       "node": "HardSwish_92"
# #     },
# # ]

# # config["partition"][4]["layers"][concat_99_idx]["streams_in"] = [
# #     {
# #       "name": "concat_99_left",
# #       "coarse": 1,
# #       "buffer_depth": 16,
# #       "node": "Resize_98"
# #     },
# #     {
# #       "name": "concat_99_right",
# #       "coarse": 1,
# #       "buffer_depth": 16,
# #       "node": "Concat_99"
# #     },
# # ]

# # config["partition"][4]["layers"][concat_108_idx]["streams_in"] = [
# #     {
# #       "name": "concat_108_left",
# #       "coarse": 1,
# #       "buffer_depth": 16,
# #       "node": "HardSwish_105"
# #     },
# #     {
# #       "name": "concat_108_right",
# #       "coarse": 1,
# #       "buffer_depth": 2000,
# #       "node": "HardSwish_107"
# #     },
# # ]

# # config["partition"][4]["layers"][concat_113_idx]["streams_in"] = [
# #     {
# #       "name": "concat_113_left",
# #       "coarse": 1,
# #       "buffer_depth": 16,
# #       "node": "HardSwish_112"
# #     },
# #     {
# #       "name": "concat_113_right",
# #       "coarse": 1,
# #       "buffer_depth": 7500,
# #       "node": "HardSwish_97_split"
# #     },
# # ]

# # ## set memory type for convolution layers
# # config["partition"][4]["layers"][conv_89_idx]["parameters"]["weights_ram_style"] = "ultra"
# # config["partition"][4]["layers"][conv_94_idx]["parameters"]["weights_ram_style"] = "ultra"

# # ## set proper scales for resize
# # config["partition"][4]["layers"][resize_98_idx]["parameters"]["scale"] = [2, 2, 1]

# # # ## add output for hardswish
# # # config["partition"][4]["layers"][hardswish_97_split_idx]["streams_out"].append({
# # #     "name": "HardSwish_97_out",
# # #     "coarse": 1,
# # #     "node": "HardSwish_97_split",
# # # })

# # # ## set memory type for convolution layers
# # # config["partition"][3]["layers"][conv_65_idx]["parameters"]["weights_ram_style"] = "ultra"
# # # config["partition"][3]["layers"][conv_71_idx]["parameters"]["weights_ram_style"] = "ultra"
# # # config["partition"][3]["layers"][conv_79_idx]["parameters"]["weights_ram_style"] = "ultra"

# # # fix partition 5 for hardware

# # ## correct input and output nodes
# # config["partition"][5]["input_nodes"] = [
# #         "/model.19/Concat_output_0",
# #         "/model.10/act/Div_output_0",

# #     ]
# # config["partition"][5]["output_nodes"] = [
# #         "output_1",
# #         "output_2",
# #     ]

# # ## give the buffers more depth
# # for i, layer in enumerate(config["partition"][5]["layers"]):
# #     if layer["name"] == "Concat_122":
# #         concat_122_idx = i
# #     if layer["name"] == "Concat_127":
# #         concat_127_idx = i
# #     if layer["name"] == "Concat_136":
# #         concat_136_idx = i
# #     if layer["name"] == "Conv_125":
# #         conv_125_idx = i
# #     if layer["name"] == "Conv_132":
# #         conv_132_idx = i
# #     if layer["name"] == "Conv_137":
# #         conv_137_idx = i

# # config["partition"][5]["layers"][concat_122_idx]["streams_in"] = [
# #     {
# #       "name": "concat_122_left",
# #       "coarse": 1,
# #       "buffer_depth": 16,
# #       "node": "HardSwish_119"
# #     },
# #     {
# #       "name": "concat_122_right",
# #       "coarse": 1,
# #       "buffer_depth": 3000,
# #       "node": "HardSwish_121"
# #     },
# # ]

# # config["partition"][5]["layers"][concat_127_idx]["streams_in"] = [
# #     {
# #       "name": "concat_127_left",
# #       "coarse": 1,
# #       "buffer_depth": 16,
# #       "node": "HardSwish_126"
# #     },
# #     {
# #       "name": "concat_127_right",
# #       "coarse": 1,
# #       "buffer_depth": 16,
# #       "node": "Concat_127"
# #     },
# # ]

# # config["partition"][5]["layers"][concat_136_idx]["streams_in"] = [
# #     {
# #       "name": "concat_136_left",
# #       "coarse": 1,
# #       "buffer_depth": 16,
# #       "node": "HardSwish_133"
# #     },
# #     {
# #       "name": "concat_136_right",
# #       "coarse": 1,
# #       "buffer_depth": 2000,
# #       "node": "HardSwish_135"
# #     },
# # ]

# # config["partition"][5]["layers"][conv_125_idx]["parameters"]["weights_ram_style"] = "ultra"
# # config["partition"][5]["layers"][conv_132_idx]["parameters"]["weights_ram_style"] = "ultra"


