import onnx
import json
import numpy as np
import onnx_graphsurgeon as gs

from fpgaconvnet.parser.Parser import Parser
from fpgaconvnet.tools.layer_enum import LAYER_TYPE

# definition of the model name
model_name = "yolov5n"

# edit the onnx graph to remove the post-processing
graph = onnx.load(f"../onnx_models/{model_name}.onnx")

# load with graph surgeon
graph = gs.import_onnx(graph)

# get the extra operations to remove
max_idx = 0
for idx, node in enumerate(graph.nodes):
    if node.name == "/model.24/Reshape":
        reshape_l_0_idx = idx
    if node.name == "/model.24/Reshape_1":
        reshape_l_1_idx = idx
    if node.name == "/model.24/Reshape_2":
        reshape_m_0_idx = idx
    if node.name == "/model.24/Reshape_3":
        reshape_m_1_idx = idx
    if node.name == "/model.24/Reshape_4":
        reshape_r_0_idx = idx
    if node.name == "/model.24/Reshape_5":
        reshape_r_1_idx = idx

# remove extra operations
del graph.nodes[reshape_r_0_idx:reshape_r_1_idx+2]
del graph.nodes[reshape_m_0_idx:reshape_m_1_idx+1]
del graph.nodes[reshape_l_0_idx:reshape_l_1_idx+1]

# get output layers
conv_l = next(filter(lambda x: x.name == "/model.24/m.2/Conv", graph.nodes))
conv_m = next(filter(lambda x: x.name == "/model.24/m.1/Conv", graph.nodes))
conv_r = next(filter(lambda x: x.name == "/model.24/m.0/Conv", graph.nodes))

# get the resize layers
resize = next(filter(lambda x: x.name == "/model.11/Resize", graph.nodes))
resize.inputs[1] = gs.Constant("roi_0", np.array([0.0,0.0,0.0,0.0]))
resize = next(filter(lambda x: x.name == "/model.15/Resize", graph.nodes))
resize.inputs[1] = gs.Constant("roi_1", np.array([0.0,0.0,0.0,0.0]))

# create the output nodes
output_l = gs.Variable("output_2", shape=[1, 36, 10, 10], dtype="float32")
output_m = gs.Variable("output_1", shape=[1, 36, 20, 20], dtype="float32")
output_r = gs.Variable("output_0", shape=[1, 36, 40, 40], dtype="float32")

# connect the output nodes
conv_l.outputs = [ output_l ]
conv_m.outputs = [ output_m ]
conv_r.outputs = [ output_r ]

# update the graph outputs
graph.outputs = [ conv_l.outputs[0], conv_m.outputs[0], conv_r.outputs[0] ]

# cleanup graph
graph.cleanup()

# save the reduced network
graph = gs.export_onnx(graph)
graph.ir_version = 8 # need to downgrade the ir version
onnx.save(graph, f"../models/{model_name}-fpgaconvnet.onnx")

# create a parser
parser = Parser(backend="chisel", quant_mode="auto", convert_gemm_to_conv=False, custom_onnx=False)

# # give specific optimiser passes
# # TODO

# parse the network and perform all optimisations
net = parser.onnx_to_fpgaconvnet(f"../models/{model_name}-fpgaconvnet.onnx",
        "../zcu104.toml", False, save_opt_model=True)
net.update_partitions()

# set fine to max for all layers
for node in net.partitions[0].graph.nodes:
    if net.partitions[0].graph.nodes[node]["type"] == LAYER_TYPE.Convolution:
        net.partitions[0].graph.nodes[node]["hw"].fine = np.prod(net.partitions[0].graph.nodes[node]["hw"].kernel_size)
net.update_partitions()

# save the optimised model here
onnx.save(net.model, "yolov5n-opt.onnx")

# set the batch size to 1024
net.batch_size = 1024

# get resource and performance estimates
print(f"predicted latency (us): {net.get_latency()*1000000}")
print(f"predicted throughput (img/s): {net.get_throughput()} (batch size={net.batch_size})")
print(f"predicted resource usage: {net.partitions[0].get_resource_usage()}")

# save the configuration file
net.save_all_partitions("config.json")
net.create_report("report.json")

