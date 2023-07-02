import math
import time
from typing import List
from dataclasses import dataclass
from functools import reduce

import numpy as np

import pynq
from pynq import MMIO

@dataclass
class Partition:
    bitstream_path: str
    n_dma: int
    weight_dma_index: int = 4
    baseaddr: int = 0xA0070000

    def __post_init__(self):

        # create an overlay
        self.overlay = pynq.Overlay(self.bitstream_path)

        # get all DMA
        self.dma = [ getattr(self.overlay, f"dma_{i}")\
                        for i in range(self.n_dma) ]

        # initialise partition register file
        self.regfile = MMIO(self.baseaddr, 0x1000)

        # create all buffers
        self.input_buffers = {}
        self.output_buffers = {}
        self.fifo_buffers = {}

        # binary points
        self.input_bp = {}
        self.output_bp = {}

        # shapes
        self.input_shape = {}
        self.output_shape = {}

        # DMA indices
        self.input_dma = {}
        self.output_dma = {}
        self.fifo_dma = {}

        # output streams
        self.output_streams = {}

        # FIFO parameters
        self.fifo_depth = {}
        self.fifo_burst = {}

    def reset_fifo_dma(self):
        for i in self.fifo_dma.keys():
            self.dma[self.fifo_dma[i][0]].recvchannel._first_transfer = True
            self.dma[self.fifo_dma[i][1]].sendchannel._first_transfer = True

    def add_input_buffer(self, index, dma_index, shape, bp=8):

        # add dma index, shape and binary point values
        # using the dma index as the key
        self.input_buffers[index] = pynq.allocate(shape=np.prod(shape), dtype=np.int16)
        self.input_dma[index] = dma_index
        self.input_shape[index] = shape
        self.input_bp[index] = bp

    def add_output_buffer(self, index, dma_index, shape, bp=8, streams=1):

        # add dma index, shape and binary point values
        # using the dma index as the key
        self.output_buffers[index] = pynq.allocate(shape=np.prod(shape), dtype=np.int16)
        self.output_dma[index] = dma_index
        self.output_shape[index] = shape
        self.output_bp[index] = bp
        self.output_streams[index] = streams

        # setup the hardware again with new output shapes
        self.setup_hardware()

    def add_fifo(self, index, dma_in, dma_out, depth, burst=64, streams=1):

        # save the fifo depth and burst size
        self.fifo_depth[index] = depth
        self.fifo_burst[index] = burst

        # add dma for fifo
        self.fifo_dma[index] = (dma_in, dma_out)

        # buffers for the fifo
        self.fifo_buffers[index] = [ pynq.allocate(
            shape=(burst), dtype=np.uint16) for _ in range(depth//burst) ]

        # setup hardware after adding fifo
        self.setup_hardware()

    def setup_hardware(self):
        # reset hardware, turn off updating, etc
        self.regfile.write(0x0, 0)

        # initialise regular output ports
        for idx in self.output_buffers.keys():
            self.regfile.write(0x8+idx*4,
                math.prod(self.output_shape[idx])//self.output_streams[idx])

        # initialise fifo output ports
        for idx in self.fifo_buffers.keys():
            self.regfile.write(0x8+idx*4,
                self.fifo_burst[idx]//1) # TODO include streams

    def reset_hardware(self):
        self.regfile.write(0x0, 0x2)
        self.regfile.write(0x0, 0x0)

    def start_hardware(self):
        self.regfile.write(0x0, 0x4)

    def stop_hardware(self):
        self.regfile.write(0x0, 0x0)

    def reload_weights(self, index: int, weights_filepath: str):

        # load the weights into a numpy array
        idx = 0
        with open(weights_filepath, "r") as f:
             weights = np.array([int(x, base=16) \
#             idx = (idx + 1)%3
#             weights = np.array([idx*512 \
                        for x in f.readlines() ], dtype=np.uint16)

        # allocate a pynq buffer for the weights
        weight_buffers = pynq.allocate(shape=weights.shape, dtype=np.uint16)

        # get the values of weights
        weight_buffers[:] = weights

        # set the weight index
        self.regfile.write(0x4, index)

        # set to update mode
        self.regfile.write(0x0, 0x1)

        # transfer the weights
        self.dma[self.weight_dma_index].sendchannel.transfer(weight_buffers)

        # wait for transfer to finish
        self.dma[self.weight_dma_index].sendchannel.wait()

        # end update mode
        self.regfile.write(0x0, 0x0)

        self.reset_hardware()

        # set the weight index somewhere else
        self.regfile.write(0x4, 0xFFFF)

        # start hardware
        self.start_hardware()

        # de-allocate weights
        del weight_buffers

    def download(self):

        # download the bitstream
        self.overlay.download()

        # setup the hardwarte
        self.setup_hardware()

    def send_dma(self, index: int):
        # self.start_hardware()
        self.dma[self.input_dma[index]].sendchannel.transfer(self.input_buffers[index])

    def recv_dma(self, index: int):
        self.dma[self.output_dma[index]].recvchannel.transfer(self.output_buffers[index])

    def wait_dma(self, index: int, send: bool = True, recv: bool = True):

        # wait to receive
        if recv:
            try:
                self.dma[index].recvchannel.wait()
            except:
                print("WARNING: recv channel finished")

        # wait to send
        if send:
            try:
                self.dma[index].sendchannel.wait()
            except:
                print("WARNING: send channel finished")

    def check_dma_done(self, idx, direction):

        # get the dma channel
        if direction == "recv":
            dma_channel = self.dma[idx].recvchannel
        if direction == "send":
            dma_channel = self.dma[idx].sendchannel

        # return true if it's the first transfer
        if dma_channel._first_transfer:
            return True

        # check running
        if not dma_channel.running:
            raise RuntimeError("DMA channel not started")

        # check error status
        error = dma_channel._mmio.read(dma_channel._offset + 4)
        if dma_channel.error:
            if error & 0x10:
                raise RuntimeError("DMA Internal Error (transfer length 0?)")
            if error & 0x20:
                raise RuntimeError(
                    "DMA Slave Error (cannot access memory map interface)"
                )
            if error & 0x40:
                raise RuntimeError("DMA Decode Error (invalid address)")

        # flush buffer
        if dma_channel.idle:
            if not dma_channel._flush_before:
                dma_channel._active_buffer.invalidate()
            dma_channel.transferred = dma_channel._mmio.read(dma_channel._offset + 0x28)

        # return if idle or not
        return dma_channel.idle

    def start_fifo(self, indices):

        # get the maximum counter value
        cntr_max = [ self.fifo_depth[i]//self.fifo_burst[i] for i in indices ]

        # get dma indices
        dma_from_idx = [ self.fifo_dma[i][0] for i in indices ]
        dma_to_idx = [ self.fifo_dma[i][1] for i in indices ]

        # cntr for in and out
        cntr_in = [ 0 for i in indices ]
        cntr_out = [ 0 for i in indices ]

        # iterate over DMA transfers
        while reduce(lambda a,b: a or b, map(lambda i: cntr_in[i] < cntr_max[i], indices)) \
                or reduce(lambda a,b: a or b, map(lambda i: cntr_out[i] < cntr_max[i], indices)):

            # iterate over FIFOs
            for i in indices:

                # transfer in
                if self.dma[dma_from_idx[i]].recvchannel._first_transfer:
                    self.dma[dma_from_idx[i]].recvchannel.transfer(
                            self.fifo_buffers[i][cntr_in[i]])
                if self.check_dma_done(dma_from_idx[i], "recv") and cntr_in[i] < cntr_max[i]:
                    cntr_in[i] += 1
                    if cntr_in[i] < cntr_max[i]:
                        self.dma[dma_from_idx[i]].recvchannel.transfer(
                                self.fifo_buffers[i][cntr_in[i]])

                # transfer out
                if self.dma[dma_to_idx[i]].sendchannel._first_transfer and cntr_out[i] < cntr_in[i]:
                    self.dma[dma_to_idx[i]].sendchannel.transfer(
                            self.fifo_buffers[i][cntr_out[i]])
                elif self.check_dma_done(dma_to_idx[i], "send") and \
                    cntr_out[i] < cntr_max[i] and cntr_out[i] < cntr_in[i]:
                    cntr_out[i] += 1
                    if cntr_out[i] < cntr_max[i]:
                        self.dma[dma_to_idx[i]].sendchannel.transfer(
                                self.fifo_buffers[i][cntr_out[i]])



# method for executing the accelerator
def run_fpgaconvnet(partition, input_data):

    # assign the input data
    data = np.moveaxis(input_data, 0, -1).flatten() * (2**partition.input_bp[0])
    partition.input_buffers[0][:] = data.astype(np.int16)

    # start all transfers
    start_time = time.perf_counter()

    # send in the input data
    partition.send_dma(0)

    # recieve data out
    partition.recv_dma(2)
    partition.recv_dma(3)
    partition.recv_dma(4)

    # start the FIFOs
    partition.start_fifo([0, 1])

    # wait for output DMAs to finish
    partition.wait_dma(0, recv=False)
    partition.wait_dma(2, send=False)
    partition.wait_dma(3, send=False)
    partition.wait_dma(4, send=False)

    # print statement
    pred_time = (time.perf_counter() - start_time)*1000

    # reset the FIFO
    partition.reset_fifo_dma()

    # get the outputs
    output_0 = np.moveaxis(np.reshape(partition.output_buffers[2],
        partition.output_shape[2]), -1, 0).astype(np.float32) / float(2**partition.output_bp[2])
    output_1 = np.moveaxis(np.reshape(partition.output_buffers[3],
        partition.output_shape[3]), -1, 0).astype(np.float32) / float(2**partition.output_bp[3])
    output_2 = np.moveaxis(np.reshape(partition.output_buffers[4],
        partition.output_shape[4]), -1, 0).astype(np.float32) / float(2**partition.output_bp[4])

    return output_0, output_1, output_2, pred_time
