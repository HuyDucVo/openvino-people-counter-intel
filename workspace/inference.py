#!/usr/bin/env python3
"""
 Copyright (c) 2018 Intel Corporation.

 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
 the following conditions:

 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import os
import sys
import logging as log
from openvino.inference_engine import IENetwork, IECore, IEPlugin

        
class Network:
    """
    Load and configure inference plugins for the specified target devices 
    and performs synchronous and asynchronous modes for the specified infer requests.
    """

    def __init__(self):
        ### TODO: Initialize any class variables desired ###
        self.plugin = None
        self.network = None
        self.input_blob = None
        self.output_blob = None
        self.exec_network = None
        self.infer_request = None

    def load_model(self, model, CPU_EXTENSION, DEVICE, REQUEST_ID, PLUGIN=None):
        ### TODO: Load the model ###
        model_xml = model
        model_bin = os.path.splitext(model_xml)[0] + ".bin"
        print("Model XML: ",model)
        print("Model Bin: ", model_bin)
        
        if not PLUGIN:
            log.info("Initializing plugin for {} device...".format(DEVICE))
            self.plugin = IEPlugin(device=DEVICE)
        else:
            self.plugin = PLUGIN
        if CPU_EXTENSION and 'CPU' in DEVICE:
            self.plugin.add_cpu_extension(CPU_EXTENSION)
        self.network = IENetwork(model=model_xml, weights=model_bin)
        ### TODO: Check for supported layers ###
        all_layers_supported(self.plugin, self.network)
        
        
        ### TODO: Add any necessary extensions ###
        self.input_blob = next(iter(self.network.inputs))
        self.output_blob = next(iter(self.network.outputs))
        if REQUEST_ID == 0:
            self.exec_network = self.plugin.load(network=self.network)
        else:
            self.net_plugin = self.plugin.load(network=self.network, num_requests=REQUEST_ID)
        ### TODO: Return the loaded inference plugin ###
        ### Note: You may need to update the function parameters. ###
        
        return
    
    def get_input_shape(self):
        ### TODO: Return the shape of the input layer ###
        input_shapes = {}
        for inp in self.network.inputs:
            input_shapes[inp] = (self.network.inputs[inp].shape)
            print("input_shapes ", inp, input_shapes[inp])
        return input_shapes

    def exec_net(self, net_input, request_id):
        ### TODO: Start an asynchronous request ###
        ### TODO: Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        self.infer_request_handle = self.exec_network.start_async(request_id, inputs=net_input)
        return self.infer_request_handle
    
    def wait(self, request_id):
        ### TODO: Wait for the request to be complete. ###
        ### TODO: Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        wait_process = self.exec_network.requests[request_id].wait(-1)
        return wait_process

    def get_output(self, request_id, output=None):
        ### TODO: Extract and return the output results
        ### Note: You may need to update the function parameters. ###
        if output:
            res = self.infer_request_handle.outputs[output]
        else:
            res = self.exec_network.requests[request_id].outputs[self.output_blob]
        return res
    
    def clean(self):
        del self.net_plugin
        del self.plugin
        del self.net

def all_layers_supported(engine, network):
    layers_supported = engine.get_supported_layers(network)
    layers = network.layers.keys()

    not_supported_layers = [l for l in layers if l not in layers_supported]
    if len(not_supported_layers) != 0:
        log.error("Following layers are not supported by "
                  "the plugin for specified device {}".format(not_supported_layers))
        log.error("Please try to specify cpu extensions library path"
                  " in command line parameters using -l "
                  "or --cpu_extension command line argument")
        sys.exit(1)