"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
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
import time
import socket
import json
import cv2

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network
import resource
def using(point=""):
    usage=resource.getrusage(resource.RUSAGE_SELF)
    return '''%s: usertime=%s systime=%s mem=%s mb
           '''%(point,usage[0],usage[1],
                usage[2]/1024.0 )

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60


def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default="/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so",
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser


def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)
    return client


def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    # Initialise the class
    inference_network = Network()
    # Set Probability threshold for detections
    probability_threshold = args.prob_threshold
    model = args.model
    
    DEVICE = args.device
    CPU_EXTENSION = args.cpu_extension
    single_image_mode = False
    last_count = 0
    total_count = 0
    start_time = 0
    ### TODO: Load the model through `infer_network` ###
    request_id=0
    inference_network.load_model(model, CPU_EXTENSION, DEVICE,request_id)
    network_shape = inference_network.get_input_shape()
    
    ### TODO: Handle the input stream ###
    if args.input == 'CAM':
        input_validated = 0
    elif args.input.endswith('.jpg') or args.input.endswith('.bmp') or args.input.endswith('.png'):
        single_image_mode = True
        input_validated = args.input
    else:
        input_validated = args.input
        assert os.path.isfile(args.input), "file doesn't exist"
        
    capture = cv2.VideoCapture(input_validated)
    capture.open(input_validated)
    
    
    width = int(capture.get(3))
    height = int(capture.get(4))
    
    input_shape = network_shape['image_tensor']
    
    
    ### TODO: Loop until stream is over ###
    
    counter_prob_hit = 0
    total_prob_hit = 0
    counter_prob_miss = 0
    total_prob_miss = 0
    max_prob = 0
    max_time = 0
    
    while capture.isOpened():
        ### TODO: Read from the video capture ###
        flag, frame = capture.read()
        if not flag:
            break
        ### TODO: Pre-process the image as needed ###
        image = cv2.resize(frame, (input_shape[3], input_shape[2]))
        image_p = image.transpose((2, 0, 1))
        image_p = image_p.reshape(1, *image_p.shape)
        ### TODO: Start asynchronous inference for specified request ###
        network_input = {'image_tensor': image_p,'image_info': image_p.shape[1:]}
        duration_report = None
        start = time.time()
        inference_network.exec_net(network_input, request_id)
        ### TODO: Wait for the result ###
        if inference_network.wait(request_id) == 0:
            ### TODO: Get the results of the inference request ###
            end = time.time()
            log.warn("Inference time")
            log.warn( str((end - start)*1000))
            log.warn(using("after"))
            network_output = inference_network.get_output(request_id)
            if end - start > max_time:
                max_time = (end - start)*1000
            log.warn("Max Time:")
            log.warn(max_time)
            ### TODO: Extract any desired stats from the results ###
            probs = network_output[0, 0, :, 2]
            
            current_count = 0
            for i, p in enumerate(probs):
                if p > probability_threshold:
                    box = network_output[0, 0, i, 3:]
                    p1 = (int(box[0] * width), int(box[1] * height))
                    p2 = (int(box[2] * width), int(box[3] * height))
                    frame = cv2.rectangle(frame, p1, p2, (0, 55, 255), 1)
                    current_count = current_count + 1
                    counter_prob_hit += 1
                    total_prob_hit += p
                else:
                    if p > probability_threshold - 0.1:
                        counter_prob_miss += 1
                        total_prob_miss += p
                
            ### TODO: Calculate and send relevant information on ###
            ### current_count, total_count and duration to the MQTT server ###
            ### Topic "person": keys of "count" and "total" ###
            ### Topic "person/duration": key of "duration" ###
            if current_count > last_count:
                start_time = time.time()
                total_count = total_count + current_count - last_count
                client.publish("person", json.dumps({"total": total_count}))
            if current_count < last_count:
                duration = int(time.time() - start_time)
                # Publish messages to the MQTT server
                client.publish("person/duration",
                               json.dumps({"duration": duration}))
                
            client.publish("person", json.dumps({"count": current_count}))
            last_count = current_count
            
            
            if counter_prob_hit != 0:
                log.warn("Average prob hit: ")
                log.warn(str(total_prob_hit/counter_prob_hit ))
            if counter_prob_miss != 0:
                log.warn("Average prob miss less than 0.1 threshold: ")
                log.warn(str(total_prob_miss/counter_prob_miss))
                
 
        ### TODO: Send the frame to the FFMPEG server ###
        frame = cv2.resize(frame, (768, 432))
        sys.stdout.buffer.write(frame)
        sys.stdout.flush()
        ### TODO: Write an output image if `single_image_mode` ###
        if single_image_mode:
            cv2.imwrite('output_image.jpg', frame)
    cap.release()
    cv2.destroyAllWindows()
    client.disconnect()
    infer_network.clean()

def main():
    """
    Load the network and parse the output.

    :return: None
    
    cmd to run
    
    python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m frozen_inference_graph.xml -d CPU -pt 0.4 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm
        python main1.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m person-detection-retail-0013.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.4 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm
        sudo ./downloader.py --name person-detection-retail-0013 -o /home/workspace
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    log.warn(using("before"))
    main()
