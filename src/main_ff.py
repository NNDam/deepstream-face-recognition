#!/usr/bin/env python3

################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2019-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################

import sys
sys.path.append('../')
import gi
import configparser
gi.require_version('Gst', '1.0')
from gi.repository import GObject, Gst
from gi.repository import GLib
from ctypes import *
import sys
import math
import numpy as np
import time

from common.is_aarch_64 import is_aarch64
from common.bus_call import bus_call
from common.FPS import GETFPS
import pyds



MAX_DISPLAY_LEN=64

MUXER_OUTPUT_WIDTH=1920
MUXER_OUTPUT_HEIGHT=1080
MUXER_BATCH_TIMEOUT_USEC=4000000
TILED_OUTPUT_WIDTH=1280
TILED_OUTPUT_HEIGHT=720
GST_CAPS_FEATURES_NVMM="memory:NVMM"
OSD_PROCESS_MODE= 0
OSD_DISPLAY_TEXT= 1

PGIE = 1
SGIE = 2
TGIE = 3

PERSON_DETECTED = {}
fps_streams={}
RFACE_POOL=[]
RFACE_POOL_MAX = 4
start_time=0
end_time=0
is_first=True



def tiler_sink_pad_buffer_probe(pad,info,u_data):
    frame_number=0
    num_rects=0
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer ")
        return

    # Retrieve batch metadata from the gst_buffer
    # Note that pyds.gst_buffer_get_nvds_batch_meta() expects the
    # C address of gst_buffer as input, which is obtained with hash(gst_buffer)
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            # Note that l_frame.data needs a cast to pyds.NvDsFrameMeta
            # The casting is done by pyds.NvDsFrameMeta.cast()
            # The casting also keeps ownership of the underlying memory
            # in the C code, so the Python garbage collector will leave
            # it alone.
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break
        frame_number=frame_meta.frame_num
        num_rects = frame_meta.num_obj_meta
        l_obj=frame_meta.obj_meta_list
        while l_obj is not None:
            try:
                # Casting l_obj.data to pyds.NvDsObjectMeta
                #obj_meta=pyds.glist_get_nvds_object_meta(l_obj.data)
                obj_meta=pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break
            if obj_meta.unique_component_id==PGIE and obj_meta.class_id==0 and (obj_meta.object_id in PERSON_DETECTED)==False:
                print("new person {} detected in frame {}.".format(obj_meta.object_id, frame_number))
                PERSON_DETECTED[obj_meta.object_id] = [None, None]
            try: 
                l_obj=l_obj.next
            except StopIteration:
                break
            
        l_obj=frame_meta.obj_meta_list
        while l_obj is not None:
            try:
                # Casting l_obj.data to pyds.NvDsObjectMeta
                #obj_meta=pyds.glist_get_nvds_object_meta(l_obj.data)
                obj_meta=pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break
            if obj_meta.unique_component_id==SGIE and obj_meta.class_id==0:
                # print(obj_meta.object_id, "  ", obj_meta.parent.object_id)
                if obj_meta.parent.object_id in PERSON_DETECTED:
                    if PERSON_DETECTED[obj_meta.parent.object_id][0] is None:
                        print("face of person-{} detected in frame{}".format(obj_meta.parent.object_id, frame_number))
                        PERSON_DETECTED[obj_meta.parent.object_id][0] = frame_number
                        
                        l_user = obj_meta.obj_user_meta_list
                        while l_user is not None:
                            try:
                                # Note that l_user.data needs a cast to pyds.NvDsUserMeta
                                # The casting is done by pyds.NvDsUserMeta.cast()
                                # The casting also keeps ownership of the underlying memory
                                # in the C code, so the Python garbage collector will leave
                                # it alone
                                user_meta=pyds.NvDsUserMeta.cast(l_user.data) 
                            except StopIteration:
                                break
                            
                            # Check data type of user_meta 
                            if(user_meta and user_meta.base_meta.meta_type==pyds.NvDsMetaType.NVDSINFER_TENSOR_OUTPUT_META): 
                                try:
                                    tensor_meta = pyds.NvDsInferTensorMeta.cast(user_meta.user_meta_data)
                                except StopIteration:
                                    break
                                
                                layer = pyds.get_nvds_LayerInfo(tensor_meta, 0)
                                output = []
                                for i in range(512):
                                    output.append(pyds.get_detections(layer.buffer, i))
                                
                                res = np.reshape(output,(512,-1))
                                norm=np.linalg.norm(res)
                                normal_array = res / norm
                                PERSON_DETECTED[obj_meta.parent.object_id][1] = normal_array
                                print("get facial features of person {}".format(obj_meta.parent.object_id))
                                try:
                                    l_user=l_user.next
                                except StopIteration:
                                    break 
                                   
            try: 
                l_obj=l_obj.next
            except StopIteration:
                break
        try:
            l_frame=l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK

def osd_sink_pad_buffer_probe(pad,info,u_data):
    frame_number=0
    num_rects=0
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer ")
        return

    # Retrieve batch metadata from the gst_buffer
    # Note that pyds.gst_buffer_get_nvds_batch_meta() expects the
    # C address of gst_buffer as input, which is obtained with hash(gst_buffer)
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            # Note that l_frame.data needs a cast to pyds.NvDsFrameMeta
            # The casting is done by pyds.NvDsFrameMeta.cast()
            # The casting also keeps ownership of the underlying memory
            # in the C code, so the Python garbage collector will leave
            # it alone.
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break
    
        l_obj=frame_meta.obj_meta_list
        while l_obj is not None:
            try:
                # Casting l_obj.data to pyds.NvDsObjectMeta
                #obj_meta=pyds.glist_get_nvds_object_meta(l_obj.data)
                obj_meta=pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break
            # obj_meta.rect_params.border_color.set(0.0, 0.0, 1.0, 0.0)
            # print(obj_meta.class_id," ", obj_meta.confidence)
            try: 
                l_obj=l_obj.next
            except StopIteration:
                break

        # Get frame rate through this probe
        fps_streams["stream{0}".format(frame_meta.pad_index)].get_fps()
        try:
            l_frame=l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK

def sgie_sink_pad_buffer_probe(pad,info,u_data):
    '''
    '''
    global RFACE_POOL_MAX, RFACE_POOL
    global start_time, end_time
    global is_first
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer ")
        return
    
    # Retrieve batch metadata from the gst_buffer
    # Note that pyds.gst_buffer_get_nvds_batch_meta() expects the
    # C address of gst_buffer as input, which is obtained with hash(gst_buffer)
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        if is_first==True:
            start_time=time.time()
            is_first = False
        if (end_time - start_time)>=1:
            RFACE_POOL.clear()
            start_time = end_time
        try:
            # Note that l_frame.data needs a cast to pyds.NvDsFrameMeta
            # The casting is done by pyds.NvDsFrameMeta.cast()
            # The casting also keeps ownership of the underlying memory
            # in the C code, so the Python garbage collector will leave
            # it alone.
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            # print(start_time, "  ", end_time)
        except StopIteration:
            break
        l_obj=frame_meta.obj_meta_list
        while l_obj is not None:
            remove_flag = 1
            try:
                # Casting l_obj.data to pyds.NvDsObjectMeta
                obj_meta=pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break
            #if obj_meta.class_id!=0:
                #remove_flag = 0      
            if obj_meta.class_id ==0 and obj_meta.unique_component_id == PGIE and (obj_meta.rect_params.width < 20 or obj_meta.rect_params.height < 20):
                remove_flag =1
            elif obj_meta.class_id==0 and (obj_meta.object_id in PERSON_DETECTED) and (PERSON_DETECTED[obj_meta.object_id][0] is not None):
                if obj_meta.object_id in RFACE_POOL:
                    RFACE_POOL.remove(obj_meta.object_id)
                remove_flag = 1
                
            elif obj_meta.class_id==0 and (obj_meta.object_id in PERSON_DETECTED)==False:
                if obj_meta.object_id in RFACE_POOL:
                    remove_flag = 1
                if obj_meta.object_id not in RFACE_POOL:
                    if len(RFACE_POOL) >= RFACE_POOL_MAX:
                        remove_flag = 1
                    else:
                        RFACE_POOL.append(obj_meta.object_id)
                        #print(RFACE_POOL)
                        remove_flag = 0 
            try:
                l_obj=l_obj.next
                if remove_flag:
                    pyds.nvds_remove_obj_meta_from_frame(frame_meta, obj_meta)
            except StopIteration:
                break
            
        try:
            end_time = time.time()
            l_frame=l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK


def main(args):
    # Check input arguments
    if len(args) < 2:
        sys.stderr.write("usage: %s <uri1> [uri2] ... [uriN]\n" % args[0])
        sys.exit(1)

    for i in range(0,len(args)-1):
        fps_streams["stream{0}".format(i)]=GETFPS(i)
    print(fps_streams)
    number_sources=len(args)-1

    # Standard GStreamer initialization
    GObject.threads_init()
    Gst.init(None)

    # Create gstreamer elements */
    # Create Pipeline element that will form a connection of other elements
    print("Creating Pipeline \n ")
    pipeline = Gst.Pipeline()
    is_live = False

    if not pipeline:
        sys.stderr.write(" Unable to create Pipeline \n")
    print("Creating streamux \n ")

    # Create nvstreammux instance to form batches from one or more sources.
    streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
    if not streammux:
        sys.stderr.write(" Unable to create NvStreamMux \n")

    pipeline.add(streammux)
    for i in range(number_sources):
        print("Creating source_bin ",i," \n ")
        uri_name=args[i+1]
        if uri_name.find("rtsp://") == 0 :
            is_live = True
        source_bin=create_source_bin(i, uri_name)
        if not source_bin:
            sys.stderr.write("Unable to create source bin \n")
        pipeline.add(source_bin)
        padname="sink_%u" %i
        sinkpad= streammux.get_request_pad(padname) 
        if not sinkpad:
            sys.stderr.write("Unable to create sink pad bin \n")
        srcpad=source_bin.get_static_pad("src")
        if not srcpad:
            sys.stderr.write("Unable to create src pad bin \n")
        srcpad.link(sinkpad)
    queue1=Gst.ElementFactory.make("queue","queue1")
    queue2=Gst.ElementFactory.make("queue","queue2")
    queue3=Gst.ElementFactory.make("queue","queue3")
    queue4=Gst.ElementFactory.make("queue","queue4")
    queue5=Gst.ElementFactory.make("queue","queue5")
    queue6=Gst.ElementFactory.make("queue","queue6")
    queue7=Gst.ElementFactory.make("queue","queue7")
    queue8=Gst.ElementFactory.make("queue","queue8")
    pipeline.add(queue1)
    pipeline.add(queue2)
    pipeline.add(queue3)
    pipeline.add(queue4)
    pipeline.add(queue5)
    pipeline.add(queue6)
    pipeline.add(queue7)
    pipeline.add(queue8)
    print("Creating Pgie \n ")
    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    if not pgie:
        sys.stderr.write(" Unable to create pgie \n")
    print("Creating sgie \n ")
    sgie = Gst.ElementFactory.make("nvinfer", "Secondary-inference")
    if not sgie:
        sys.stderr.write(" Unable to create sgie \n")

    print("Creating tgie \n ")
    tgie = Gst.ElementFactory.make("nvinfer", "Third-inference")
    if not tgie:
        sys.stderr.write(" Unable to create tgie \n")
    print("Creating tracker \n ")
    tracker = Gst.ElementFactory.make("nvtracker", "tracker")
    if not tracker:
        sys.stderr.write(" Unable to create tracker \n")
    print("Creating tiler \n ")
    tiler=Gst.ElementFactory.make("nvmultistreamtiler", "nvtiler")
    if not tiler:
        sys.stderr.write(" Unable to create tiler \n")
    print("Creating nvvidconv \n ")
    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
    if not nvvidconv:
        sys.stderr.write(" Unable to create nvvidconv \n")
    print("Creating nvosd \n ")
    nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
    if not nvosd:
        sys.stderr.write(" Unable to create nvosd \n")
    nvosd.set_property('process-mode',OSD_PROCESS_MODE)
    nvosd.set_property('display-text',OSD_DISPLAY_TEXT)
    if(is_aarch64()):
        print("Creating transform \n ")
        transform=Gst.ElementFactory.make("nvegltransform", "nvegl-transform")
        if not transform:
            sys.stderr.write(" Unable to create transform \n")


    print("Creating FileSink \n")
    nvvidconv1 = Gst.ElementFactory.make("nvvideoconvert", "convertor1")
    videoconvert = Gst.ElementFactory.make("videoconvert", "video-converter")
    x264enc = Gst.ElementFactory.make("x264enc", "h264 encoder")
    qtmux = Gst.ElementFactory.make("qtmux", "muxer")

    sink = Gst.ElementFactory.make("filesink", "filesink")
    sink.set_property("location", "output.mp4")
        
    if not sink:
        sys.stderr.write(" Unable to create egl sink \n")

    if is_live:
        print("Atleast one of the sources is live")
        streammux.set_property('live-source', 1)

    streammux.set_property('width', 1920)
    streammux.set_property('height', 1080)
    streammux.set_property('batch-size', number_sources)
    streammux.set_property('batched-push-timeout', 4000000)

    pgie.set_property('config-file-path', "configs/config_scrfd_fullframe.txt")
    pgie_batch_size=pgie.get_property("batch-size")
    if(pgie_batch_size != number_sources):
        print("WARNING: Overriding infer-config batch-size",pgie_batch_size," with number of sources ", number_sources," \n")
        pgie.set_property("batch-size",number_sources)

    sgie.set_property('config-file-path', "configs/config_face_embedding_fullframe.txt")

    tiler_rows=int(math.sqrt(number_sources))
    tiler_columns=int(math.ceil((1.0*number_sources)/tiler_rows))
    tiler.set_property("rows",tiler_rows)
    tiler.set_property("columns",tiler_columns)
    tiler.set_property("width", TILED_OUTPUT_WIDTH)
    tiler.set_property("height", TILED_OUTPUT_HEIGHT)
    sink.set_property("qos",0)
    sink.set_property("sync",0)

    print("Adding elements to Pipeline \n")
    pipeline.add(pgie)
    pipeline.add(sgie)
    # pipeline.add(tracker)
    pipeline.add(tiler)
    pipeline.add(nvvidconv)
    pipeline.add(nvosd)
    pipeline.add(nvvidconv1)
    pipeline.add(videoconvert)
    pipeline.add(x264enc)
    pipeline.add(qtmux)
    pipeline.add(sink)

    print("Linking elements in the Pipeline \n")
    streammux.link(queue1)
    queue1.link(pgie)
    pgie.link(queue2)
    # queue2.link(tracker)
    # tracker.link(queue3)
    queue2.link(sgie)
    sgie.link(queue4)
    
    queue4.link(tiler)
    tiler.link(queue6)
    queue6.link(nvvidconv)
    nvvidconv.link(queue7)
    queue7.link(nvosd)
    
    nvosd.link(nvvidconv1)
    nvvidconv1.link(videoconvert) 
    videoconvert.link(x264enc)
    x264enc.link(qtmux)
    qtmux.link(sink)     

    # create an event loop and feed gstreamer bus mesages to it
    loop = GObject.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect ("message", bus_call, loop)
    
    # tiler_sink_pad = tiler.get_static_pad("sink")
    # if not tiler_sink_pad:
    #     sys.stderr.write(" Unable to get sink pad of tiler \n")
    # else:
    #     i =1
    #     tiler_sink_pad.add_probe(Gst.PadProbeType.BUFFER, tiler_sink_pad_buffer_probe, 0)
    
    # sgie_sink_pad = queue3.get_static_pad("sink")
    # if not sgie_sink_pad:
    #     sys.stderr.write(" Unable to get sink pad of tiler \n")
    # else:
    #     i=2
    #     sgie_sink_pad.add_probe(Gst.PadProbeType.BUFFER, sgie_sink_pad_buffer_probe, 0)


    osd_sink_pad=nvosd.get_static_pad("sink")
    if not osd_sink_pad:
        sys.stderr.write(" Unable to get sink pad of osd \n")
    else:
        i=3
        osd_sink_pad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe, 0)

    # List the sources
    print("Now playing...")
    for i, source in enumerate(args):
        if (i != 0):
            print(i, ": ", source)

    print("Starting pipeline \n")
    # start play back and listed to events      
    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except:
        pass
    # cleanup
    print("Exiting app\n")
    pipeline.set_state(Gst.State.NULL)

def cb_newpad(decodebin, decoder_src_pad,data):
    print("In cb_newpad\n")
    caps=decoder_src_pad.get_current_caps()
    gststruct=caps.get_structure(0)
    gstname=gststruct.get_name()
    source_bin=data
    features=caps.get_features(0)

    # Need to check if the pad created by the decodebin is for video and not
    # audio.
    print("gstname=",gstname)
    if(gstname.find("video")!=-1):
        # Link the decodebin pad only if decodebin has picked nvidia
        # decoder plugin nvdec_*. We do this by checking if the pad caps contain
        # NVMM memory features.
        print("features=",features)
        if features.contains("memory:NVMM"):
            # Get the source bin ghost pad
            bin_ghost_pad=source_bin.get_static_pad("src")
            if not bin_ghost_pad.set_target(decoder_src_pad):
                sys.stderr.write("Failed to link decoder src pad to source bin ghost pad\n")
        else:
            sys.stderr.write(" Error: Decodebin did not pick nvidia decoder plugin.\n")

def decodebin_child_added(child_proxy,Object,name,user_data):
    print("Decodebin child added:", name, "\n")
    if(name.find("decodebin") != -1):
        Object.connect("child-added",decodebin_child_added,user_data)

    # if "source" in name:
    #     Object.set_property("drop-on-latency", True)

def create_source_bin(index,uri):
    print("Creating source bin")

    # Create a source GstBin to abstract this bin's content from the rest of the
    # pipeline
    bin_name="source-bin-%02d" %index
    print(bin_name)
    nbin=Gst.Bin.new(bin_name)
    if not nbin:
        sys.stderr.write(" Unable to create source bin \n")

    # Source element for reading from the uri.
    # We will use decodebin and let it figure out the container format of the
    # stream and the codec and plug the appropriate demux and decode plugins.
    uri_decode_bin=Gst.ElementFactory.make("uridecodebin", "uri-decode-bin")
    if not uri_decode_bin:
        sys.stderr.write(" Unable to create uri decode bin \n")
    # We set the input uri to the source element
    uri_decode_bin.set_property("uri",uri)
    # Connect to the "pad-added" signal of the decodebin which generates a
    # callback once a new pad for raw data has beed created by the decodebin
    uri_decode_bin.connect("pad-added",cb_newpad,nbin)
    uri_decode_bin.connect("child-added",decodebin_child_added,nbin)

    # We need to create a ghost pad for the source bin which will act as a proxy
    # for the video decoder src pad. The ghost pad will not have a target right
    # now. Once the decode bin creates the video decoder and generates the
    # cb_newpad callback, we will set the ghost pad target to the video decoder
    # src pad.
    Gst.Bin.add(nbin,uri_decode_bin)
    bin_pad=nbin.add_pad(Gst.GhostPad.new_no_target("src",Gst.PadDirection.SRC))
    if not bin_pad:
        sys.stderr.write(" Failed to add ghost pad in source bin \n")
        return None
    return nbin

if __name__ == '__main__':
    sys.exit(main(sys.argv))

