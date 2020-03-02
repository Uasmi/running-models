import tensorflow as tf
import numpy as np
import cv2
import PIL
import random
import time
import threading 
#import matplotlib.pyplot as plt

# Audio imports
#-----------------------#
import pyaudio
import struct
import wave
from scipy.fftpack import fft 
#-----------------------#

def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the 
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it 
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name="prefix")
    return graph
def adjust_dynamic_range(data, drange_in, drange_out):
    if drange_in != drange_out:
        scale = (np.float32(drange_out[1]) - np.float32(drange_out[0])) / (np.float32(drange_in[1]) - np.float32(drange_in[0]))
        bias = (np.float32(drange_out[0]) - np.float32(drange_in[0]) * scale)
        data = data * scale + bias
    return data
def convert_to_pil_image(image, drange=[0,255]):
    assert image.ndim == 2 or image.ndim == 3
    if image.ndim == 3:
        if image.shape[0] == 1:
            image = image[0] # grayscale CHW => HW
        else:
            image = image.transpose(1, 2, 0) # CHW -> HWC

    image = adjust_dynamic_range(image, drange, [0,255])
    image = np.rint(image).clip(0, 255).astype(np.uint8)
    format = 'RGB' if image.ndim == 3 else 'L'
    #print (image, format)
    return PIL.Image.fromarray(image, format)
config = tf.ConfigProto()
config.intra_op_parallelism_threads = 4
config.inter_op_parallelism_threads = 0

def audio_processing(name):
    wf = wave.open('audiofiles/inst16b.wav','rb')
    Chunk = 1024
    p = pyaudio.PyAudio()
    stream = p.open(
        format = p.get_format_from_width(wf.getsampwidth()),
        channels = wf.getnchannels(),
        rate = wf.getframerate(),
        output = True,
        input = False,
        )

    audioData = wf.readframes(Chunk)
    while audioData != '':
        stream.write(audioData)
        audioData = wf.readframes(Chunk)
        dataInt = struct.unpack(str(4 * Chunk) + 'B', audioData)
        fftData = fft(dataInt)
        fftArray = np.abs(fftData[0: Chunk]) * 2 / (256 * Chunk)
        fftArray = fftArray[1: 10]

            #print (fftArray)
        if fftArray.max() > 0.7:
            print()

def network():
    graph = load_graph("dash/frozen_model.pb")

    x = graph.get_tensor_by_name('prefix/Gs/latents_in:0')
    x2 = graph.get_tensor_by_name('prefix/Gs/labels_in:0')
    y = graph.get_tensor_by_name('prefix/Gs/images_out:0')


    with tf.Session(graph=graph, config = config) as sess:
        # Note: we don't nee to initialize/restore anything
        # There is no Variables in this graph, only hardcoded constants 
        while True:
            start_time = time.time()

            latents = np.random.randn(1, 512).astype(np.float32)
            labels = np.zeros([latents.shape[0], 0], np.float32)
            y_out = sess.run(y, feed_dict = { x: latents, x2: labels})
            data = y_out[0, :, :, :]
            data = data * 127.5
            data = data + 127.5
            data = convert_to_pil_image(data)    
            cvimage = cv2.cvtColor(np.array(data), cv2.COLOR_RGB2BGR)
            cv2.imshow("image", cvimage)
            cv2.waitKey(1)
            print("--- %s seconds ---" % (time.time() - start_time))

def network_generate_images():
    graph = load_graph("dash/frozen_model.pb")

    x = graph.get_tensor_by_name('prefix/Gs/latents_in:0')
    x2 = graph.get_tensor_by_name('prefix/Gs/labels_in:0')
    y = graph.get_tensor_by_name('prefix/Gs/images_out:0')


    with tf.Session(graph=graph, config = config) as sess:
        # Note: we don't nee to initialize/restore anything
        # There is no Variables in this graph, only hardcoded constants 
        i = 0
        while i < 20:
            latents = np.random.randn(1, 512).astype(np.float32)
            labels = np.zeros([latents.shape[0], 0], np.float32)
            y_out = sess.run(y, feed_dict = { x: latents, x2: labels})
            data = y_out[0, :, :, :]
            data = data * 127.5
            data = data + 127.5
            data = convert_to_pil_image(data)    
            data.save('images/new%d' %i,'JPEG')
            j = 0
            while j < 20:
                latents += random.uniform(-0.5,0.5)
                y_out = sess.run(y, feed_dict = { x: latents, x2: labels})
                data = y_out[0, :, :, :]
                data = data * 127.5
                data = data + 127.5
                data = convert_to_pil_image(data)    
                data.save('images/new%d_morphed%d' %(i, j),'JPEG')
                j += 1
            i += 1


if __name__ == '__main__':
    
    network_generate_images()
    #start_time = time.time()
    
 
    #graph = load_graph("dash/frozen_model.pb")

    # We can verify that we can access the list of operations in the graph
    #for op in graph.get_operations():
        #print(op.name)
        
    #print(tf.contrib.graph_editor.get_tensors(tf.get_default_graph()))  
    # We access the input and output nodes 

    #x = graph.get_tensor_by_name('prefix/Gs/latents_in:0')
    #x2 = graph.get_tensor_by_name('prefix/Gs/labels_in:0')
    #y = graph.get_tensor_by_name('prefix/Gs/images_out:0')
    
    #----------------------------------------------------------------------------#
    '''
    wf = wave.open('audiofiles/inst16b.wav','rb')
    Chunk = 1024
    p = pyaudio.PyAudio()
    stream = p.open(
        format = p.get_format_from_width(wf.getsampwidth()),
        channels = wf.getnchannels(),
        rate = wf.getframerate(),
        output = True,
        input = False,
        )

    audioData = wf.readframes(Chunk)
    '''
    #----------------------------------------------------------------------------#
    # We launch a Session
    '''
    with tf.Session(graph=graph, config = config) as sess:
        # Note: we don't nee to initialize/restore anything
        # There is no Variables in this graph, only hardcoded constants 

        latents = np.random.randn(1, 512).astype(np.float32)
        labels = np.zeros([latents.shape[0], 0], np.float32)
        i = 0
        y_out = sess.run(y, feed_dict = { x: latents, x2: labels})
        #y_out = sess.run(enqueue_op)
    '''
    
    #x = threading.Thread(target=audio_processing, args=(1,))
    #x2 = threading.Thread(target=network)
    #x.start()
    #x2.start()
    # Audio stream
    #----------------------------------------------------------------------------#
    '''
        while audioData != '':
            stream.write(audioData)
            audioData = wf.readframes(Chunk)
            dataInt = struct.unpack(str(4 * Chunk) + 'B', audioData)
            fftData = fft(dataInt)
            fftArray = np.abs(fftData[0: Chunk]) * 2 / (256 * Chunk)
            fftArray = fftArray[1: 10]
            #print (fftArray)
            if fftArray.max() > 0.7:
                print("bang")        
    '''
    #----------------------------------------------------------------------------#
    # Old Version without audio
    #----------------------------------------------------------------------------#
    '''
        while True:
            #start_time = time.time()

            i += 1
            y_out = sess.run(y, feed_dict = { x: latents, x2: labels})
            data = y_out[0, :, :, :]
            data = data * 127.5
            data = data + 127.5
            data = convert_to_pil_image(data)


            #print("--- %s seconds ---" % (time.time() - start_time))
            latents += random.uniform(-0.5,0.5)
    '''
    #----------------------------------------------------------------------------#




        # I taught a neural net to recognise when a sum of numbers is bigger than 45
        # it should return False in this case

#data = np.rint(data).clip(0, 255).astype(np.uint8)
#print (data)
#im = PIL.Image.fromarray(data, 'RGB')
#data = convert_to_pil_image(data)

#data.save('test.jpg', 'JPEG')
#print(y_out.shape) # [[ False ]] Yay, it works!