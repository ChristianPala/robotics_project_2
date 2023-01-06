from PIL import Image
from matplotlib.colors import hsv_to_rgb
import numpy as np
import cv2
import onnx
from onnxruntime import get_device
import onnxruntime.backend as backend
from torchvision import transforms

import json
import open3d as o3d
import pyrealsense2 as rs

import roboticstoolbox as rtb
import spatialmath as sm

device = get_device()
model = onnx.load('fcn-resnet101-11/fcn-resnet101-11.onnx')
classes = [line.rstrip('\n') for line in open('fcn-resnet101-11/voc_classes.txt')]
num_classes = len(classes)
toTens = transforms.ToTensor()

def get_palette():
    # prepare and return palette
    palette = [0] * num_classes * 3

    for hue in range(num_classes):
        if hue == 0: # Background color
            colors = (0, 0, 0)
        else:
            colors = hsv_to_rgb((hue / num_classes, 0.75, 0.75))

        for i in range(3):
            palette[hue * 3 + i] = int(colors[i] * 255)

    return palette


def colorize(labels):
    # generate colorized image from output labels and color palette
    result_img = Image.fromarray(labels).convert('P', colors=num_classes)
    result_img.putpalette(get_palette())
    return np.array(result_img.convert('RGB'))


def visualize_output(image, output):
    assert(image.shape[0] == output.shape[1] and image.shape[1] == output.shape[2]) # Same height and width
    assert(output.shape[0] == num_classes)

    # get classification labels
    raw_labels = np.argmax(output, axis=0).astype(np.float)

    # comput confidence score
    confidence = float(np.max(output, axis=0).mean())

    # generate segmented image
    result_img = colorize(raw_labels.astype(np.float32))

    # generate blended image
    if image.dtype == 'float32':
        blended_img = cv2.addWeighted(image[:, :, ::-1], 0.5, result_img.astype(np.float32), 0.5, 0)
    else:
        blended_img = cv2.addWeighted(image[:, :, ::-1], 0.5, result_img, 0.5, 0)

    #result_img = Image.fromarray(result_img)
    #blended_img = Image.fromarray(blended_img)

    return confidence, result_img, blended_img, raw_labels

def onnx_segmentation(image_array):

    inputs = []
    inputs_num = len(image_array)
    for i in range(inputs_num):
        img_tens = toTens(image_array[i])
        preprocess = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        img_tens = preprocess(img_tens)
        temp = img_tens.numpy()
        inputs.append(temp[np.newaxis, :, :, :])

    #### Run the model on the backend
    outputs = list(backend.run(model, inputs, device))

    return outputs



def open_camera(filename) -> o3d.t.io.RealSenseSensor:
    camera = o3d.t.io.RealSenseSensor()

    with open(filename) as cf:
        config = o3d.t.io.RealSenseSensorConfig(json.load(cf))
    camera.init_sensor(config)

    return camera


def get_depth_intrinsics() -> o3d.camera.PinholeCameraIntrinsic:

    pipeline = rs.pipeline()
    pipeline_config = rs.config()

    width = 1280
    height = 720
    pipeline_config.enable_stream(rs.stream.depth, int(width), int(height), rs.format.z16, 30)

    cfg = pipeline.start(pipeline_config)  # Start pipeline and get the configuration it found

    profile = cfg.get_stream(rs.stream.depth)  # Fetch stream profile for color stream
    data = profile.as_video_stream_profile().get_intrinsics()

    pipeline.stop()

    intrinsics = o3d.camera.PinholeCameraIntrinsic(data.width, data.height, data.fx, data.fy, data.ppx, data.ppy)
    return intrinsics


def get_color_intrinsics() -> o3d.camera.PinholeCameraIntrinsic:

    pipeline = rs.pipeline()
    pipeline_config = rs.config()

    width = 1280
    height = 720
    pipeline_config.enable_stream(rs.stream.color, int(width), int(height), rs.format.rgb8, 30)

    cfg = pipeline.start(pipeline_config)  # Start pipeline and get the configuration it found

    profile = cfg.get_stream(rs.stream.color)  # Fetch stream profile for color stream
    data = profile.as_video_stream_profile().get_intrinsics()

    pipeline.stop()

    intrinsics = o3d.camera.PinholeCameraIntrinsic(data.width, data.height, data.fx, data.fy, data.ppx, data.ppy)
    return intrinsics


def build_transform_matrix(rotation: np.ndarray, translation_vector: np.ndarray) -> np.ndarray:
    """
    Given as inputs rotation_matrix [3x3] or rotation_vector [3,1] or [1,3]
    builds a 4x4 and the translation vector, builds the 4x4 transform matrix

    :param rotation: rotation component of the transform
    :param translation_vector: translation component of the transform
    :return: transform matrix
    """

    if rotation.shape == (1, 3) or rotation.shape == (3, 1):
        rotation, _ = cv2.Rodrigues(rotation)

    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = np.reshape(translation_vector, (3,))

    return matrix

def rigid_registration(model, acq):

    # Input: pnt_model [nx3], pnt_acq [nx3],
    # Output: rotation matrix R [3x3], translation vector [3x1], scale factor s

    model_bar = list(np.mean(model, 0))
    model_dev = model - model_bar

    acq_bar = list(np.mean(acq, 0))
    acq_dev = acq - acq_bar

    # mm = np.matmul(acq_dev.T,model_dev)

    scale = np.sqrt(np.sum(np.power(acq_dev, 2)) / np.sum(np.power(model_dev, 2)))
    m = acq_dev.T @ model_dev

    U, S, Vh = np.linalg.svd(m, full_matrices=True)

    R = U @ [[1, 0, 0], [0, 1, 0], [0, 0, np.sign(np.linalg.det(U @ Vh))]] @ Vh
    T = acq_bar - scale * R @ model_bar

    return R, T, scale

def LRMate200iD4S_gen():
    lrmate_DH = rtb.DHRobot(
        [
            rtb.RevoluteDH(alpha=-np.pi/2),
            rtb.RevoluteDH(a=0.260, alpha=np.pi, offset=-np.pi/2),
            rtb.RevoluteDH(a=-0.02, alpha=np.pi/2, offset=-np.pi),
            rtb.RevoluteDH(d=-0.290, alpha=-np.pi/2),
            rtb.RevoluteDH(alpha=-np.pi/2, offset=-np.pi),
            rtb.RevoluteDH(d=-0.07, alpha=-np.pi)
        ], name="LRMate200iD4s")
    #lrmate_DH.tool = sm.SE3(0, 0, 0.115)
    #lrmate_DH.base = sm.SE3(0, 0, 0.33)

    return lrmate_DH


def joints_fanuc2corke(q):
    if q.ndim == 1:
        q = [q]
    q_adj = q
    q_adj[:, 2] = q[:, 2] + q[:, 1]
    q_adj = q_adj/180*np.pi
    return q_adj
