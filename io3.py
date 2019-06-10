from environment_data_types import AVATAR_KEYS, ACTION_KEYS, OBJECT
import numpy as np
import json
import base64
import copy
from PIL import Image

try:
    from StringIO import StringIO
except ImportError:
    from io import BytesIO as StringIO


def decode_xyz_vector(vec):
    vec = json.loads(vec)
    return [vec["x"], vec["y"], vec["z"]]


def decode_xyzw_vector(vec):
    vec = json.loads(vec)
    return [vec["x"], vec["y"], vec["z"], vec["w"]]


def decode_info(info):
    try:
        info = json.loads(info)
    except TypeError:
        info = json.loads(info.decode("utf-8"))
    for k in AVATAR_KEYS:
        info[k] = decode_xyz_vector(info[k])
    for obj in info["observed_objects"]:
        for prop in [
            OBJECT.POSITION,
            OBJECT.EXTENTS,
            OBJECT.CENTER,
            OBJECT.PARTICLE_INFO,
        ]:
            obj[prop] = decode_xyz_vector(obj[prop])
        obj[OBJECT.ROTATION] = decode_xyzw_vector(obj[OBJECT.ROTATION])
    return info


def decode_image(image):
    if isinstance(image, np.ndarray):
        return image
    else:
        return np.asarray(Image.open(StringIO(image)).convert("RGB"))


def decode_particles(particles):
    return np.frombuffer(particles, dtype=np.float32)


def decode_static_particles(particles):
    particles = base64.b64decode(particles)
    particles = np.frombuffer(particles, dtype=np.float32)
    return np.reshape(particles, [-1, 7])


def add_static_particle_info(data):
    info = data["info"]
    if "static_particles" in info:
        data["static_particles"] = copy.deepcopy(info["static_particles"])
        data["static_particles"][0] = decode_static_particles(
            info["static_particles"][0]
        )
    return data


def decode_segmentation_image(image):
    return image[:, :, 0] * 256 ** 2 + image[:, :, 1] * 256 + image[:, :, 2]


def decode_depth_image(image):
    # return (image[:,:,0] * 256.0 + image[:,:,1] + image[:,:,2] / 256.0) / 1000.0
    return (
        (image[:, :, 0] * 256.0 * 256.0 + image[:, :, 1] * 256.0 + image[:, :, 2])
        / (256.0 * 256.0 * 256.0)
        * 30.3
    )


def recv_with_timeout(sock, poller, timeout=30):
    if poller.poll(timeout * 1000):
        return sock.recv()
    else:
        raise IOError("Did not receive message within timeout")


def receive_message(sock, poller, n_cameras, shaders, is_send_particles, timeout):
    msg = {}
    msg["n_cameras"] = n_cameras
    msg["shaders"] = shaders
    msg["is_send_particles"] = is_send_particles
    # Environment information
    msg["info"] = recv_with_timeout(sock, poller, timeout)
    # Particle information (if requested)
    if is_send_particles:
        msg["particles"] = recv_with_timeout(sock, poller, timeout)
    # Iterate over all cameras and shaders
    for cam in range(n_cameras):
        for shader in shaders:
            # Handle set of images per camera
            shadercam_name = shader + str(cam + 1)
            assert shadercam_name not in msg, (
                "duplicate message name %s" % shadercam_name
            )
            msg[shadercam_name] = recv_with_timeout(sock, poller, timeout)
    return msg


def decode_message(msg):
    # Decode info
    msg["info"] = decode_info(msg["info"])
    msg["worldinfo"] = json.dumps(msg["info"])
    # Decode particles
    if msg["is_send_particles"]:
        msg["particles"] = decode_particles(msg["particles"])
        msg = add_static_particle_info(msg)
    # Decode images
    for cam in range(msg["n_cameras"]):
        for shader in msg["shaders"]:
            # Handle set of images per camera
            shadercam_name = shader + str(cam + 1)
            msg[shadercam_name] = decode_image(msg[shadercam_name])
            # Depth and object images are decoded from base 256
            if "depths" in shadercam_name:
                msg["numeric_" + shadercam_name] = decode_depth_image(
                    msg[shadercam_name]
                )
            if "objects" in shadercam_name:
                msg["numeric_" + shadercam_name] = decode_segmentation_image(
                    msg[shadercam_name]
                )
    msg["valid"] = True
    return msg


def handle_message(sock, poller, n_cameras, shaders, is_send_particles, timeout):
    msg = receive_message(sock, poller, n_cameras, shaders, is_send_particles, timeout)
    return decode_message(msg)


def send_message(sock, msg):
    assert "action_type" in msg, "Action has no action type"
    return sock.send_json(msg)


def is_valid_action_message(msg):
    return all(k in msg for k in ACTION_KEYS)
