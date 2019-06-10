"""
Environment object for interaction with 3World
"""
from __future__ import print_function
import time
import numpy as np
import zmq
import copy
import json
from PIL import Image
from tdw_client import TDWClient
import signal
import os
from collections import namedtuple
import old_assets_query_result_example
import io3
import h5py
from writer import HDF5Writer
from environment_data_types import Attribute

try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO


synset_for_table = [[u"n04379243"]]

rollie_synsets = [
    [u"n03991062"],
    [u"n02880940"],
    [u"n02946921"],
    [u"n02876657"],
    [u"n03593526"],
]

other_vaguely_stackable_synsets = [
    [u"n03207941"],
    [u"n04004475"],
    [u"n02958343"],
    [u"n03001627"],
    [u"n04256520"],
    [u"n04330267"],
    [u"n03593526"],
    [u"n03761084"],
    [u"n02933112"],
    [u"n03001627"],
    [u"n04468005"],
    [u"n03691459"],
    [u"n02946921"],
    [u"n03337140"],
    [u"n02924116"],
    [u"n02801938"],
    [u"n02828884"],
    [u"n03001627"],
    [u"n04554684"],
    [u"n02808440"],
    [u"n04460130"],
    [u"n02843684"],
    [u"n03928116"],
]

shapenet_inquery = {
    "type": "shapenetremat",
    "has_texture": True,
    "version": 0,
    "complexity": {"$exists": True},
    "center_pos": {"$exists": True},
    "boundb_pos": {"$exists": True},
    "isLight": {"$exists": True},
    "anchor_type": {"$exists": True},
    "aws_address": {"$exists": True},
}

dosch_inquery = {
    "type": "dosch",
    "has_texture": True,
    "version": 1,
    "complexity": {"$exists": True},
    "center_pos": {"$exists": True},
    "boundb_pos": {"$exists": True},
    "isLight": {"$exists": True},
    "anchor_type": {"$exists": True},
    "aws_address": {"$exists": True},
}


default_keys = [
    "boundb_pos",
    "isLight",
    "anchor_type",
    "aws_address",
    "complexity",
    "center_pos",
]


table_query = copy.deepcopy(shapenet_inquery)
table_query["synset"] = {"$in": synset_for_table}
rolly_query = copy.deepcopy(shapenet_inquery)
rolly_query["synset"] = {"$in": rollie_synsets}
other_reasonables_query = copy.deepcopy(shapenet_inquery)
other_reasonables_query["synset"] = {"$in": other_vaguely_stackable_synsets}

query_dict = {
    "SHAPENET": shapenet_inquery,
    "ROLLY": rolly_query,
    "TABLE": table_query,
    "OTHER_STACKABLE": other_reasonables_query,
}


def query_results_to_unity_data(query_results, scale, mass, var=0.01, seed=0):
    item_list = []
    for i in range(len(query_results)):
        res = query_results[i]
        item = {}
        item["type"] = res["type"]
        item["has_texture"] = res["has_texture"]
        item["center_pos"] = res["center_pos"]
        item["boundb_pos"] = res["boundb_pos"]
        item["isLight"] = res["isLight"]
        item["anchor_type"] = res["anchor_type"]
        # print(res['aws_address'])
        item["aws_address"] = res["aws_address"]
        item["mass"] = mass
        item["scale"] = {
            "option": "Absol_size",
            "scale": scale,
            "var": var,
            "seed": seed,
            "apply_to_inst": True,
        }
        item["_id_str"] = str(res["_id"])
        item_list.append(item)
    #        for item in item_list:
    #            for k, v in item.items():
    #                print(k)
    #                print(v)
    #        raise Exception('Got what we wanted!')
    return item_list


def init_msg(n_frames):
    msg = {
        "n": n_frames,
        "msg": {
            "msg_type": "CLIENT_INPUT",
            "get_obj_data": True,
            "send_scene_info": False,
            "actions": [],
            "send_particles": True,
        },
    }

    msg["msg"]["vel"] = [0, 0, 0]
    msg["msg"]["ang_vel"] = [0, 0, 0]
    msg["msg"]["action_type"] = "NO_OBJ_ACT"
    return msg


def convert_to_float_list(iterable):
    return [float(elt_) for elt_ in iterable]


def particle_interaction_helper(msg, particle_forces, use_absolute_coordinates=True):
    """
    Adds particle forces to a message for environment.
    msg : an initialized message, assumed to have 'action' value an empty list
    particle_forces : a ParticleForces object
    use_absolute_coordinates : whether to use absolute or
    relative (to agent) coordinates to specify forces
    """

    for obj_id in particle_forces.get_object_ids():
        msg["msg"]["actions"].append(
            {
                "force": [0.0, 0.0, 0.0],
                "torque": [0.0, 0.0, 0.0],
                "id": str(obj_id),
                "object": str(obj_id),
                "use_absolute_coordinates": use_absolute_coordinates,
                "action_pos": [],
                "particle_forces": [
                    {
                        "force": convert_to_float_list(part_force.force),
                        "pid": float(part_force.particle_id),
                    }
                    for part_force in particle_forces.get_object_forces(obj_id)
                ],
            }
        )


ObjectTeleport = namedtuple("ObjectTeleport", "object_id position orientation")


def object_teleport_helper(msg, object_teleports, use_absolute_coordinates=True):
    """Iterates through a list of ObjectTeleports, adding to message"""
    for obj_tele in object_teleports:
        msg["msg"]["actions"].append(
            {
                "id": str(obj_tele.object_id),
                "use_absolute_coordinates": use_absolute_coordinates,
                "teleport_to": {
                    "position": [
                        float(elt_) for elt_ in obj_tele.position
                    ],  # need to make sure not float32
                    "rotation": [float(elt_) for elt_ in obj_tele.orientation],
                },
            }
        )


# [{'DisplayDepth': 'png'}, {'GetIdentity' : 'png'}, {'Images' : 'png'}]
SHADERS = []
# [{'DisplayDepth' : 'depths'}, {'GetIdentity' : 'objects'}, {'Images' : 'images'}]
HDF5_NAMES = []


SHADERS_DEPTH = [{"DisplayDepth": "png"}, {"GetIdentity": "png"}]
HDF5_NAMES_DEPTH = [{"DisplayDepth": "depths"}, {"GetIdentity": "objects"}]

SHADERS_LONG = [
    {"DisplayNormals": "png"},
    {"GetIdentity": "png"},
    {"DisplayDepth": "png"},
    {"DisplayVelocity": "png"},
    {"DisplayAcceleration": "png"},
    {"DisplayJerk": "png"},
    {"DisplayVelocityCurrent": "png"},
    {"DisplayAccelerationCurrent": "png"},
    {"DisplayJerkCurrent": "png"},
    {"Images": "jpg"},
]
HDF5_NAMES_LONG = [
    {"DisplayNormals": "normals"},
    {"GetIdentity": "objects"},
    {"DisplayDepth": "depths"},
    {"DisplayVelocity": "velocities"},
    {"DisplayAcceleration": "accelerations"},
    {"DisplayJerk": "jerks"},
    {"DisplayVelocityCurrent": "velocities_current"},
    {"DisplayAccelerationCurrent": "accelerations_current"},
    {"DisplayJerkCurrent": "jerks_current"},
    {"Images": "images"},
]


class Environment(object):
    def __init__(self):
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()

    def step(self, action):
        """Returns observation after taking action.
        And a boolean representing whether .reset() should
        be called.
        """
        raise NotImplementedError()


class TDWClientEnvironment(Environment):
    """
        A barebones interface with tdw server.
        step accepts messages formatted for socket communication.

        unity_seed: random seed to be handed over to unity.
        screen_dims: observation dimensions (height x width)
        room_dims: height by width of the room.
        host_address: address of server for rendering
        selected_build: which binary to use
        msg_names: message name translation
        shaders: shader to data type
        n_cameras: number of cameras in simulation
        gpu_num: gpu on which we render
    """

    def __init__(
        self,
        unity_seed,
        random_seed,
        host_address,
        selected_build,
        screen_dims=(128, 170),
        room_dims=(20.0, 20.0),
        msg_names=HDF5_NAMES,
        shaders=SHADERS,
        send_particles=True,
        n_cameras=1,
        gpu_num="0",
        environment_timeout=60,
        image_dir="",
        hdf5_dir="",
        hdf5_n_frames=1024,
        hdf5_batch_size=256,
    ):
        for k_ in [
            "unity_seed",
            "random_seed",
            "host_address",
            "shaders",
            "selected_build",
            "send_particles",
            "image_dir",
            "hdf5_dir",
            "hdf5_n_frames",
            "hdf5_batch_size",
        ]:
            setattr(self, k_, eval(k_))

        self.num_steps = 0
        assert isinstance(msg_names, list)
        assert isinstance(shaders, list)
        if self.image_dir:
            if "Images" not in [s.keys()[0] for s in shaders]:
                shaders.append({"Images": "png"})
            if "Images" not in [s.keys()[0] for s in msg_names]:
                msg_names.append({"Images": "images"})
        self.msg_names = []
        assert len(shaders) == len(msg_names)
        for i in range(len(shaders)):
            assert len(shaders[i].keys()) == 1
            for k in shaders[i]:
                assert k in msg_names[i]
                self.msg_names.append(msg_names[i][k])
        self.msg_names = [self.msg_names] * n_cameras

        self.is_write_hdf5 = False
        if self.hdf5_dir and self.hdf5_n_frames > 0:
            assert isinstance(hdf5_dir, str)
            self.is_write_hdf5 = True
            self.hdf5_file = self._create_hdf5_file(self.hdf5_dir)
            print("Writing %s" % self.hdf5_file)
            self.hdf5_writer = HDF5Writer(self.hdf5_n_frames, self.hdf5_batch_size)
            self.hdf5_writer.open(self.hdf5_file)
            self.hdf5_writer.add_fields(self._create_hdf5_fields())
            self.hdf5_writer.run()

        self.environment_pid = None
        self.num_frames_per_msg = 1 + 1 + n_cameras * len(shaders)
        ctx = zmq.Context()
        self.tc = self.init_tdw_client()
        self.not_yet_joined = True
        # query comm particulars
        self.rng = np.random.RandomState(random_seed)
        self.CACHE = {}
        self.COMPLEXITY = (
            1500
        )  # I think this is irrelevant, or it should be. TODO check
        self.NUM_LIGHTS = 4
        self.ROOM_LENGTH, self.ROOM_WIDTH = room_dims
        self.SCREEN_HEIGHT, self.SCREEN_WIDTH = screen_dims
        self.gpu_num = str(gpu_num)
        assert self.gpu_num in [str(i_) for i_ in range(10)], self.gpu_num
        self.timeout = environment_timeout
        self.grouping = None

    def _create_hdf5_file(self, hdf5_dir):
        if not os.path.exists(hdf5_dir):
            os.makedirs(hdf5_dir)
        self.dataset_number = 0
        self.hdf5_file = os.path.join(hdf5_dir, "dataset%d.hdf5" % self.dataset_number)
        while os.path.isfile(self.hdf5_file):
            self.dataset_number += 1
            self.hdf5_file = os.path.join(
                hdf5_dir, "dataset%d.hdf5" % self.dataset_number
            )
        return self.hdf5_file

    def _create_hdf5_fields(self):
        self.hdf5_fields = [
            Attribute("valid", (), np.bool),
            Attribute("worldinfo", (), h5py.special_dtype(vlen=str)),
            Attribute("actions", (), h5py.special_dtype(vlen=str)),
            Attribute("particles", (), h5py.special_dtype(vlen=np.dtype("float32"))),
        ]

        for cam in range(len(self.msg_names)):
            for n in range(len(self.msg_names[cam])):
                # Handle set of images per camera
                shadercam_name = self.msg_names[cam][n] + str(cam + 1)
                self.hdf5_fields.append(
                    Attribute(shadercam_name, (128, 170, 3), np.uint8)
                )
        return self.hdf5_fields

    def _make_record(self, environment_message, action_message):
        record = dict(
            [
                (field.name, environment_message[field.name])
                for field in self.hdf5_fields
                if field.name is not "actions"
            ]
        )
        record["actions"] = json.dumps(action_message)
        return record

    def init_tdw_client(self):
        return TDWClient(
            self.host_address,
            initial_command="request_create_environment",
            description="test script",
            selected_build=self.selected_build,  # or skip to select from UI
            # queue_port_num="23402",
            get_obj_data=True,
            send_scene_info=False,
            num_frames_per_msg=self.num_frames_per_msg,
            send_particles=self.send_particles,
            shaders=self.shaders,
        )

    def get_items(
        self, rng, q, num_items, scale, mass, var=0.01, shape_pool=None, color=0
    ):
        for _k in default_keys:
            if _k not in q:
                q[_k] = {"$exists": True}
        print("first query")
        # TODO: do away with this horrific hack in case shape_pool is specified
        # might want to just initialize this once
        query_res = old_assets_query_result_example.query_res
        query_unity_data = query_results_to_unity_data(
            query_res, scale, mass, var=var, seed=self.unity_seed + 1
        )
        # changing from previous version, information must be passed in this way.
        assert shape_pool is not None
        for qu_data in query_unity_data:
            shape_this_time = rng.choice(shape_pool)
            # qu_data['aws_address'] = 'PhysXResources/StandardShapes/Solids' + color + '/' + shape_this_time + '.prefab'
            qu_data["aws_address"] = (
                "FlexResources/SmallShapes/Solids{num_env}/"
                + shape_this_time
                + ".prefab"
            )
            print("shape chosen: " + shape_this_time)
        return query_unity_data

    def reset(self, *round_info):
        self.nn_frames = 0
        self.round_info = round_info
        rounds = [
            {
                "items": self.get_items(
                    self.rng,
                    query_dict[info["type"]],
                    info["num_items"] * 4,
                    info["scale"],
                    info["mass"],
                    info["scale_var"],
                    shape_pool=info.get("shape_pool"),
                    color=info.get("color", "0"),
                ),
                "num_items": info["num_items"],
            }
            for info in round_info
        ]

        self.config = {
            "environment_scene": "ProceduralGeneration",
            # Omit and it will just choose one at random. Chosen seeds are output into the log(under warning or log level).
            "random_seed": self.unity_seed,
            "should_use_standardized_size": False,
            "standardized_size": [1.0, 1.0, 1.0],
            "complexity": self.COMPLEXITY,
            "random_materials": True,
            "num_ceiling_lights": self.NUM_LIGHTS,
            "intensity_ceiling_lights": 1,
            "use_standard_shader": True,
            "minimum_stacking_base_objects": 5,
            "minimum_objects_to_stack": 5,
            "disable_rand_stacking": 0,
            "room_width": self.ROOM_WIDTH,
            "room_height": 10.0,
            "room_length": self.ROOM_LENGTH,
            "wall_width": 1.0,
            "door_width": 1.5,
            "door_height": 3.0,
            # standard window ratio is 1:1.618
            "window_size_width": (5.0 / 1.618),
            "window_size_height": 5.0,
            "window_placement_height": 2.5,
            "window_spacing": 7.0,  # Average spacing between windows on walls
            "wall_trim_height": 0.5,
            "wall_trim_thickness": 0.01,
            "min_hallway_width": 5.0,
            "number_rooms": 1,
            "max_wall_twists": 3,
            # Maximum number of failed placements before we consider a room fully filled.
            "max_placement_attempts": 300,
            "grid_size": 0.4,  # Determines how fine tuned a grid the objects are placed on during Proc. Gen. Smaller the number, the
            "use_mongodb_inter": 1,
            "rounds": rounds,
        }
        if self.not_yet_joined:
            self.tc.load_config(self.config)
            self.tc.load_profile(
                {
                    "screen_width": self.SCREEN_WIDTH,
                    "screen_height": self.SCREEN_HEIGHT,
                    "gpu_num": self.gpu_num,
                }
            )
            print("about to hit run")
            msg = {"msg": {}}
            self.sock = self.tc.run()
            # Poller for timeouts:
            self.sock.setsockopt(zmq.LINGER, 0)
            self.poller = zmq.Poller()
            self.poller.register(self.sock, zmq.POLLIN)
            print("run done")
            self.not_yet_joined = False
            # else:
            self.observation = self._observe_world()
        print("waiting before switch...")
        waiting_time = 10
        for i in range(waiting_time):
            waiting_msg = {
                "msg_type": "CLIENT_INPUT",
                "action_type": "WAITING",
                "get_obj_data": True,
                "send_scene_info": False,
                "vel": [0, 0, 0],
                "ang_vel": [0, 0, 0],
                "actions": [],
            }
            msg = {"n": self.num_frames_per_msg, "msg": waiting_msg}
            self.sock.send_json(msg)
            self.observation = self._observe_world()
        print("switching scene... TODO REPLACE WITH TELEPORT")
        # scene_switch_msg = {"msg_type" : "SCENE_SWITCH", "config" : self.config, "get_obj_data" : True, "send_scene_info" : True, 'SHADERS' : self.shaders, 'send_particles': True}
        print("scene switch")
        # FLEX agent teleport -> agent looks at box
        agent_pos = [33.96, 9.0, 32.71]
        agent_rot = [-1.0, -0.65, -0.5]
        waiting_msg["teleport_to"] = {
            "position": list(agent_pos),
            "rotation": list(agent_rot),
        }
        # FLEX object teleport -> put objects in box
        object1_pos = [29.0, 1.0, 29.0]
        object1_rot = [1.0, 0.0, 0.0]
        object1_id = 47
        object2_pos = [31.0, 2.0, 31.0]
        object2_rot = [0.0, 1.0, 0.0]
        object2_id = 48
        waiting_msg["actions"] = [
            {
                "teleport_to": {
                    "position": list(object1_pos),
                    "rotation": list(object1_rot),
                },
                "id": str(object1_id),
                "use_absolute_coordinates": True,
            },
            {
                "teleport_to": {
                    "position": list(object2_pos),
                    "rotation": list(object2_rot),
                },
                "id": str(object2_id),
                "use_absolute_coordinates": True,
            },
        ]
        msg = {"n": self.num_frames_per_msg, "msg": waiting_msg}
        self.sock.send_json(msg)

        self._construct_hierarchical_representation()
        return self.observation

    def _construct_hierarchical_representation(self):
        print("constructing initial hierarchical representation")
        """
        Takes a couple waiting steps and generates an hierarchical repn.
        Fills in field grouping
        """

        initial_wait = 20
        for i in range(initial_wait):
            observation = self._observe_world()
            waiting_msg = {
                "msg_type": "CLIENT_INPUT",
                "action_type": "WAITING",
                "get_obj_data": True,
                "send_scene_info": False,
                "vel": [0, 0, 0],
                "ang_vel": [0, 0, 0],
                "actions": [],
            }
            msg = {"n": self.num_frames_per_msg, "msg": waiting_msg}
            self.sock.send_json(msg)
        observation = self._observe_world()
        # processed_observation = flex_environment_utils.basic_postprocess([observation],
        #                  [])
        # self.grouping = generate_grouping.generate_grouping(processed_observation['flex_states'][0, :, 0:3],
        #                                               processed_observation['particle_ids'][0])

    def get_grouping(self):
        if self.grouping is None:
            raise Exception("Need to run a reset before getting grouping!")
        return self.grouping

    def assign_particles_to_objects(self, data):
        particles = data["particles"]
        for obs_obj in data["info"]["observed_objects"]:
            particle_info = json.loads(obs_obj[9])
            particle_number = int(particle_info["x"])
            particle_offset = int(particle_info["y"])
            obs_obj.append(
                particles[particle_offset : particle_offset + particle_number]
            )
        # FLEX This means that you will be able to find the particles
        # at obs_obj[-1]
        return data

    def write_image_to_path(self, image, num, base_dir):
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        Image.fromarray(image).save(os.path.join(base_dir, "%06d.png" % num))

    def _observe_world(self):
        self.nn_frames += 1
        # try to receive a message, if no message can be received within
        # the timeout period or the message is errornous close the
        # current environment and restart a new one
        try:
            self.observation = self.handle_message_new(
                self.msg_names, timeout=self.timeout
            )
            self.decoded_observation = io3.decode_message(
                copy.deepcopy(self.observation)
            )
            if hasattr(self, "image_dir") and self.image_dir:
                assert "images1" in self.observation
                self.write_image_to_path(
                    self.observation["images1"], self.nn_frames, self.image_dir
                )
            self.observation["info"] = json.loads(self.observation["info"])
            self.observation = self.assign_particles_to_objects(self.observation)
            # parse dict to list
            for key in [
                "avatar_position",
                "avatar_up",
                "avatar_forward",
                "avatar_right",
                "avatar_velocity",
                "avatar_angvel",
                "avatar_rotation",
            ]:
                rec = json.loads(self.observation["info"][key])
                self.observation["info"][key] = [rec["x"], rec["y"], rec["z"]]
            for obj in self.observation["info"]["observed_objects"]:
                for i in [2, 3, 6, 7]:
                    rec = json.loads(obj[i])
                    if i == 3:
                        obj[i] = [rec["x"], rec["y"], rec["z"], rec["w"]]
                    else:
                        obj[i] = [rec["x"], rec["y"], rec["z"]]
            self.environment_pid = self.observation["info"]["environment_pid"]
        except IOError as e:
            print("Current environment is broken. Starting new one...")
            print("I/O error: {}".format(e))
            self.tc.close()
            self.tc = self.init_tdw_client()
            self.not_yet_joined = True
            print("...New environment started.")
            if self.environment_pid is not None:
                try:
                    os.kill(self.environment_pid, signal.SIGKILL)
                    print("Killed old environment with pid %d." % self.environment_pid)
                except:
                    print(
                        "Could not kill old environment with pid %d. Already dead?"
                        % self.environment_pid
                    )
            raise IOError("Environment restarted, provide new config")

        return self.observation

    def handle_message_new(
        self, msg_names, write=False, outdir="", imtype="png", prefix="", timeout=None
    ):
        if timeout is None:
            info = self.sock.recv()
        else:
            if self.poller.poll(timeout * 1000):  # timeout in seconds
                info = self.sock.recv()
            else:
                raise IOError("Did not receive message within timeout")
        data = {"info": info}
        if self.poller.poll(timeout * 1000):
            data["particles"] = np.reshape(
                np.frombuffer(self.sock.recv(), dtype=np.float32), [-1, 7]
            )
        else:
            raise IOError("Did not receive message within timeout")
        # Iterate over all cameras
        data["n_cameras"] = len(msg_names)
        data["shaders"] = msg_names[0]
        data["is_send_particles"] = True
        for cam in range(len(msg_names)):
            for n in range(len(msg_names[cam])):
                # Handle set of images per camera
                if timeout is None:
                    imgstr = self.sock.recv()
                else:
                    if self.poller.poll(timeout * 1000):  # timeout in seconds
                        imgstr = self.sock.recv()
                    else:
                        raise IOError("Did not receive message within timeout")
                imgarray = np.asarray(Image.open(StringIO(imgstr)).convert("RGB"))
                field_name = msg_names[cam][n] + str(cam + 1)
                assert field_name not in data, "duplicate message name %s" % field_name
                data[field_name] = imgarray
        return data

    def _termination_condition(self):
        return False

    def step(self, action):
        # gets message. action_to_message_fn can make adjustments to action
        # other data is included so that we can manage a cache of data all
        # in one place, but the environment otherwise does not interact with it
        if not hasattr(self, "start"):
            self.nn_frames = 0
            self.start = time.time()

        if self.is_write_hdf5:
            self.hdf5_writer.write(
                self._make_record(self.decoded_observation, action["msg"])
            )
            if self.nn_frames >= self.hdf5_n_frames:
                self.hdf5_writer.close()
                self.is_write_hdf5 = False
                with open(self.hdf5_file + ".done", "w") as f:
                    msg = "Finished writing %d examples into file: %s" % (
                        self.hdf5_n_frames,
                        self.hdf5_file,
                    )
                    print(msg)
                    f.write(msg)
                    f.close()

        msg = action
        self.sock.send_json(msg)
        self.observation = self._observe_world()
        # print(msg['msg']['actions'], self.nn_frames)
        # print("FPS: %.2f" % (self.nn_frames / (time.time() - self.start)))
        term_signal = self._termination_condition()
        return self.observation, term_signal


if __name__ == '__main__':
    tdw_environment = TDWClientEnvironment(0, 0, 'localhost')
    import ipdb
    ipdb.set_trace()
