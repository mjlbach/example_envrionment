from __future__ import print_function

import zmq
import os
import json
import base64
import numpy as np
import random
import io
from PIL import Image
from io import BytesIO
from pdb import set_trace
import tdw_client
import argparse
import ast
import logging
from collections import defaultdict
import math
import copy
import pprint
import time as time_module
from time import sleep
import cv2
import base64

logger = logging.getLogger(__name__)
hdlr = logging.FileHandler('environment.log')
fmtr = logging.Formatter('%(asctime)s - %(name)s - %(threadName)s -  %(levelname)s - %(message)s')
hdlr.setFormatter(fmtr)
logger.addHandler(hdlr)
logger.setLevel(logging.DEBUG)

# Keys used in output message from server.
key_image_data = "image_data"
key_avatar_data = "avatar_data"
key_observed_object_data = "observed_object_data"
key_time = "time_data"
key_avatar_children_data = "avatar_children_data"
key_object_data = "objects_data"

# Keys in image data
key_sensor_name_sensor_container = "SensorContainer"
key_sensor_name_follow_camera = "FollowCamera"

# Keys in avatar child data
keys_head_pose = ["Head2"]
keys_arm_pose = ["New_Left_arm", "left_upper_arm", "left_elbow", "left_lower_arm", "left_wrist", "left_mitten",
                 "New_Right_arm", "right_upper_arm", "right_elbow", "right_lower_arm", "right_wrist", "right_mitten"]
DEFAULT_SCENE_OBJECTS = ["baseball", "toy_monkey_medium", "zenblocks", "coffee_001"]


def hardcode_circular_switch_coordinate(center, radius, period, counter, num_times_each_way, phase_start = 0., direction_start = 1.):
    assert len(center) == 2
    counter_mod = counter % (2 * num_times_each_way * period)
    if counter_mod < num_times_each_way * period:
        direction = direction_start
    else:
        direction = -1. * direction_start
    phase = phase_start + direction * 2 * np.pi * float(counter_mod) / period
    return center + radius * np.array([np.cos(phase), np.sin(phase)])

class Environment(object):
    def __init__(self, environment_params):

        logger.debug("Started log in environment.")

        # Set class attributes from environment params dictionary
        for k, v in environment_params.items():
            setattr(self, k, v)

        # Avatar segmentation
        self.avatar_seg_colors = []
        self.avatar_seg_id = []
        self.baby_id = 'a0'

        #debugging
        self.old_kettles = None
        self._num_check_calls = 0
        self.step_counter = 0

        # Interaction
        self.actualized_action = None

        # Number of avatars, objects, spatial splits
        self.num_objects_per_quad = len(self.objects)
        self.num_avatars = len(self.avatars)
        self.num_quad = len(self.object_quadrants)
        self.num_avatars_objects = (self.num_objects_per_quad*self.num_quad) + self.num_avatars

        # Room dimensions
        self.steps_per_wall = 3.15
        self.half_spw = self.steps_per_wall / 2.
        self.room_w = self.room_dims[0]
        self.room_l = self.room_dims[1]
        self.y_position = '0.0'

        # Object ID, Bounds within quadrant where objects can be spawn
        self.confine_len_x = self.half_spw * self.room_w / 6.
        self.confine_len_z = self.half_spw * self.room_l / 6.
        self.object_ids = []
        self.animate_quadrant = None # When you only spawn object in animate quad

        # initial head position
        self.avatar_tilt = 0.
        self.avatar_pan = 0.
        self.avatar_rotation = 0.
        self.avatar_twist_right_arm = 0.
        self.avatar_twist_left_arm = 0.

        # Server communication
        self.output_socket = None # output of build
        self.input_socket = None # input to build
        self.use_server = environment_params.get('use_server')
        self.controller_id = 'A'.encode('utf-8')
        self.build_id = 'build_0'.encode('utf-8')
        self.send_time = []
        self.recv_time = []
        self.num_render = 0

        if self.use_server:
            print("Connecting to server at {}".format(self.server_path))
            self.tc = tdw_client.TDWClient(self.host_name,
                                           self.build_path,
                                           self.server_path,
                                           self.render_images,
                                           self.controller_id,
                                           self.build_id,
                                           self.screen_height,
                                           self.screen_width,
                                           self.render_port)
        else:
            print("Not using server")
            self.tc = tdw_client_no_server.TDWClient(self.host_name,
                                                     self.build_path,
                                                     self.server_path,
                                                     self.render_images,
                                                     self.controller_id,
                                                     self.build_id,
                                                     self.screen_height,
                                                     self.screen_width)

        self._counter = 0
        #for hardcode prototyping
        self._periodic_rng = np.random.RandomState(seed = 0)
        self._avatar_indices = {'animate' : 0, 'periodic' : 1, 'random' : 2, 'static' : 3}
        #hardcoded circulaf periodic workaround
        

    def get_unique_id(self):
        return random.randint(0, 10000)

    def do_nothing(self):
        return {"$type": "translate_avatar_by", "direction": {'x': 0, 'y': 0, 'z': 0}, "magnitude": 10, "avatar_id": self.baby_id, "env_id": 0}

    def generate_x(self, quadrant, type):

       if type == 'avatar':
           if (quadrant == 'TR') | (quadrant == 'BR'):
                x_position = str((self.room_w*self.half_spw)/2)
           elif (quadrant == 'TL'): # (quadrant == 'TL') | (quadrant == 'BL')
                x_position = str((-self.room_w*self.half_spw)/2)
           elif (quadrant == 'BL'):
                x_position = str((-self.room_w*self.half_spw)/2 + 0.5)
           elif (quadrant == 'random'):
                x_position = str(random.uniform((-self.room_w * 0.5) + 0.5, (self.room_w * 0.5) - 0.5))
           elif (quadrant == 'center'):
                return str(0)
           else:
                print('invalid avatar position argument!')

       elif type == 'object':
           if (quadrant == 'TR') | (quadrant == 'BR'):
                x_position = str( random.uniform(self.confine_len_x, (self.room_w * self.half_spw) - self.confine_len_x))
           elif (quadrant == 'TL') | (quadrant == 'BL'):
                x_position = str( random.uniform((-self.room_w * self.half_spw) + self.confine_len_x, -self.confine_len_x ))
           else:
                x_position = str( random.uniform( (-self.room_l* self.half_spw) + self.confine_len_x, (self.room_l * self.half_spw) - self.confine_len_x ))

       return x_position

    def generate_z(self, quadrant, type):

        if type == 'avatar':
            if (quadrant == 'TR') | (quadrant == 'TL'):
                z_position = str( (self.room_l * self.half_spw)/2 )
            elif (quadrant == 'BR'): # (quadrant == 'BR') | (quadrant == 'BL')
                z_position = str( (-self.room_l * self.half_spw)/2 )
            elif (quadrant == 'BL'):
                z_position = str( (-self.room_l * self.half_spw)/2 + 0.5 )
            elif(quadrant == 'center'):
                return str(0)
            else:
                print('invalid avatar position argument!')

        elif type == 'object':
            if (quadrant == 'TR') | (quadrant == 'TL'):
                z_position = str( random.uniform(self.confine_len_z, (self.room_l * self.half_spw) - self.confine_len_z ))
            elif (quadrant == 'BR') | (quadrant == 'BL'):
                z_position = str( random.uniform((-self.room_l * self.half_spw) + self.confine_len_z, -self.confine_len_z))
            else:
                z_position = str( random.uniform((-self.room_l * self.half_spw) + self.confine_len_z, (self.room_l * self.half_spw) - self.confine_len_z))

        return z_position


    def generate_det_z(self, quadrant, i):

        det_z_BL = [-(self.room_l * self.half_spw)/2 + 1 , -(self.room_l * self.half_spw)/2 - 1,  -(self.room_l * self.half_spw)/2 - 1]

        if quadrant == 'BL' or quadrant == 'BR':
            return str(det_z_BL[i])

        elif quadrant == 'TL' or quadrant == 'TR':
            return str(det_z_BL[i] + self.room_l * self.half_spw)


    def generate_det_x(self, quadrant, i):
        det_x_BL = [-(self.room_w * self.half_spw)/2, -(self.room_w * self.half_spw)/2 + 1, -(self.room_w * self.half_spw)/2 - 1]

        if quadrant == 'BL' or quadrant == 'TL':
            return str(det_x_BL[i])

        elif quadrant == 'BR' or quadrant == 'TR':
            return str(det_x_BL[i] + self.room_w * self.half_spw)


    def generate_y(self, avatar):
        if (avatar == 'baby'):
            return str(0)
        elif (avatar == 'ball'):
            return str(self.ball_y)
        else:
            print('invalid avatar!')

    def get_color(self):
        return str(random.uniform(0,1))

    def make_obj_layout(self):
        obj_list = []
        for j, quadrant in enumerate(self.object_quadrants):
            for i, obj in enumerate(self.objects):
                cur_id = self.get_unique_id()
                cur_obj = {"name": obj,
                           "position": {"x": self.generate_x(quadrant, 'object'),
                                        "y": self.y_position,
                                        "z": self.generate_z(quadrant, 'object')},
                           "orientation": {"x": 0.0, "y": 0.0, "z": 0.0},
                           "id": str(cur_id),
                           }
                obj_list.append(cur_obj)
                self.object_ids.append(cur_id)

        return {"$type": "add_objects", "env_id": 0, "objects": obj_list}

    def make_det_obj_layout(self):
        obj_list = []
        for j, quadrant in enumerate(self.object_quadrants):
            for i, obj in enumerate(self.objects):
                cur_id = self.get_unique_id()
                cur_obj = {"name": obj,
                           "position": {"x": self.generate_det_x(quadrant, i),
                                        "y": self.y_position,
                                        "z": self.generate_det_z(quadrant, i)},
                           "orientation": {"x": 0.0, "y": 0.0, "z": 0.0},
                           "id": str(cur_id),
                           }
                obj_list.append(cur_obj)
                self.object_ids.append(cur_id)

        return {"$type": "add_objects", "env_id": 0, "objects": obj_list}

    def set_object_masses(self):
        cmd = []
        for i, obj_id in enumerate(self.object_ids):
            cmd.append({"$type": "set_mass", "mass": self.object_masses, "id": obj_id})

        return cmd

    def make_avatar_layout(self):

        # Set the avatar's output.
        # Set the image pass mask.
        # Set simulation to pause mode.

        cmd = []
        pass_cmd = []
        for i, avatar in enumerate(self.avatars):
            # Base command
            curr_id = "a" + str(i)
            cmd += [{"$type": "create_avatar", "type": avatar, "id": curr_id}]

            # Agent
            if avatar == "A_StickyMitten_Baby":

                cmd += [{"$type": "set_img_pass_encoding", "value": False},
                        {"$type": "set_avatar_output", "avatar_id": curr_id, "env_id": 0,
                         "images": True, "object_info": True, "avatar_info": True,
                         "child_info": False, "collision_info": False, "sensors_info": False},
                        {"$type": "set_pause", "value": True},
                        {"$type": "teleport_avatar_to", "avatar_id": curr_id, "env_id": 0,
                         "position": {"x": self.generate_x('center', 'avatar'),
                                      "y": self.generate_y('baby'),
                                      "z": self.generate_z('center', 'avatar')}}]

                pass_cmd += [{"$type": "set_pass_masks", "avatar_id": curr_id, "pass_masks": ["_img", "_id"], "env_id": 0}]

            # Ext. agents (Spheres)
            else:
                quadrant = self.quadrants[i-1]
                color = self.avatar_colors[i-1]

                cmd += [{"$type": "set_avatar_output", "avatar_id": curr_id, "env_id": 0,
                         "images": False, "object_info": False, "avatar_info": True,
                         "child_info": True, "collision_info": False, "sensors_info": False},
                        {"$type": "teleport_avatar_to", "avatar_id": curr_id, "env_id": 0,
                         "position": {"x": self.generate_x(quadrant, 'avatar'),
                                      "y": self.generate_y('ball'),
                                      "z": self.generate_z(quadrant, 'avatar')}},
                        {"$type": "change_avatar_body", "body_type": "Sphere", "avatar_id": curr_id, "env_id": 0},
                        {"$type": "change_avatar_color", "avatar_id": curr_id, "env_id": 0,
                         "color": {"r": color[0], "g": color[1], "b": color[2], "a": 1.0}},
                        {"$type": "scale_avatar", "scale_factor": str(self.ball_scale),
                         "avatar_id": curr_id, "env_id": 0}]

                pass_cmd += [{"$type": "toggle_sensor", "avatar_id": curr_id, "env_id": 0, "sensor_name": "SensorContainer"}]
                #pass_cmd += [{"$type": "set_pass_masks", "avatar_id": curr_id, "pass_masks": [], "env_id": 0}]

                #save quadrant of animate agent, so that we know where to place objects
                if i == 1:
                    self.animate_quadrant = quadrant

        return cmd, pass_cmd


    def _scale_action(self, action):
        '''
        scale the action vector accordingly
        '''
        def get_scaled_value_helper(value, new_value, thresh=0.1):
            if self.continuous_actions:
                 return value * new_value
            else:
                if value < -thresh:
                     return -new_value
                elif value > thresh:
                     return new_value
                else:
                     return 0.

        scaled_action = copy.deepcopy(action)

        for act, amt in scaled_action.act_dict.items():
            if act == 'move':
                scaled_action.act_dict['move'] = get_scaled_value_helper(scaled_action.act_dict['move'], 2.)

            elif act == 'turn':
                rel_turn_amt = get_scaled_value_helper(scaled_action.act_dict['turn'], 0.25) # 0.25 before
                self.avatar_rotation += rel_turn_amt
                scaled_action.act_dict['turn'] = self.avatar_rotation

            elif act == 'tilt':
                rel_tilt_amt = get_scaled_value_helper(scaled_action.act_dict['tilt'], 0.5)
                # don't allow tilt past +/- 45
                if self.avatar_tilt <= -45 and rel_tilt_amt <= 0.:
                    rel_tilt_amt = 0.
                if self.avatar_tilt >= 45 and rel_tilt_amt > 0.:
                    rel_tilt_amt = 0.
                self.avatar_tilt += rel_tilt_amt
                scaled_action.act_dict['tilt'] = self.avatar_tilt
            elif act == 'pan':
                rel_pan_amt = get_scaled_value_helper(scaled_action.act_dict['pan'], 0.5)
                # don't allow pan past +/- 60
                if self.avatar_pan <= -60 and rel_pan_amt <= 0.:
                    rel_pan_amt = 0.
                if self.avatar_pan >= 60 and rel_pan_amt > 0.:
                    rel_pan_amt = 0.
                self.avatar_pan += rel_pan_amt
                scaled_action.act_dict['pan'] = self.avatar_pan

            elif act == 'bend_right_arm':
                scaled_action.act_dict['bend_right_arm'] = get_scaled_value_helper(scaled_action.act_dict['bend_right_arm'], 5.)
            elif act == 'bend_left_arm':
                scaled_action.act_dict['bend_left_arm'] = get_scaled_value_helper(scaled_action.act_dict['bend_left_arm'], 5.)
            elif act == 'twist_right_arm':
                scaled_action.act_dict['twist_right_arm'] = get_scaled_value_helper(scaled_action.act_dict['twist_right_arm'], 5.)
            elif act == 'twist_left_arm':
                scaled_action.act_dict['twist_left_arm'] = get_scaled_value_helper(scaled_action.act_dict['twist_left_arm'], 5.)
            elif act == 'bend_right_wrist':
                 scaled_action.act_dict['bend_right_wrist'] = get_scaled_value_helper(scaled_action.act_dict['bend_right_wrist'], 1.)
            elif act == 'bend_left_wrist':
                 scaled_action.act_dict['bend_left_wrist'] = get_scaled_value_helper(scaled_action.act_dict['bend_left_wrist'], 1.)

            elif 'rotate_right_arm' in act:
                 scaled_action.act_dict[act] = get_scaled_value_helper(scaled_action.act_dict[act], 5.)
            elif 'rotate_left_arm' in act:
                 scaled_action.act_dict[act] = get_scaled_value_helper(scaled_action.act_dict[act], 5.)

            else:
                raise NotImplementedError('scaling for {} action not implemented'.format(act))

        scaled_action.assign_vars_from_act_dict() # not necessary if we don't use Action instance variables
        return scaled_action

    def _action_to_message(self, act_vec):
        # Default message items for the baby agent
        messages = []
        self.actualized_action = np.copy(act_vec)
        mapping_dict = {act:idx for idx, act in enumerate(self.action_space)}
        action = Action(act_vec, mapping_dict)
        scaled_action = self._scale_action(action)
        self._add_actions_to_message(scaled_action, messages)
        return messages


    def _add_actions_to_message(self, action, messages):
        def update_vec_dict(key, value, vec_dict={"x": 0.0, "y": 0.0, "z": 0.0}):
            vec_dict.update({key:value})
            return vec_dict

        act_dict = action.act_dict
        for key in act_dict.keys():
            message = {"avatar_id": "a0", "env_id": self.env_id}
            if key == 'move':
                message['$type'] = 'translate_avatar_by'
                message['magnitude'] = act_dict['move']
                message['direction'] = self.avatar_forward
            elif key == 'turn':
                message['$type'] = 'rotate_avatar_to'
                message['rotation'] = {"x": 0.0, "y": act_dict['turn'], "z": 0.0}
            elif key == 'tilt':
                message['$type'] = 'rotate_sensor_container_by'
                #xprint act_dict['tilt']
                message['rotation'] = {"x": 0.0, "y": act_dict['tilt'], "z": 0.0}
            elif key == 'pan':
                message['$type'] = 'rotate_sensor_container_by'
                message['rotation'] = {"x": act_dict['pan'], "y": 0.0, "z": 0.0}
            messages.append(message)

    def _check_coordinates(self, explicit_coord_no_mask):
        self._num_check_calls += 1
        #print('check call ' + str(self._num_check_calls))
        hrw = half_room_width = 4.5#just hardcode, this is debugging
        buff_hrw = hrw + 1.5

        #reshape kettles
        try:
            exp_coord = np.array(explicit_coord_no_mask)
            kettles = explicit_coord_no_mask[4 * 2 : 2 * 4 + 3 * 2 * 4].reshape((3 * 4, 2))
        except:
            raise Exception('Failed array and reshape: ' + str(explicit_coord_no_mask))

        #now just check that each coordinate is within room bounds
        for ket_ in kettles:
            for coord_ in ket_:
                assert - hrw < coord_ < hrw, (explicit_coord_no_mask, self._num_check_calls)

        #check that agents are in their right places
        #animate agent check has been moved to be inside pagent
        '''
        assert -buff_hrw < exp_coord[0] < 0. and -buff_hrw < exp_coord[1] < 0. and\
            0. < exp_coord[2] < buff_hrw and 0. < exp_coord[3] < buff_hrw and\
            -buff_hrw < exp_coord[4] < 0. and 0. < exp_coord[5] < buff_hrw and\
            0. < exp_coord[6] < buff_hrw and -buff_hrw < exp_coord[7] < 0., (explicit_coord_no_mask,  self._num_check_calls)
        '''

        #check for minimal kettle motion
        #if self.old_kettles is None:
        #    self.old_kettles = kettles
        #else:
        #    assert np.linalg.norm(kettles - self.old_kettles) < 1., (explicit_coord_no_mask, kettles, self.old_kettles)



    def _handle_message(self, exp_num_msgs):
        return_data = {'images': None,
                       'seg_images': None,
                       'objects_in_view': None,
                       'centroids': None,
                       'object_data': None,
                       'avatar_data': None,
                       'object_ids': None,
                       'object_centers': None,
                       'centers_rel_baby': [],
                       'object_type': None,
                       'num_seen_avatars': {'animate': 0, 'random': 0, 'periodic': 0, 'static': 0},
                       'explicit_coord': None}

        self.step_counter += 1

        t1 = time_module.time()
        messages = []
        print("getting message")
        for k in range(exp_num_msgs):
            if self.poller.poll(self.environment_timeout * 1000):
                messages += self.output_socket.recv_multipart()
            else:
                raise IOError("[_handle_messages() in env.py: Did not receive message within timeout")
        print("received message")

        t2 = time_module.time()
        self.recv_time.append(t2 - t1)

        # collect data in these, before appending to return_data at the end
        render_imgstr = ''
        segment_imgstr = ''
        follow_render_imgstr = ''
        avatar_data = []
        object_ids = []
        object_centers = []
        object_relative_centers = []
        object_type = []
        object_colors = [];
        objects_in_view = [None]*(self.num_avatars_objects)
        colors_in_view = []
        centroids = [None]*(self.num_avatars_objects)
        avatar_pos = None

        '''
        print('_____________________')
        msg_dict_list = []
        obj_msg = []
        for msg in messages:
            message_dict = json.loads(msg.decode('utf-8'))
            if message_dict['$type'] == 'objects_data':
                obj_msg.append(message_dict)
            msg_dict_list.append(message_dict)
            #print(message_dict['$type'])
        print('______________________')
        if obj_msg:
            for i in range(12):
                print((obj_msg[0]['objects'][i]['name'],obj_msg[0]['objects'][i]['center']))
        else:
            print('no obj_msg')
        '''

        #set_trace()
        for msg in reversed(messages):
            try:
                message_dict = json.loads(msg.decode('utf-8'))
                parsed = True
            except:
                message_dict = None
                parsed = False
            finally:
                if not parsed:
                    print("#-----FAILED TO PARSE-----#")
                    print(msg)

            # Images
            if (message_dict['$type'] == key_image_data):
                if (message_dict['avatar_id'] == self.baby_id) & (message_dict['sensor_name'] == 'SensorContainer') & (len(message_dict['passes']) > 0):

                    imgstr = message_dict['passes'][0]['image']
                    segstr = message_dict['passes'][1]['image']

                    # Normal image
                    render_imgstr = imgstr
                    image_bytes = base64.b64decode(imgstr.encode('utf-8'))

                    # New Solution
                    imgarr = np.fromstring(image_bytes, dtype=np.uint8)
                    imgarr = cv2.imdecode(imgarr, cv2.IMREAD_ANYCOLOR)
                    imgarr = cv2.cvtColor(imgarr,cv2.COLOR_BGR2RGB)
                    return_data['images'] = imgarr

                    #same for segmentation
                    seg_bytes = base64.b64decode(segstr.encode('utf-8'))
                    segarr = np.fromstring(seg_bytes, dtype = np.uint8)
                    segarr = cv2.imdecode(segarr, cv2.IMREAD_ANYCOLOR)
                    segarr = cv2.cvtColor(segarr, cv2.COLOR_BGR2RGB)

                    return_data['seg_images'] = segarr

            # Avatars
            elif message_dict['$type'] == key_avatar_data:
                # More avatar_data msgs to parse after this one
                if len(avatar_data) < self.num_avatars:
                    avatar_data.append(message_dict)
                    object_centers.append(message_dict['position'])
                    object_ids.append(message_dict['avatar_id'])
                    avatar_number = int(message_dict['avatar_id'][1])
                    object_type.append(self.avatar_types[avatar_number])

                    # Update internal baby agent state variables
                    if self.avatar_types[avatar_number] == 'a_baby':
                        self.avatar_rotation = message_dict['rotation']['y']
                        print("Step: {}, curagent pos: {}".format(self.step_counter, message_dict['position']))

                if len(avatar_data) == self.num_avatars:
                    return_data['avatar_data'] = avatar_data

            # Objects
            elif message_dict['$type'] == key_observed_object_data: # message_dict.keys() = [$type, objects, avatar_id, env_id]
                '''
                print('------------------')
                for obj in message_dict['objects']:
                    print("Name: {}, Coord: {}".format(obj['name'], obj['center']))
                print('------------------')
                '''

                # Get observed object data
                obs_obj = message_dict['objects']
                avatar_names = dict((v,k) for k,v in self.avatar_ids.items())

                # Default avatar_pos
                sing_avatar_pos = None
                avatar_pos = [0.] * 8
                indicators = [0.] * 4 # animate, periodic, random, static
                non_avatar_pos =[]

                # Get information for avatars in view
                for obj in obs_obj:
                    if obj['name'][:-3] == 'A_Simple_Body(Clone)':
                        av_id = int(obj['name'][-1]) - 1
                        av_name = avatar_names[av_id+1][2:-3]
                        return_data['num_seen_avatars'][av_name] = 1
                        indicators[av_id] = 1
                        sing_avatar_pos = np.array([obj['center']['x'], obj['center']['z']])

                # If more than one agent in view at a time, then report
                if np.count_nonzero(return_data['num_seen_avatars'].values()) > 1:
                    print("CRITICAL ERROR: More than one ext. agent in view at a time")

                elif sing_avatar_pos is not None:
                    # Rotate to coordinates in agent's frame of reference
                    pi_ang = 2* np.pi * self.avatar_rotation / 360.
                    rot_mat = np.array([[np.cos(pi_ang), -np.sin(pi_ang)], [np.sin(pi_ang), np.cos(pi_ang)]])
                    #sing_avatar_pos = list(np.matmul(rot_mat, sing_avatar_pos))
                    sing_avatar_pos = list(sing_avatar_pos)
                    avatar_pos[av_id*2 : av_id*2+2] = sing_avatar_pos

            # Avatar and objects data
            elif message_dict['$type'] == key_object_data:

                #collect avatar positions without masks
                avatar_pos_no_mask = [0.] * (2 * (len(avatar_names) - 1))

                for observed_object in message_dict['objects']:
                    # "Sphere" contains the segmentation color for the objects
                    if observed_object['name'] == 'Sphere' and len(self.avatar_seg_colors) < self.num_avatars - 1:
                        colors = [observed_object['color']['r'],
                                  observed_object['color']['g'],
                                  observed_object['color']['b']]

                        x = observed_object['center']['x']
                        z = observed_object['center']['z']

                        # Get avatar type that corresponds to each quadrant
                        if x < 0 and z < 0:
                            avatar_type = self.avatar_types[self.quadrants.index('BL') + 1][2:]

                        elif x < 0 and z > 0:
                            avatar_type = self.avatar_types[self.quadrants.index('TL') + 1][2:]

                        elif x > 0 and z < 0:
                            avatar_type = self.avatar_types[self.quadrants.index('BR') + 1][2:]

                        else:
                            avatar_type = self.avatar_types[self.quadrants.index('TR') + 1][2:]

                        # seg_id[i] has color seg_colors[i]
                        self.avatar_seg_colors.append(tuple([round(x*255) for x in colors]))
                        self.avatar_seg_id.append(avatar_type)

                        print("Acquired segmentation color for avatar {}".format(avatar_type))

                    elif observed_object['name'] == 'kettle_0':
                        object_ids.append(observed_object['id']) # Each object (kettle) has some unique 4 digit id
                        object_centers.append(observed_object['center'])
                        object_type.append(observed_object['name'])
                        colors = [observed_object['color']['r'], observed_object['color']['g'], observed_object['color']['b']]
                        object_colors.append(tuple([round(x*255) for x in colors]))

                    #get avatar positions without masking for visibility
                    elif observed_object['name'][:-3] == 'A_Simple_Body(Clone)':
                        av_id = int(observed_object['name'][-1]) - 1
                        av_name = avatar_names[av_id+1][2:-3]
                        sing_avatar_pos = [observed_object['center']['x'], observed_object['center']['z']]
                        avatar_pos_no_mask[av_id*2 : av_id*2+2] = sing_avatar_pos

                # Kettles are the first 12 objects
                return_data['object_data'] = message_dict['objects'][:12]
                self.object_ids = object_ids

            # Time
            elif message_dict['$type'] == key_time:
                time = message_dict['time']

            # Parsing error
            elif message_dict['$type'] != key_avatar_children_data:
                print("MESSAGE PARSING ERROR: Unrecognized Message Type: {}".format(message_dict['$type']))


        # Get objects in view and centroids via segementation image
        if return_data['object_data'] != None and return_data['avatar_data'] != None and avatar_pos != None:

            segmentation_indicators = [0.] * 4
            colors_in_view = np.unique(return_data['seg_images'].reshape(-1, 3), axis = 0)
            colors_in_view = set([tuple(arr_) for arr_ in colors_in_view])
            avatar_seg_colors = copy.deepcopy(self.avatar_seg_colors)

            for color in colors_in_view:
                if color in avatar_seg_colors:
                    # return_data['num_seen_avatars'][self.avatar_seg_id[avatar_seg_colors.index(color)]] += 1
                    _ava_name = self.avatar_seg_id[avatar_seg_colors.index(color)]
                    segmentation_indicators[self._avatar_indices[_ava_name]] = 1.
                    avatar_seg_colors.pop(avatar_seg_colors.index(color))

            # Process all entity = (agent, external agent, objects) coordinates
            num_avatar_and_objects = self.num_avatars_objects
            num_avatar_and_object_data_returned = len(object_ids)

            # Make sure information for all the avatars were returned
            if num_avatar_and_object_data_returned == self.num_avatars_objects:

                # Sort object information by id number and pass back as lists
                sorted_inds = np.argsort(object_ids)
                return_data['object_ids'] = [object_ids[i] for i in list(sorted_inds)]
                return_data['object_centers'] = [object_centers[i] for i in list(sorted_inds)]
                return_data['object_data'] = sorted(return_data['object_data'], key=lambda k: k['id'])
                return_data['object_type'] = [object_type[i] for i in list(sorted_inds)]

                # Get relative positions of objects & avatars of the baby
                baby_index = return_data['object_type'].index('a_baby')
                baby_center = return_data['object_centers'][baby_index]

                for center in return_data['object_centers']:
                    return_data['centers_rel_baby'].append({x: center[x] - baby_center[x] for x in center if x in baby_center})

                # Get explicit coordinates of objects
                for i, object_center in enumerate(return_data['object_centers']):
                    if return_data['object_type'][i] == 'kettle_0':
                        #print(object_center)
                        non_avatar_pos += [object_center['x'], object_center['z']]

                #this should be the correct coordinate transformation (angle is between -180 ~ +180)
                avatar_rot_radians = (self.avatar_rotation - 0.) * 2 * np.pi / 360.
                avatar_orientation = [np.sin(avatar_rot_radians), np.cos(avatar_rot_radians)]

                indicators = segmentation_indicators


                #quick hack mods for mouse exps
                if self.mask_out_anim_rand:
                    avatar_pos_no_mask[0:2] = [0., 0.]
                    avatar_pos_no_mask[4:6] = [0., 0.]
                    indicators[0] = 0
                    indicators[2] = 0

                if self.periodic_hardcoding is not None and self.periodic_hardcoding != 'None':
                    print('using periodic hardcoding: ' + self.periodic_hardcoding)
                    if self.periodic_hardcoding == 'circular_switch':
                        new_periodic_pos = hardcode_circular_switch_coordinate(
                                self.periodic_center,
                                self.periodic_radius,
                                self.periodic_period,
                                self._counter,
                                2,                                
                            )
                    else:
                        raise Exception('Unrecognized periodic hardcoding')
                    self._counter += 1
                    avatar_pos_no_mask[2:4] = new_periodic_pos
                    #this is a bit silly to compute the angle again, but just to use tested fns...
                    cur_ang_ = np.arctan2(avatar_orientation[0], avatar_orientation[1]) * 180. / np.pi
                    per_ang_ = np.arctan2(new_periodic_pos[0], new_periodic_pos[1]) * 180. / np.pi
                    ang_diff = abs(per_ang_ - cur_ang_)
                    #just choosing something about right
                    if ang_diff < 28.:
                        indicators[1] = 1
                        return_data['num_seen_avatars']['periodic'] = 1
                    else:
                        indicators[1] = 0
                        return_data['num_seen_avatars']['periodic'] = 0


                ava_pos_arr_ = np.array(avatar_pos_no_mask)
                for i_, ind_ in enumerate(indicators):
                    ava_pos_arr_[2 * i_ : 2 * i_ + 2] *= ind_



                #compute new avatar pos with indicators

                return_data['explicit_coord'] = np.array(list(ava_pos_arr_) + list(np.sort(non_avatar_pos)) + avatar_orientation + indicators)
                return_data['explicit_coord_no_mask'] = np.array(avatar_pos_no_mask + list(np.sort(non_avatar_pos)) + avatar_orientation + indicators)

                return_data['kettle_positions' ] = np.array(non_avatar_pos)
                return_data['segmentation_indicators'] = np.array(segmentation_indicators)
                print(return_data['explicit_coord'])

                # Checking validity of coordinate
                self._check_coordinates(return_data['explicit_coord_no_mask'])

                #print(non_avatar_pos)

                '''
                # Package the explicit coordinates
                if avatar_pos == [0.]*8:
                    return_data['explicit_coord'] = np.array(avatar_pos + [(self.avatar_rotation - 180.)/90.] + [0] + list(np.sort(non_avatar_pos)))
                else:
                    return_data['explicit_coord'] = np.array(avatar_pos + [(self.avatar_rotation - 180.)/90.] + [1] + list(np.sort(non_avatar_pos)))
                '''

                '''
                avatar_idx_list = [return_data['object_ids'].index('a1'),
                                   return_data['object_ids'].index('a2'),
                                   return_data['object_ids'].index('a3'),
                                   return_data['object_ids'].index('a4')]
                avatar_name_list = ['animate', 'periodic', 'random', 'static']
                # Default avatar_pos
                avatar_pos = [0., 0.]
                non_avatar_pos =[]
                # If more than one agent in view at a time, then raise an error
                if np.count_nonzero(return_data['num_seen_avatars'].values()) > 1:
                    print("CRITICAL ERROR: More than one environment agent in view at a time")
                    #exit(1)
                else:
                    for i, avatar_idx in enumerate(avatar_idx_list):
                        if return_data['num_seen_avatars'][avatar_name_list[i]]:
                            #print("Agent in view: {}".format(avatar_name_list[i]))
                            avatar_pos = [return_data['object_centers'][avatar_idx]['x'],
                                          return_data['object_centers'][avatar_idx]['z']]
                for i, object_center in enumerate(return_data['object_centers']):
                    if return_data['object_type'][i] == 'kettle_0':
                        non_avatar_pos += [object_center['x'], object_center['z']]
                # Rotate to coordinates in agent's frame of reference
                pi_ang = 2* np.pi * self.avatar_rotation / 360.
                rot_mat = np.array([[np.cos(pi_ang), -np.sin(pi_ang)], [np.sin(pi_ang), np.cos(pi_ang)]])
                avatar_pos = np.array(avatar_pos)
                avatar_pos = list(np.matmul(rot_mat, avatar_pos))
                # Mask the loss
                if avatar_pos == [0., 0.]:
                    return_data['explicit_coord'] = np.array(avatar_pos + [0] + list(np.sort(non_avatar_pos)) + [(self.avatar_rotation - 180.)/90.])
                else:
                    return_data['explicit_coord'] = np.array(avatar_pos + [1.] + list(np.sort(non_avatar_pos)) + [(self.avatar_rotation - 180.)/90.])
                '''

            else:
                print("MESSAGE PARSING ERROR: Too much/little avatar,object information (Required: {}, Received: {})".format(num_avatar_and_objects, num_avatar_and_object_data_returned))

        # t3 = time_module.time()
        # Send back the image string if running visualizer
        if self.render_images:
            # Poll for 10 milliseconds
            if self.num_render == 0:
                if self.render_poller.poll(1):
                    print("Request Received")
                    recv = self.render_socket.recv_multipart()
                    self.num_render = int(recv[0]) # Number of images to send back
                    self.render_socket.send_multipart(['SEND', render_imgstr.encode('utf-8')])

            else:
                # Once interacting, you can wait for longer time
                if self.render_poller.poll(10000):
                    recv = self.render_socket.recv_multipart()
                    # Done sending images
                    if self.num_render == 1:
                        self.render_socket.send_multipart(['DONE', render_imgstr.encode('utf-8')])
                    else:
                        self.render_socket.send_multipart(['SEND', render_imgstr.encode('utf-8')])
                    self.num_render -= 1

        #t4 = time_module.time()
        #print("Render Image Time: {} seconds".format(t4 - t3))

        return return_data


    def _observe_world(self, exp_num_msgs=1):

        try:
            t1 = time_module.time()
            observation = self._handle_message(exp_num_msgs)
            t2 = time_module.time()
            #print("Handle Message Time: {}".format(t2 - t1))
            self.observation = observation
            return observation

        except (zmq.ZMQError, IOError) as e:
            logger.debug('Current environment broken: {}.'.format(e))
            self.output_socket, self.input_socket, self.render_socket = self.tc.quit()
            self.tc = tdw_client.TDWClient(self.host_name,
                                           self.build_path,
                                           self.server_path,
                                           self.render_images,
                                           self.controller_id,
                                           self.build_id,
                                           self.screen_height,
                                           self.screen_width)
            raise IOError("Environment restarted.")

    def _send(self, message):
        """
        Sends a multipart message to the server.
        :param message: The message to send. This may be 1 message or a list of messages.
        """

        if not isinstance(message, list):
            message = [message]

        payload = json.dumps(message).encode('utf-8')

        # Always send the build ID as the first part of the multipart message.
        t1 = time_module.time()
        self.input_socket.send_multipart([self.build_id, payload])
        t2 = time_module.time()
        self.send_time.append(t2 - t1)



    def send_init(self, sock, commands):
        """
        Send an init command to the build.
        :param sock: The sending socket.
        :param commands: The commands.
        """

        commands = [b'build_0', json.dumps(commands).encode('utf-8')]
        sock.send_multipart(commands)


    def reset_attributes(self):
        self.env_id = 0
        self.idnum = 0
        self.avatar_seg_colors = []
        self.avatar_seg_id = []
        self.object_ids = []
        self.actualized_action = None
        self.animate_quadrant = None
        self.num_render = 0

        # Initial avatar orientation
        self.avatar_tilt = 0.
        self.avatar_pan = 0.
        self.avatar_rotation = 0.
        self.avatar_twist_right_arm = 0.
        self.avatar_twist_left_arm = 0.

        # Benchmarking metrics
        self.send_time = []
        self.recv_time = []

    def reset(self, *round_info):
        '''
        Reset the environment
        '''

        # Reset all necessary attributes
        self.reset_attributes()

        print('_________________________________________')
        print("Resetting environment...")

        # Connect to server
        if self.output_socket is None:
            self.output_socket = self.tc.get_output_socket() # Output of build
            self.input_socket =  self.tc.get_input_socket() # Input to build
            self.render_socket = self.tc.get_render_socket()
        print("Hooking up sockets to server.py...")

        # Poller
        self.poller = zmq.Poller()
        self.poller.register(self.output_socket, zmq.POLLIN)

        # Poller for render socket
        self.render_poller = zmq.Poller()
        self.render_poller.register(self.render_socket, zmq.POLLIN)

        # Load the scene.
        self._send({"$type": "load_scene", "scene_name": "ProcGenScene"})
        print("Loaded scene")

        # Build the rooms
        self._send({"$type": "build_rooms", "num_rooms_x": 1, "num_rooms_z": 1, "room_height": 1,
                    "room_length": self.room_l, "room_width": self.room_w, "windows": False})
        print("Built room")

        # Make object layout
        if self.objects:
            obj_layout = self.make_det_obj_layout() if self.use_det_layout else self.make_obj_layout()
            self._send(obj_layout)
            print('Set object layout')

            # Set masses of objects
            obj_masses = self.set_object_masses()
            self._send(obj_masses)
            print('Set object masses')

        # Make avatar layout and get image data
        av_layout, pass_cmd = self.make_avatar_layout()
        print('about to  send')
        self._send(av_layout)
        print('sent!')
        rcv = self.output_socket.recv_multipart()
        print('Received Avatar Layout response')


        self._send(pass_cmd)
        print('Set pass mask')
        rcv = self.output_socket.recv_multipart()
        print('Received pass mask response')

        # Get segmentation colors
        print("Getting segmentation colors")
        self._send({"$type": "get_objects_data"})
        #pdb.set_trace()
        self._observe_world(exp_num_msgs=2)

        # Rotate baby agent to random orientation, tilt its head down, and toggle off its follow camera

        # Sample initial avatar angle
        ang = random.choice([45., 135., 225., 315.]) if self.use_det_reset else random.randint(1, 359)
        #ang = 225.
        #print("Reset angle: {}".format(ang))

        # Set initial avatar rotation and camera angle
        self._send([{"$type": "rotate_avatar_to","avatar_id": self.baby_id, "env_id": 0,
                     "rotation": {"x": 0.0, "y": ang, "z": 0.0}},
                    {"$type": "rotate_sensor_container_to", "avatar_id": self.baby_id, "env_id": 0,
                     "rotation": {"x": 27.0, "y": 0, "z": 0}},
                    {"$type": "toggle_sensor", "sensor_name": "FollowCamera", "avatar_id": self.baby_id, "env_id": 0}])
        self._observe_world()
        #print("Set initial avatar configuration")

        # Test action commands
        #print("Testing env.step()")
        action_dict = {'scripted_agents': [{'$type': 'move_avatar_forward_by',
                                            'magnitude': 0.0, 'avatar_id': 'a0', 'env_id': 0}]}
        obs, _ = self.step(action_dict)

        # Test step (action + observation)
        print('Reset Test')
        reset_try_counter = 0
        while np.absolute(self.avatar_rotation - ang) > 0.1:
            self._send({"$type": "rotate_avatar_to", "rotation": {"x": 0.0, "y": ang, "z": 0.0}, "avatar_id": self.baby_id, "env_id": 0})
            obs = self._observe_world()
            #print(self.avatar_rotation)
            reset_try_counter += 1
        print('SUCCESS (Reset angle: {}, Observed angle: {}, Reset Attempts: {})'.format(ang, self.avatar_rotation, reset_try_counter))
        print('_________________________________________')

        # Make sure again that scene is loaded properly
        assert np.absolute(self.avatar_rotation - ang) < 0.1

        return obs

    def _quit(self):
        '''
        quit: Close the input and output sockets and destroy the context
        '''
        self.tc.quit()

    def _termination_condition(self):
        '''
        pprint.pprint(self.observation)
        # check if avatar in bounds
        avatar_pos = self.observation['avatar_data']['position']
        avatar_pos = [float(avatar_pos[pos]) for pos in ['x', 'y', 'z']]
        wall_len = 3.
        tol = 0.3
        room_bounds = [wall_len*self.room_dims[0]/2.+tol, -0.1, wall_len*self.room_dims[1]/2.+tol]
        term_cond = (np.abs(avatar_pos[0]) >= room_bounds[0]) or (np.abs(avatar_pos[2]) >= room_bounds[2]) or (avatar_pos[1] <= room_bounds[1])
        if term_cond:
            print('Term condition reached...')
            logger.debug('Avatar at {}, bounds are {}'.format(avatar_pos, room_bounds))
        '''

        term_cond = False

        return term_cond

    def step(self, action_dict):
        '''
        step: Takes in action. Converts action into a message sent to the Unity "build".
        '''

        if self.act_repeat:  # action repeat
            step_action = [{"$type": "step_physics", "frames": 1}]
            action_list = []
            #[act, step, act, step, act, step, act]
            for i in range(self.num_act_repeat):
                action_list += action_dict.get('scripted_agents', [])
                if action_dict.get('curious_agent', None) is not None:
                    self.encoded_message = self._action_to_message(action_dict['curious_agent'])
                    action_list += self.encoded_message

                if i < (self.num_act_repeat - 1):
                    action_list += step_action

            action_list.append({"$type": "get_objects_data"})
            self._send(action_list)
            observation = self._observe_world(exp_num_msgs=2)

        termination_signal = self._termination_condition()

        return observation, termination_signal


class Action:
    def __init__(self, act_vec, mapping_dict={'move': 0, 'turn': 1, 'tilt': None, 'pan': None, 'bend_right_arm': None, 'bend_left_arm': None}):
        self.act_vec = act_vec
        self.mapping_dict = mapping_dict
        self.act_dict = self._action_vec_to_dict()
        self.assign_vars_from_act_dict()

    def _action_vec_to_dict(self):
        act_dict = {}
        for act, idx in self.mapping_dict.items():
            if idx is not None:
                assert isinstance(idx, int), 'Values in mapping_dict must be integer, but is type {}'.format(type(v))
                assert idx < len(self.act_vec), 'Index v, {}, must be less than length of act_vec, {}'.format(idx, len(self.act_vec))
                act_dict[act] = self.act_vec[idx]
        return act_dict

    def assign_vars_from_act_dict(self):
        # for backwards compatability
        for act, amt in self.act_dict.items():
            assert isinstance(act, str)
            exec('self.' + act + 'amt='+str(amt))


if __name__ == '__main__':
    environment_params = {
        'render_images' : False,
        'host_name' : 'localhost',
        'build_path' : None,
        'server_path' : None,
        'act_repeat' : True,
        'body_pose_dim' : 78,
        'action_space' : ['tilt'], #['move', 'turn', 'tilt', 'pan'],
        'continuous_actions' : True,
        'objects': ["kettle", "kettle", "kettle"],
        'avatars': ["A_StickyMitten_Baby", "A_Simple_Body", "A_Simple_Body", "A_Simple_Body", "A_Simple_Body"],
        'avatar_ids': {"a_baby_id": 0, "a_subanimate_id": 1, "a_periodic_id": 2, "a_random_id": 3, "a_static_id": 4},
        'object_quadrants': ['BL', 'BR', 'TL', 'TR'],
        'quadrants': ['BL', 'TR', 'TL', 'BR'],
        'room_dims' : (3, 3),
        'avatar_colors': [[1.,1.,0], [0,0,1.], [0,1.,0], [1.,0,0]],
        'step_size': 4.0,
        'ball_scale': 0.2,
        'screen_height': 300,
        'screen_width': 300,
        'render_port': 3001,
        'ball_y': 0.2,
        'rotation_step_size': 8,
        'goal_noise_thresh': 1.0,
        'path_noise_thresh': 1.0,
        'use_det_layout': 1,
        'object_masses': '100000',
        'use_server': True
       }

    #keys = [(0, 'w', 1), (0, 's', -1), (1, 'a', -1), (1, 'd', 1), (2, 'q', -1), (2, 'e', 1), (3, 'r', -1), (3, 'f', 1)]
    keys = [(0, 'w', 1), (0, 's', -1)]
    import ipdb
    ipdb.set_trace()
    env = Environment(environment_params)
    env.reset()
    running = True
    while running:
        action = [0] * len(environment_params['action_space']) * 2
        i = raw_input("q=quit \n> ")
        #i = random.choice(['d', 'a', 'w', 's'])
        if i == 'q':
            env._quit()
            running = False
        for (idx, key, val) in keys:
            if i == key:
                action[idx] = val
        if running:
            obs, term = env.step(action)
            if term:
                print('Term condition reached...')
                env.reset()
