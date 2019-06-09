import subprocess
import socket
from contextlib import closing
import os
import sys
import time
import zmq
from threading import Thread
import json
import signal
import tempfile
from pdb import set_trace


def format_args_python(arg_dict):
    """
    Takes in a dictionary of kv pairs and return the formatted command line string that is accepted by python arg parser
    """
    formatted_args = []
    for key, value in arg_dict.items():
        formatted_args += ["-" + str(key)]
        if(type(value) == list):
            formatted_args += [str(v) for v in value]
        else:
            formatted_args += [str(value)]
    return formatted_args


def format_args_unity(arg_dict):
    """
    Takes in a dictionary of kv pairs and return the formatted command line string that is accepted by unity arg parser
    """
    formatted_args = []
    for key, value in arg_dict.items():
        prefix = "-" + key + "="
        if(type(value) == list):
            prefix += ",".join([str(v) for v in value])
        else:
            prefix += str(value)
        formatted_args += [prefix]
    return formatted_args


def find_free_port():
    """
    Returns a free port as a string.
    """
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        return int(s.getsockname()[1])


class TDWClient():
    def __init__(self,
                 host_name,
                 build_path,
                 server_path,
                 render_images,
                 controller_id,
                 build_id,
                 screenHeight,
                 screenWidth,
                 render_port):

        self.build_path = build_path
        self.server_path = server_path
        self.render_images = render_images
        self.controller_id = controller_id
        self.build_id = build_id
        self.output_sock = None
        self.input_sock = None
        self.render_sock = None

        # Get ports
        #self.render_port = 1399  # Just so that I don't have to change render port each time.
        self.render_port = render_port
        if build_path is not None:
            self.build_to_server_ports = find_free_port()
            self.server_to_build_port = find_free_port()
            self.controller_to_server_port = find_free_port()
            self.server_to_controller_port = find_free_port()
        else:
            self.build_to_server_ports = 1341
            self.server_to_build_port = 1337
            self.controller_to_server_port = 1339
            self.server_to_controller_port = 1340

        self.host_address = 'tcp://' + host_name + ':'
        # Start the server
        if server_path is not None:
            server_args = format_args_python({
                "B": self.server_to_build_port,
                "b": self.build_to_server_ports,
                "c": self.controller_to_server_port,
                "C": self.server_to_controller_port})
            print("Server args:")
            print(server_args)
            f_server = tempfile.NamedTemporaryFile()
            if not os.path.isfile(server_path):
                raise Exception('Server path ill specified')
            proc = subprocess.Popen([sys.executable, server_path] + server_args, close_fds=True, stdout=f_server)
            print(f_server.name)
            self.server_pid = proc.pid
            print("Server started with PID:{}".format(self.server_pid))

        if build_path is not None:  # Using build instead of editor
            build_args = format_args_unity({
                "screenWidth": str(screenWidth),
                "screenHeight": str(screenHeight),
                "sendToServerPort": self.build_to_server_ports,
                "recvFromServerPort": self.server_to_build_port,
                "buildID": self.build_id,
            })
            print("Build args:")
            print(build_args)
            proc = subprocess.Popen([build_path] + build_args, close_fds=True, stdout=subprocess.PIPE)
            self.build_pid = proc.pid
            print("Build started with PID:{}".format(self.build_pid))
        time.sleep(5)

    def get_ports(self):
        return self.build_to_server_ports, self.server_to_build_port, self.controller_to_server_port, self.server_to_controller_port, self.render_port

    def get_output_socket(self):
        """
        Returns socket connected to online or initializing environment.
        """
        self.output_ctx = zmq.Context()
        self.output_sock = self.output_ctx.socket(zmq.DEALER)
        self.output_sock.setsockopt(zmq.IDENTITY, self.controller_id)
        self.output_sock.setsockopt(zmq.LINGER, 0)
        self.output_sock.connect(self.host_address + str(self.server_to_controller_port))
        return self.output_sock

    def get_input_socket(self):
        self.input_ctx = zmq.Context()
        self.input_sock = self.input_ctx.socket(zmq.DEALER)
        self.input_sock.setsockopt(zmq.IDENTITY, self.controller_id)
        self.input_sock.setsockopt(zmq.LINGER, 0)
        self.input_sock.connect(self.host_address + str(self.controller_to_server_port))
        return self.input_sock

    def get_render_socket(self):
        render_address = "tcp://*:"
        self.render_ctx = zmq.Context()
        self.render_sock = self.render_ctx.socket(zmq.REP)
        self.render_sock.bind(render_address + str(self.render_port))
        return self.render_sock

    def close_input_socket(self):
        self.input_sock.close()
        self.input_ctx.destroy()
        self.input_sock = None

    def close_output_socket(self):
        self.output_sock.close()
        self.output_ctx.destroy()
        self.output_sock = None

    def close_render_socket(self):
        self.render_sock.close()
        self.render_ctx.destroy()
        self.render_sock = None

    def quit(self):
        """
        Closes all sockets, stops the server, kills the environment process.
        """
        self.close_output_socket()
        if self.render_images:
            self.close_render_socket()
        if self.input_sock:
            self.close_input_socket()
        print("Closed sockets.")
        if self.server_path is not None:
            try:
                os.kill(self.server_pid, signal.SIGKILL)
                print("Killed old server with PID: {}".format(self.server_pid))
            except:
                print("Could not kill old server with PID: %d. Already dead?" % self.server_pid)
        if self.build_path is not None:
            try:
                os.kill(self.build_pid, signal.SIGKILL)
                print("Killed old environment with PID: {}".format(self.build_pid))
            except:
                print ("Could not kill old environment with PID: %d. Already dead?" % self.build_pid)
        return self.output_sock, self.input_sock, self.render_sock
