import copy
import logging
import time

import numpy as np

import utility.transformations as ts
import config
from experimental_modes.nat_net.mocap_data import MoCapData
from experimental_modes.nat_net.nat_net_client import NatNetClient
from utility.messaging import MOTIVE_BONE_IDS


class MotiveListener:
    def __init__(self):

        self.__tag = "MOTIVE LISTENER"
        self.__frame_count = 0  # for Hz estimations

        # stores position and rotation of bones of interest
        self.__cf = {}
        self.__smallest_id = None  # motive sets the skeleton body IDs incrementally. We reduce them to start with 1

        self.__streaming_client = NatNetClient()

        self.__streaming_client.set_client_address(config.IP_OWN)  # this machine
        self.__streaming_client.set_server_address(config.IP_MOTIVE_SERVER)  # motive machine
        self.__streaming_client.set_use_multicast(False)  # only works in unicast setting

        # Configure the streaming client to call our rigid body handler on the emulator to send data out.
        self.__streaming_client.new_frame_listener = self.receive_new_frame
        # Showing only received frame numbers and suppressing data descriptions
        self.__streaming_client.set_print_level(0)

    def stream_loop(self):
        # Start up the streaming client now that the callbacks are set up.
        # This will run perpetually, and operate on a separate thread.
        logging.info("[{}] Start motive streaming client".format(self.__tag))
        try:
            is_running = self.__streaming_client.run()

            if not is_running:
                raise UserWarning("ERROR: Could not start streaming client.")

            is_looping = True
            time.sleep(1)
            if self.__streaming_client.connected() is False:
                raise UserWarning("ERROR: Could not connect properly.  Check that Motive streaming is on.")

            # loop forever until an error occurs
            while is_looping:
                time.sleep(5)
                logging.info("[{}] {} Hz".format(self.__tag, self.__frame_count / 5))
                self.__frame_count = 0
        finally:
            self.__streaming_client.shutdown()

    def receive_new_frame(self, data_dict, mocap_data: MoCapData):
        """
        called when receiving a new frame. Overwrites the current frame and increases count for Hz estimations
        :param data_dict:
        :return:
        """

        self.__frame_count += 1

        # write bones into new dictionary and replace old one
        rb_dat = mocap_data.rigid_body_data

        # find the smallest ID
        if self.__smallest_id is None:
            if not rb_dat.rigid_body_list:
                return
            self.__smallest_id = min([x.id_num for x in rb_dat.rigid_body_list]) - 1
            logging.info(f"[{self.__tag}] smallest id {self.__smallest_id}")
            return

        # reset current frame and temporary dictionary nb
        nb = {}
        # parse frame data
        for rb in rb_dat.rigid_body_list:
            if rb.tracking_valid:
                nb[rb.id_num - self.__smallest_id] = [np.array(rb.pos), np.array(rb.rot)]
        # update current and valid frame
        self.__cf = nb

    def get_ground_truth(self):
        cb = copy.deepcopy(self.__cf)
        data = []
        try:
            for k, v in MOTIVE_BONE_IDS.items():
                d = cb[v]
                data.append(ts.mocap_quat_to_global(d[1]))
                data.append(ts.mocap_pos_to_global(d[0]))
        except KeyError:
            return None
        return np.hstack(data)

    @staticmethod
    def get_ground_truth_header():
        """
        descriptive labels for what get_ground_truth() returns
        :return:
        """
        header = []
        for k, v in MOTIVE_BONE_IDS.items():
            header += [k + "_quat_g_w", k + "_quat_g_x", k + "_quat_g_y", k + "_quat_g_z"]
            header += [k + "_orig_g_x", k + "_orig_g_y", k + "_orig_g_z"]
        return header

    def get_unity_message(self):
        cb = copy.deepcopy(self.__cf)
        try:
            # limb rotations of interest
            hip_rot_g = ts.mocap_quat_to_global(cb[MOTIVE_BONE_IDS["Hips"]][1])
            hand_rot_g = ts.mocap_quat_to_global(cb[MOTIVE_BONE_IDS["LeftHand"]][1])
            uarm_rot_g = ts.mocap_quat_to_global(cb[MOTIVE_BONE_IDS["LeftUpperArm"]][1])
            larm_rot_g = ts.mocap_quat_to_global(cb[MOTIVE_BONE_IDS["LeftLowerArm"]][1])

            # limb origins of interest
            uarm_orig_g = ts.mocap_pos_to_global(cb[MOTIVE_BONE_IDS["LeftUpperArm"]][0])
            larm_orig_g = ts.mocap_pos_to_global(cb[MOTIVE_BONE_IDS["LeftLowerArm"]][0])
            hand_orig_g = ts.mocap_pos_to_global(cb[MOTIVE_BONE_IDS["LeftHand"]][0])
        except KeyError:
            return None

        # estimate rotations relative to hip
        hand_rot_rh = ts.hamilton_product(ts.quat_invert(hip_rot_g), hand_rot_g)
        larm_rot_rh = ts.hamilton_product(ts.quat_invert(hip_rot_g), larm_rot_g)
        uarm_rot_rh = ts.hamilton_product(ts.quat_invert(hip_rot_g), uarm_rot_g)

        # estimate positions relative to upper arm origin
        larm_origin_rua = ts.quat_rotate_vector(ts.quat_invert(hip_rot_g), np.array(larm_orig_g - uarm_orig_g))
        hand_orig_rua = ts.quat_rotate_vector(ts.quat_invert(hip_rot_g), np.array(hand_orig_g - uarm_orig_g))

        return np.hstack([hand_rot_rh, hand_orig_rua, larm_rot_rh, larm_origin_rua, uarm_rot_rh, hip_rot_g])
