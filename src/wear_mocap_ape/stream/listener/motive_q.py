import logging
import queue
import time

import numpy as np


import wear_mocap_ape.utility.transformations as ts
import wear_mocap_ape.config as config
from wear_mocap_ape.experimental_modes.nat_net.mocap_data import MoCapData
from wear_mocap_ape.experimental_modes.nat_net.nat_net_client import NatNetClient
from wear_mocap_ape.utility.messaging import MOTIVE_BONE_IDS


class MotiveQListener:
    def __init__(self):

        self.__tag = "MOTIVE Q LISTENER"
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

        # the receive_new_frame puts mocap data from Motive into this queue for further processing
        self.__q = queue.Queue()

    def stream_loop(self):
        # Start up the streaming client now that the callbacks are set up.
        # This will run perpetually, and operate on a separate thread.
        logging.info("[{}] Start motive streaming client".format(self.__tag))
        try:
            is_running = self.__streaming_client.run()

            if not is_running:
                raise UserWarning("ERROR: Could not start streaming client.")

            time.sleep(1)
            if self.__streaming_client.connected() is False:
                raise UserWarning("ERROR: Could not connect properly.  Check that Motive streaming is on.")

            # loop forever and log regular Hz updates until an error occurs
            while True:
                time.sleep(5)
                logging.info(f"[{self.__tag}] {self.__frame_count / 5} Hz - queue size {self.__q.qsize()}")
                self.__frame_count = 0
        finally:
            self.__streaming_client.shutdown()

    def receive_new_frame(self, data_dict, mocap_data: MoCapData):
        """
        called when receiving a new frame from Motive. Appends the data to the internal queue and increases count
        for Hz estimations.
        :param data_dict: Dictionary from the NatNet Client
        """
        self.__frame_count += 1  # to keep track of the data processing rate per second

        # write bones into new dictionary and replace old one
        rb_dat = mocap_data.rigid_body_data

        # find the smallest ID
        if self.__smallest_id is None:
            if not rb_dat.rigid_body_list:
                return
            self.__smallest_id = min([x.id_num for x in rb_dat.rigid_body_list]) - 1
            logging.info(f"[{self.__tag}] smallest id {self.__smallest_id}")
            return

        # parse frame data into temporary dictionary
        nb = {}
        for rb in rb_dat.rigid_body_list:
            if rb.tracking_valid:
                nb[rb.id_num - self.__smallest_id] = [np.array(rb.pos), np.array(rb.rot)]

        # now, parse the dict into an array in the correct order
        data = []
        try:
            for k, v in MOTIVE_BONE_IDS.items():
                d = nb[v]
                data.append(ts.mocap_quat_to_global(d[1]))
                data.append(ts.mocap_pos_to_global(d[0]))
        except KeyError:
            # cancel if not all required IDs are present
            # logging.info(f"[{self.__tag}] skipped frame because bone was missing")
            return

        # add array to queue with timestamp
        self.__q.put([time.time(), np.hstack(data)])

        # it's FIFO, we remove old entries if the queue gets too large
        while self.__q.qsize() > 240:
            self.__q.get()

    def get_ground_truth(self, averaging: bool = False):
        """
        :param averaging: if True, it calculates the average of the last second of stored observations that
        weren't part of the previous call
        :return: rotations and positions of all MOTIVE_BONE_IDS as a numpy array
        """
        # get the first observation and abort if there is none
        fo = self.__q.get()
        if fo is None:
            return None

        all_d = [fo]
        while not self.__q.empty():
            # while more data is available
            all_d.append(self.__q.get())

        if not averaging:
            # no averaging -> simply return the most recent observation
            return all_d[-1][1]

        # only keep data not longer than a second old
        fil_d = []  # filtered data
        all_d.reverse()  # the queue is FIFO, so reverse to have the most recent observation at 0
        sts = all_d[0][0]  # most recent time stamp
        for i, (t_s, d) in enumerate(all_d):
            if sts - t_s < 1.0:
                fil_d.append(d)
            else:
                logging.warning(f"[{self.__tag}] Discarded data older than 1s")
                break

        # now all data is filtered and combined
        all_d = np.array(fil_d)
        # average quaternions and positions and produce final data
        avg_d = []
        for i, k in enumerate(MOTIVE_BONE_IDS.keys()):
            avg_d.append(ts.average_quaternions(all_d[:, (i * 7):(i * 7 + 4)]))
            avg_d.append(np.mean(all_d[:, (i * 7 + 4):(i * 7 + 7)], axis=0))

        logging.warning(f"[{self.__tag}] averaged {all_d.shape[0]} rows")
        return np.hstack(avg_d)

    @staticmethod
    def get_ground_truth_header():
        """
        descriptive labels for what get_ground_truth() returns
        :return:
        """
        header = []
        for k in MOTIVE_BONE_IDS.keys():
            header += [k + "_quat_g_w", k + "_quat_g_x", k + "_quat_g_y", k + "_quat_g_z"]
            header += [k + "_orig_g_x", k + "_orig_g_y", k + "_orig_g_z"]
        return header

    @staticmethod
    def gt_to_unity_message(gt_msg: np.array):

        # back to dict
        d = {}
        for i, (k, v) in enumerate(MOTIVE_BONE_IDS.items()):
            d[k] = [
                gt_msg[i * 7: i * 7 + 4],
                gt_msg[i * 7 + 4: i * 7 + 7]
            ]

        # limb rotations of interest
        hip_rot_g = d["Hips"][0]
        # rotations and positions of interest
        hand_rot_g, hand_orig_g = d["LeftHand"]
        uarm_rot_g, uarm_orig_g = d["LeftUpperArm"]
        larm_rot_g, larm_orig_g = d["LeftLowerArm"]

        # estimate rotations relative to hip
        hand_rot_rh = ts.hamilton_product(ts.quat_invert(hip_rot_g), hand_rot_g)
        larm_rot_rh = ts.hamilton_product(ts.quat_invert(hip_rot_g), larm_rot_g)
        uarm_rot_rh = ts.hamilton_product(ts.quat_invert(hip_rot_g), uarm_rot_g)

        # estimate positions relative to upper arm origin
        larm_origin_rua = ts.quat_rotate_vector(ts.quat_invert(hip_rot_g), np.array(larm_orig_g - uarm_orig_g))
        hand_orig_rua = ts.quat_rotate_vector(ts.quat_invert(hip_rot_g), np.array(hand_orig_g - uarm_orig_g))

        return np.hstack([hand_rot_rh, hand_orig_rua, larm_rot_rh, larm_origin_rua, uarm_rot_rh])
