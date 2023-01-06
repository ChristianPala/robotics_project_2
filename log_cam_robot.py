# IMPORTANT: run this script!
# Executing line by line prevents correct saving of bag file

import open3d as o3d
from fanucpy import Robot
import numpy as np
import json
import time
import pickle


if __name__ == '__main__':
    # Recording time
    tmax = 10

    # Prepare table to collect joint positions
    joint_angles = np.zeros([1, 6])

    # Data of robot to communicate with
    robot = Robot(
        robot_model="Fanuc",
        host="192.168.1.10",
        port=18735,
        ee_DO_type="RDO",
        ee_DO_num=7,
    )

    # Open TCP/IP connection
    robot.connect()

    # Open configuration file of the camera
    with open("rs_d455_config.json") as cf:
        rs_cfg = o3d.t.io.RealSenseSensorConfig(json.load(cf))

    # Initialize camera with saving to bag file
    rs = o3d.t.io.RealSenseSensor()
    ret = rs.init_sensor(rs_cfg, 0, "bag_record.bag")

    print (f"Sensor initialized? {ret}")

    # Image + joints capture cycle
    rs.start_capture(True)
    start_time = time.time()
    while (time.time() - start_time) < tmax:
        im_rgbd = rs.capture_frame(True, True)  # wait for frames and align them
        joints_now = np.expand_dims(np.array(robot.get_curjpos()), 0) # read robot joints via TCP/IP
        joint_angles = np.vstack([joint_angles, joints_now])
    rs.stop_capture()

    robot.disconnect()

    # Save joints data (bag file is saved during acquisition)
    file = open("joint_angles", 'wb')
    pickle.dump(joint_angles[1:, :], file)
    # close the file
    file.close()
