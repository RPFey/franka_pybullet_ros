#!/usr/bin/env python
import rospy
from std_msgs.msg import Float64MultiArray, Bool
from sensor_msgs.msg import JointState
import tf2_ros
from geometry_msgs.msg import TransformStamped

import os
import numpy as np
import pybullet as p
import pybullet_data

from panda_robot import FrankaPanda
from bullet_client import BulletClient


class FrankaPandaEnv:
    def __init__(self, connection_mode=p.GUI, frequency=1000.,
                 include_gripper=True, simple_model=False, object_from_sdf=False, object_from_list=False):

        self.bc = BulletClient(connection_mode)
        self.bc.configureDebugVisualizer(rgbBackground=[33. / 255., 90. / 255., 127. / 255.])
        # self.bc.configureDebugVisualizer(self.bc.COV_ENABLE_GUI, 0)
        self.bc.configureDebugVisualizer(self.bc.COV_ENABLE_WIREFRAME, 1)
        self.frequency = frequency

        if object_from_sdf or object_from_list:
            self.plane_id = self.bc.loadURDF(os.path.join(pybullet_data.getDataPath(), "plane.urdf"),
                                             basePosition=[0, 0, -0.65], useFixedBase=True)
            self.bc.changeVisualShape(self.plane_id, -1, rgbaColor=[1., 1., 1., 0.])
            self.table_id = self.bc.loadURDF(os.path.join(pybullet_data.getDataPath(), "table/table.urdf"),
                                             basePosition=[0.5, 0, -0.65], useFixedBase=True)
            self.object_id = []

            if object_from_list:
                self.object_list = ["YcbBanana", "YcbPear", "YcbHammer", "YcbScissors", "YcbStrawberry", "YcbChipsCan",
                                    "YcbCrackerBox", "YcbFoamBrick", "YcbGelatinBox", "YcbMasterChefCan",
                                    "YcbMediumClamp",
                                    "YcbMustardBottle", "YcbPottedMeatCan", "YcbPowerDrill", "YcbTennisBall",
                                    "YcbTomatoSoupCan"]
                self.add_ycb_objects_from_list(self.object_list)
            elif object_from_sdf:
                self.add_ycb_objects_from_sdf('./grasp_sdf_env/layout_1.sdf')

        self.panda_robot = FrankaPanda(self.bc, include_gripper=include_gripper, simple_model=simple_model)

        rospy.init_node('franka_viz', anonymous=True)
        self.tfBuffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tfBuffer)
        self.joint_state_sub = rospy.Subscriber('joint', JointState, self.get_joint_state)
        self.rate = rospy.Rate(frequency)

        self.current_joint_state = self.panda_robot.home_joint

    def get_joint_state(self, reading):
        self.current_joint_state = reading.position

    def simulate_step(self):
        current_joint = self.current_joint_state
        for i in range(len(current_joint)):
            self.bc.resetJointState(self.panda_robot.robot_id, self.panda_robot.joints[i], targetValue=current_joint[i])

        for id in self.object_id:
            try:
                trans = self.tfBuffer.lookup_transform('world', str(id), rospy.Time())
                pos = [trans.transform.translation.x, trans.transform.translation.y, trans.transform.translation.z]
                orn = [trans.transform.rotation.x, trans.transform.rotation.y, trans.transform.rotation.z,
                       trans.transform.rotation.w]
                self.bc.resetBasePositionAndOrientation(id, pos, orn)
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                continue
        self.get_image()

        self.rate.sleep()

    def simulation_loop(self):
        import time
        while not rospy.is_shutdown():
            start = time.time()
            self.simulate_step()
            print(1 / (time.time() - start))
        self.bc.disconnect()
        print("Simulation end")

    def add_urdf_object(self, filename):
        flags = self.bc.URDF_USE_INERTIA_FROM_FILE
        self.object_id.append(self.bc.loadURDF(filename, [0.5, 0.0, 0.4], flags=flags))

    def add_ycb_objects_from_list(self, object_list):
        self.bc.configureDebugVisualizer(self.bc.COV_ENABLE_RENDERING, 0)
        print("Loading objects")
        for obj in object_list:
            self.add_urdf_object("./ycb_objects/" + obj + "/model.urdf")
            for i in range(200):
                self.bc.stepSimulation()
        self.bc.configureDebugVisualizer(self.bc.COV_ENABLE_RENDERING, 1)

    def add_ycb_objects_from_sdf(self, object_sdf):
        self.bc.configureDebugVisualizer(self.bc.COV_ENABLE_RENDERING, 0)
        print("Loading objects")
        object_id_temp = self.bc.loadSDF(object_sdf)
        for id in object_id_temp:
            pos, orn = self.bc.getBasePositionAndOrientation(id)
            pos = (pos[0] + 0.9, pos[1] - 0.15, pos[2])
            self.bc.resetBasePositionAndOrientation(id, pos, orn)
        self.object_id += object_id_temp
        self.bc.configureDebugVisualizer(self.bc.COV_ENABLE_RENDERING, 1)

    def get_image(self):
        pos = [-1., 0., 0.5]
        ori = [0., 0., 0., 1.]

        rot_matrix = self.bc.getMatrixFromQuaternion(ori)
        rot_matrix = np.array(rot_matrix).reshape(3, 3)

        # Initial vectors
        init_camera_vector = (1, 0, 0)  # z-axis
        init_up_vector = (0, 0, 1)  # y-axis

        # Rotated vectors
        camera_vector = rot_matrix.dot(init_camera_vector)
        up_vector = rot_matrix.dot(init_up_vector)

        self.projectionMatrix = self.bc.computeProjectionMatrixFOV(fov=58, aspect=1.5, nearVal=0.02, farVal=5)

        view_matrix_gripper = self.bc.computeViewMatrix(pos, pos + 0.1 * camera_vector, up_vector)
        img = self.bc.getCameraImage(640, 360, view_matrix_gripper, self.projectionMatrix,
                                     renderer=self.bc.ER_BULLET_HARDWARE_OPENGL,
                                     flags=self.bc.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX)

        return img
