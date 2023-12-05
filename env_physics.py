#!/usr/bin/env python
import rospy
from std_msgs.msg import Float64MultiArray, Bool
from sensor_msgs.msg import JointState
import tf2_ros
from geometry_msgs.msg import TransformStamped

import os
import pybullet as p
import pybullet_data

from panda_robot import FrankaPanda
from bullet_client import BulletClient


class FrankaPandaEnv:
    def __init__(self, connection_mode=p.GUI, frequency=1000., controller='position',
                 include_gripper=True, simple_model=False, object_from_sdf=False, object_from_list=False):

        self.bc = BulletClient(connection_mode)
        self.bc.configureDebugVisualizer(rgbBackground=[33. / 255., 90. / 255., 127. / 255.])
        self.bc.configureDebugVisualizer(self.bc.COV_ENABLE_GUI, 0)
        self.frequency = frequency
        self.bc.setTimeStep(1 / frequency)
        self.bc.setGravity(0, 0, -9.81)

        if object_from_sdf or object_from_list:
            self.plane_id = self.bc.loadURDF(os.path.join(pybullet_data.getDataPath(), "plane.urdf"),
                                             basePosition=[0, 0, -0.65], useFixedBase=True)
            self.table_id = self.bc.loadURDF(os.path.join(pybullet_data.getDataPath(), "table/table.urdf"),
                                             basePosition=[0.5, 0, -0.65], useFixedBase=True)
            self.object_id = []

            if object_from_list:
                self.object_list = ["YcbBanana", "YcbPear", "YcbHammer", "YcbScissors", "YcbStrawberry", "YcbChipsCan",
                                    "YcbCrackerBox", "YcbFoamBrick", "YcbGelatinBox", "YcbMasterChefCan", "YcbMediumClamp",
                                    "YcbMustardBottle", "YcbPottedMeatCan", "YcbPowerDrill", "YcbTennisBall",
                                    "YcbTomatoSoupCan"]
                # self.object_list = ["YcbGelatinBox"]
                self.add_ycb_objects_from_list(self.object_list)
            elif object_from_sdf:
                self.add_ycb_objects_from_sdf('./grasp_sdf_env/layout_1.sdf')

        self.panda_robot = FrankaPanda(self.bc, include_gripper=include_gripper, simple_model=simple_model)

        rospy.init_node('franka', anonymous=True)
        self.controller = controller
        rospy.Subscriber(self.controller, Float64MultiArray, self.get_joint_input)
        rospy.Subscriber('gripper', Bool, self.get_gripper_input)
        self.joint_state_pub = rospy.Publisher('joint', JointState, queue_size=10)
        self.object_tf_broadcaster = tf2_ros.StaticTransformBroadcaster()
        self.rate = rospy.Rate(frequency)

        if self.controller == 'position':
            self.current_joint_input = self.panda_robot.home_joint[:self.panda_robot.dof]
        else:
            self.current_joint_input = [0, 0, 0, 0, 0, 0, 0]
        self.current_gripper_input = True

        self.joint_data = JointState()
        self.joint_data.name = self.panda_robot.joint_names
        self.joint_data.header.frame_id = "franka_panda_joint"

    def get_joint_input(self, input):
        self.current_joint_input = input.data

    def get_gripper_input(self, input):
        self.current_gripper_input = input.data

    def simulate_step(self):
        pos_reading, vel_reading, force_reading, effort_reading = self.panda_robot.get_pos_vel_force_torque()
        self.joint_data.position = pos_reading
        self.joint_data.velocity = vel_reading
        self.joint_data.effort = effort_reading
        self.joint_data.header.stamp = rospy.get_rostime()

        if self.controller == 'position':
            self.panda_robot.set_target_positions(self.current_joint_input)
        elif self.controller == 'velocity':
            self.panda_robot.set_target_velocities(self.current_joint_input)
        elif self.controller == 'torque':
            self.panda_robot.set_target_torques(self.current_joint_input)

        self.joint_state_pub.publish(self.joint_data)
        if self.current_gripper_input:
            self.panda_robot.open_gripper()
        else:
            self.panda_robot.close_gripper()

        self.publish_object_tf()

        self.bc.stepSimulation()
        self.rate.sleep()

    def simulation_loop(self):
        import time
        while not rospy.is_shutdown():
            start = time.time()
            self.simulate_step()
            print(1/(time.time()-start))
        self.bc.disconnect()
        print("Simulation end")

    def add_urdf_object(self, filename):
        flags = self.bc.URDF_USE_INERTIA_FROM_FILE
        self.object_id.append(self.bc.loadURDF(filename, [0.5, 0.0, 0.4], flags=flags))
        # self.object_id.append(self.bc.loadURDF(filename, [0.55, 0.0, 0.47], flags=flags))

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

    def publish_object_tf(self):
        static_tf = TransformStamped()
        static_tf.header.stamp = rospy.Time.now()
        static_tf.header.frame_id = "world"
        for id in self.object_id:
            pos, orn = self.bc.getBasePositionAndOrientation(id)
            static_tf.child_frame_id = str(id)
            static_tf.transform.translation.x = pos[0]
            static_tf.transform.translation.y = pos[1]
            static_tf.transform.translation.z = pos[2]
            static_tf.transform.rotation.x = orn[0]
            static_tf.transform.rotation.y = orn[1]
            static_tf.transform.rotation.z = orn[2]
            static_tf.transform.rotation.w = orn[3]
            self.object_tf_broadcaster.sendTransform(static_tf)

