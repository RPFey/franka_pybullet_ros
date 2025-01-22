import pybullet as p
import numpy as np
import time

# get current directory
import os
dir_path = os.path.dirname(__file__)

class FrankaPanda:
    def __init__(self, bc, include_gripper=False, simple_model=False):
        self.bc = bc
        self.include_gripper = include_gripper

        # bring up the robot's URDF
        self.bc.configureDebugVisualizer(self.bc.COV_ENABLE_RENDERING, 0)
        self.bc.setAdditionalSearchPath(os.path.join(dir_path, 'model_description'))
        panda_model = "panda.urdf" if include_gripper else "panda_nohand.urdf"
        flags = (self.bc.URDF_USE_INERTIA_FROM_FILE | self.bc.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
                 | self.bc.URDF_USE_SELF_COLLISION | self.bc.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT)
        if simple_model:
            flags = flags | self.bc.URDF_MERGE_FIXED_LINKS
        self.robot_id = self.bc.loadURDF(panda_model, useFixedBase=True, flags=flags)
        if simple_model:
            for i in range(-1, self.bc.getNumJoints(self.robot_id)):
                self.bc.changeVisualShape(self.robot_id, i, rgbaColor=[248. / 255., 174. / 255., 201. / 255, 1.])
        self.bc.configureDebugVisualizer(self.bc.COV_ENABLE_RENDERING, 1)

        # hard-coded information about the robot
        self.dof = 7
        self.joints = (np.ones(9) * -1 if include_gripper else np.ones(7)).astype(int)
        self.joint_names = ['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4',
                            'panda_joint5', 'panda_joint6', 'panda_joint7',
                            'panda_finger_joint1', 'panda_finger_joint2']
        for i in range(self.bc.getNumJoints(self.robot_id)):
            joint_info = self.bc.getJointInfo(self.robot_id, i)
            joint_name = joint_info[1].decode()
            try:
                self.joints[self.joint_names.index(joint_name)] = joint_info[0]
            except:
                continue

        # enable force torque sensor for each movable joints
        for i in range(len(self.joints)):
            self.bc.enableJointForceTorqueSensor(self.robot_id, self.joints[i], 1)

        # bring the robot to its home position
        self.home_joint = [0, 0, 0, -1.57079, 0, 1.57079, -0.7853, 0.05, 0.05]
        self.reset_state()

        # initialize the gripper
        if include_gripper:
            c = self.bc.createConstraint(self.robot_id,
                                         self.joints[-2],
                                         self.robot_id,
                                         self.joints[-1],
                                         jointType=p.JOINT_GEAR,
                                         jointAxis=[0, 0, 1],
                                         parentFramePosition=[0, 0, 0],
                                         childFramePosition=[0, 0, 0])
            self.bc.changeConstraint(c, gearRatio=-1, maxForce=1000, erp=0.2)
            self.gripper_opening = True
            self.gripper_target_reached = True
            self.gripper_moving = False
            self.gripper_target_pos = self.home_joint[-2:]
            self.open_gripper()

        self.joint_effort = [0.] * self.dof

    def reset_state(self):
        print(f"self.joins: {self.joints}")
        print(f"getNumJoints: {self.bc.getNumJoints(self.robot_id)}")
        for i in range(self.bc.getNumJoints(self.robot_id)):
            self.bc.resetJointState(self.robot_id, i, targetValue=self.home_joint[i])
        # breakpoint()
        time.sleep(1)
        for i in range(self.bc.getNumJoints(self.robot_id)):
            print(f"Resetting joint {i}")
            self.bc.setJointMotorControl2(self.robot_id, i, self.bc.VELOCITY_CONTROL, force=0)
        # self.bc.setJointMotorControlArray(bodyUniqueId=self.robot_id,
        #                                   jointIndices=self.joints[:self.dof],
        #                                   controlMode=self.bc.VELOCITY_CONTROL,
        #                                   forces=[0. for _ in self.joints[:self.dof]])

    def get_pos_vel_force_torque(self):
        joint_states = self.bc.getJointStates(self.robot_id, self.joints)
        joint_pos = [state[0] for state in joint_states]
        joint_vel = [state[1] for state in joint_states]
        joint_force = [state[2] for state in joint_states]
        joint_effort = [state[3] for state in joint_states]
        if joint_effort[:self.dof] == [0.] * self.dof:
            joint_effort = self.joint_effort + joint_effort[self.dof:]
        return joint_pos, joint_vel, joint_force, joint_effort

    def calculate_inverse_kinematics(self, position, orientation):
        return self.bc.calculateInverseKinematics(self.robot_id, self.dof, position, orientation)

    def calculate_inverse_dynamics(self, pos, vel, desired_acc):
        assert len(pos) == len(vel) and len(vel) == len(desired_acc)
        pos = [pos[i] if i < 7 else 0 for i in range(len(pos))]
        vel = [vel[i] if i < 7 else 0 for i in range(len(vel))]
        simulated_torque = list(self.bc.calculateInverseDynamics(self.robot_id, pos, vel, desired_acc))
        return simulated_torque

    def set_target_positions(self, desired_pos):
        self.bc.setJointMotorControlArray(bodyUniqueId=self.robot_id,
                                          jointIndices=self.joints[:self.dof],
                                          controlMode=self.bc.POSITION_CONTROL,
                                          positionGains=[0.005 for _ in range(self.dof)],
                                          targetPositions=desired_pos[:self.dof])

    def set_target_velocities(self, desired_vel):
        self.bc.setJointMotorControlArray(bodyUniqueId=self.robot_id,
                                          jointIndices=self.joints[:self.dof],
                                          controlMode=self.bc.VELOCITY_CONTROL,
                                          targetVelocities=desired_vel[:self.dof])

    def set_target_torques(self, desired_torque):
        self.joint_effort = desired_torque[:self.dof]
        self.bc.setJointMotorControlArray(bodyUniqueId=self.robot_id,
                                          jointIndices=self.joints[:self.dof],
                                          controlMode=self.bc.TORQUE_CONTROL,
                                          forces=desired_torque[:self.dof])
        
    def close_gripper(self):
        try:
            assert self.include_gripper
        except AssertionError as e:
            print('The robot does not have a gripper')
            return
        
        if self.gripper_opening:
            self.gripper_opening = False
            self.gripper_moving = True
            self.gripper_target_reached = False
            self.gripper_target_pos = [0., 0.]

        if self.gripper_moving:
            joint_states = self.bc.getJointStates(self.robot_id, self.joints[-2:])
            joint_pos = np.array([state[0] for state in joint_states])
            joint_torque = np.array([state[3] for state in joint_states])
            self.gripper_target_pos = joint_pos

            if np.linalg.norm(joint_pos) < 1e-4:
                self.gripper_moving = False
                self.gripper_target_reached = True
            else:
                self.gripper_target_reached = False

            if joint_torque.min() <= -40.:
                self.gripper_moving = False

        if not self.gripper_moving:
            if self.gripper_target_reached:
                self.gripper_target_pos = [0., 0.]
                self.gripper_opening = True
                self.bc.setJointMotorControlArray(bodyIndex=self.robot_id,
                                                jointIndices=self.joints[-2:],
                                                controlMode=p.POSITION_CONTROL,
                                                targetPositions=self.gripper_target_pos,
                                                forces=[200, 200])
            else:
                # set the final force.
                # 
                self.bc.setJointMotorControlArray(bodyIndex=self.robot_id,
                                              jointIndices=self.joints[-2:],
                                              controlMode=p.TORQUE_CONTROL,
                                              forces=[150, 150])
            
        elif self.gripper_moving:
            self.bc.setJointMotorControlArray(bodyIndex=self.robot_id,
                                              jointIndices=self.joints[-2:],
                                              controlMode=p.VELOCITY_CONTROL,
                                              targetVelocities=[-0.10, -0.10],
                                              forces=[70, 70])

    def open_gripper(self):
        try:
            assert self.include_gripper
        except AssertionError as e:
            print('The robot does not have a gripper')
            return
        if not self.gripper_opening:
            self.gripper_opening = True
            self.gripper_moving = True
            self.gripper_target_reached = False
            self.gripper_target_pos = [0.05, 0.05]

        if self.gripper_moving:
            joint_states = self.bc.getJointStates(self.robot_id, self.joints[-2:])
            joint_pos = np.array([state[0] for state in joint_states])
            joint_torque = np.array([state[3] for state in joint_states])
            self.gripper_target_pos = joint_pos

            if np.linalg.norm(joint_pos - np.array([0.05, 0.05])) < 1e-4:
                self.gripper_moving = False
                self.gripper_target_reached = True
            else:
                self.gripper_target_reached = False

            if joint_torque.max() >= 40.:
                self.gripper_moving = False

        if not self.gripper_moving:
            if self.gripper_target_reached:
                self.gripper_target_pos = [0.05, 0.05]
            self.bc.setJointMotorControlArray(bodyIndex=self.robot_id,
                                              jointIndices=self.joints[-2:],
                                              controlMode=p.POSITION_CONTROL,
                                              targetPositions=self.gripper_target_pos,
                                              forces=[200, 200])
        else:
            self.bc.setJointMotorControlArray(bodyIndex=self.robot_id,
                                              jointIndices=self.joints[-2:],
                                              controlMode=p.VELOCITY_CONTROL,
                                              targetVelocities=[0.05, 0.05],
                                              forces=[70, 70])
