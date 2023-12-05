#!/usr/bin/env python
import rospy
from std_msgs.msg import Float64MultiArray
from movement_datasets import read_fep_dataset

pub = rospy.Publisher('position', Float64MultiArray, queue_size=10)  # TOPIC
rospy.init_node('waypoints_publisher', anonymous=True)
rate = rospy.Rate(1000)  # 1000hz
my_msg = Float64MultiArray()
index = 0

DTYPE = 'float64'
FEP_MOVEMENT_DATASET_PATH = "./movement_datasets/fep_state_to_pid-corrected-torque_55s_dataset.csv"
pos, _, _, _ = read_fep_dataset(FEP_MOVEMENT_DATASET_PATH, DTYPE)
index = 0

while not rospy.is_shutdown():
    my_msg.data = pos[index]
    pub.publish(my_msg)
    rate.sleep()
    index += 1
    index %= pos.shape[0]