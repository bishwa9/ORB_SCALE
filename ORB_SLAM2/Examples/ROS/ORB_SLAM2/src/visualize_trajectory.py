#!/usr/bin/python
import rospy
import numpy as np

from geometry_msgs.msg import PoseStamped, PoseArray, Pose

import sys

# Timestamp.toSecs() X Y Z q0 q1 q2 q3
data_size = 8

pub = rospy.Publisher(sys.argv[1], PoseArray, queue_size=10)

rospy.init_node('orb_slam_'+sys.argv[1].split('/')[-1])

disp_factor = 10

## Get the data from the file
def read_data(filename):
    print "File:", filename
    fp = open(filename, 'r')
    tmp_data = np.zeros(data_size)
    data = []
    for line in fp:
        tmp_data = [ float(n) for n in line.split(" ")]
        data.append(tmp_data)
    return data

## As sequence of poses
## Publish as poses
def publish_sequence(data, pub):
    pose_msg = PoseStamped()
    r = rospy.Rate(5)
    for d in data:
        pose_msg = PoseStamped()
        # Timestamp
        pose_msg.header.stamp = rospy.Time.from_sec(d[0])
        pose_msg.header.frame_id = 'map'
        # Position
        pose_msg.pose.position.x = d[1]*disp_factor
        pose_msg.pose.position.y = d[2]*disp_factor
        pose_msg.pose.position.z = d[3]*disp_factor
        # Orientation
        pose_msg.pose.orientation.x = d[4]
        pose_msg.pose.orientation.y = d[5]
        pose_msg.pose.orientation.z = d[6]
        pose_msg.pose.orientation.w = d[7]

        pub.publish(pose_msg)
        r.sleep()

def read_all_datas(filenames):
    datas = []
    for fn in filenames:
        datas.append(read_data(fn))
    return datas

## get pose array msg
def publish_pose_array_msg(data, pub):
    # Timestamp
    posarray_msg = PoseArray()
    posarray_msg.header.stamp = rospy.Time.from_sec(data[0][0])
    posarray_msg.header.frame_id = 'map'
    r = rospy.Rate(5)
    while not rospy.is_shutdown():
        for d in data:
            pose = Pose()
            # Position
            pose.position.x = d[1]*disp_factor
            pose.position.y = d[2]*disp_factor
            pose.position.z = d[3]*disp_factor
            # Orientation
            pose.orientation.x = d[4]
            pose.orientation.y = d[5]
            pose.orientation.z = d[6]
            pose.orientation.w = d[7]

            posarray_msg.poses.append(pose)
        pub.publish(posarray_msg)
        r.sleep()

def publish_all_pose_arrays(data_arr, pubs):
    for i in range(len(data_arr)):
        publish_pose_array_msg(data_arr[i], pubs[i])

def main():
    publish_pose_array_msg(read_data(sys.argv[2]), pub)
    rospy.spin()

if __name__ == '__main__':
    main()