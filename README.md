# ORB_SCALE


## Build Instructions
```
cd ORB_SLAM2
sudo chmod +x build.sh
./build.sh
cd Examples/ROS/ORB_SLAM2/
mkdir build
cd build
cmake .. -DROS_BUILD_TYPE=Release
make -j
```


## Run Instructions
-- Copy data (bag file) into ORB_SLAM2/data/
```
cd ORB_SLAM2/Examples/ROS/
export ROS_PACKAGE_PATH=${ROS_PACKAGE_PATH}:`pwd`
cd ORB_SLAM2/
roslaunch launch/orb_scale.launch
```

## Visualization of trajectories using rviz
```
cd ORB_SLAM2/Examples/ROS/
export ROS_PACKAGE_PATH=${ROS_PACKAGE_PATH}:`pwd`
cd ORB_SLAM2/
roslaunch launch/visualize_trajectories.launch traj_mono:=<mono_trajectory_file.txt> traj_mono_scaled:=<mono_scaled_trajectory_file.txt> traj_rgbd:=<rgbd_trajectory_file.txt>
```
-- Rviz displays: Green-Monocular. Red-Monocular,Scaled. Blue-RGBD
