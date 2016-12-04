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