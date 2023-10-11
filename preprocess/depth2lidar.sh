set -e
exeFunc(){
    num_seq=$1
    python utils/depth2lidar.py --calib_dir  ~/MonoOcc/kitti/dataset/sequences/$num_seq \
    --depth_dir ~/MonoOcc/preprocess/mobilestereonet/depth/sequences/$num_seq \
    --save_dir ~/MonoOcc/preprocess/mobilestereonet/lidar/sequences/$num_seq

    cp data_odometry_calib/sequences/$num_seq/calib.txt ~/MonoOcc/preprocess/mobilestereonet/lidar/sequences/$num_seq/
    cp data_odometry_calib/sequences/$num_seq/poses.txt ~/MonoOcc/preprocess/mobilestereonet/lidar/sequences/$num_seq/
}

# mkdir -p $data_path/lidar
# ln -s $data_path/lidar ./mobilestereonet/lidar
for i in {00..21}
do
    exeFunc $i
done
