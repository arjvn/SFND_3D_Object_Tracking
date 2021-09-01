#!/bin/bash

./3D_object_tracking SHITOMASI BRISK
./3D_object_tracking SHITOMASI BRIEF
./3D_object_tracking SHITOMASI ORB
./3D_object_tracking SHITOMASI FREAK
./3D_object_tracking SHITOMASI AKAZE
./3D_object_tracking SHITOMASI SIFT

./3D_object_tracking HARRIS BRISK
./3D_object_tracking HARRIS BRIEF
./3D_object_tracking HARRIS ORB
./3D_object_tracking HARRIS FREAK
./3D_object_tracking HARRIS AKAZE
./3D_object_tracking HARRIS SIFT

./3D_object_tracking FAST BRISK
./3D_object_tracking FAST BRIEF
./3D_object_tracking FAST ORB
./3D_object_tracking FAST FREAK
./3D_object_tracking FAST AKAZE
./3D_object_tracking FAST SIFT

./3D_object_tracking BRISK BRISK
./3D_object_tracking BRISK BRIEF
./3D_object_tracking BRISK ORB
./3D_object_tracking BRISK FREAK
./3D_object_tracking BRISK AKAZE
./3D_object_tracking BRISK SIFT

./3D_object_tracking ORB BRISK
./3D_object_tracking ORB BRIEF
./3D_object_tracking ORB ORB
./3D_object_tracking ORB FREAK
./3D_object_tracking ORB AKAZE
./3D_object_tracking ORB SIFT

./3D_object_tracking AKAZE BRISK
./3D_object_tracking AKAZE BRIEF
./3D_object_tracking AKAZE ORB
./3D_object_tracking AKAZE FREAK
./3D_object_tracking AKAZE AKAZE
./3D_object_tracking AKAZE SIFT

./3D_object_tracking SIFT BRISK
./3D_object_tracking SIFT BRIEF
./3D_object_tracking SIFT ORB
./3D_object_tracking SIFT FREAK
./3D_object_tracking SIFT AKAZE
./3D_object_tracking SIFT SIFT
