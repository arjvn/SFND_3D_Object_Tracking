
/* INCLUDES FOR THIS PROJECT */
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <limits>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "dataStructures.h"
#include "matching2D.hpp"
#include "objectDetection2D.hpp"
#include "lidarData.hpp"
#include "camFusion.hpp"

using namespace std;

/* MAIN PROGRAM */
int main(int argc, const char *argv[]) {
  /* INIT VARIABLES AND DATA STRUCTURES */

  // data location
  string dataPath = "../";

  // camera
  string imgBasePath = dataPath + "images/";
  string imgPrefix = "KITTI/2011_09_26/image_02/data/000000"; // left camera, color
  string imgFileType = ".png";
  int imgStartIndex = 0; // first file index to load (assumes Lidar and camera names have identical naming convention)
  int imgEndIndex = 18;   // last file index to load
  int imgStepWidth = 1;
  int imgFillWidth = 4;  // no. of digits which make up the file index (e.g. img-0001.png)

  // object detection
  string yoloBasePath = dataPath + "dat/yolo/";
  string yoloClassesFile = yoloBasePath + "coco.names";
  string yoloModelConfiguration = yoloBasePath + "yolov3.cfg";
  string yoloModelWeights = yoloBasePath + "yolov3.weights";

  // Lidar
  string lidarPrefix = "KITTI/2011_09_26/velodyne_points/data/000000";
  string lidarFileType = ".bin";

  // calibration data for camera and lidar
  cv::Mat P_rect_00(3,4,cv::DataType<double>::type); // 3x4 projection matrix after rectification
  cv::Mat R_rect_00(4,4,cv::DataType<double>::type); // 3x3 rectifying rotation to make image planes co-planar
  cv::Mat RT(4,4,cv::DataType<double>::type); // rotation matrix and translation vector

  RT.at<double>(0,0) = 7.533745e-03; RT.at<double>(0,1) = -9.999714e-01; RT.at<double>(0,2) = -6.166020e-04; RT.at<double>(0,3) = -4.069766e-03;
  RT.at<double>(1,0) = 1.480249e-02; RT.at<double>(1,1) = 7.280733e-04; RT.at<double>(1,2) = -9.998902e-01; RT.at<double>(1,3) = -7.631618e-02;
  RT.at<double>(2,0) = 9.998621e-01; RT.at<double>(2,1) = 7.523790e-03; RT.at<double>(2,2) = 1.480755e-02; RT.at<double>(2,3) = -2.717806e-01;
  RT.at<double>(3,0) = 0.0; RT.at<double>(3,1) = 0.0; RT.at<double>(3,2) = 0.0; RT.at<double>(3,3) = 1.0;

  R_rect_00.at<double>(0,0) = 9.999239e-01; R_rect_00.at<double>(0,1) = 9.837760e-03; R_rect_00.at<double>(0,2) = -7.445048e-03; R_rect_00.at<double>(0,3) = 0.0;
  R_rect_00.at<double>(1,0) = -9.869795e-03; R_rect_00.at<double>(1,1) = 9.999421e-01; R_rect_00.at<double>(1,2) = -4.278459e-03; R_rect_00.at<double>(1,3) = 0.0;
  R_rect_00.at<double>(2,0) = 7.402527e-03; R_rect_00.at<double>(2,1) = 4.351614e-03; R_rect_00.at<double>(2,2) = 9.999631e-01; R_rect_00.at<double>(2,3) = 0.0;
  R_rect_00.at<double>(3,0) = 0; R_rect_00.at<double>(3,1) = 0; R_rect_00.at<double>(3,2) = 0; R_rect_00.at<double>(3,3) = 1;

  P_rect_00.at<double>(0,0) = 7.215377e+02; P_rect_00.at<double>(0,1) = 0.000000e+00; P_rect_00.at<double>(0,2) = 6.095593e+02; P_rect_00.at<double>(0,3) = 0.000000e+00;
  P_rect_00.at<double>(1,0) = 0.000000e+00; P_rect_00.at<double>(1,1) = 7.215377e+02; P_rect_00.at<double>(1,2) = 1.728540e+02; P_rect_00.at<double>(1,3) = 0.000000e+00;
  P_rect_00.at<double>(2,0) = 0.000000e+00; P_rect_00.at<double>(2,1) = 0.000000e+00; P_rect_00.at<double>(2,2) = 1.000000e+00; P_rect_00.at<double>(2,3) = 0.000000e+00;

  // misc
  double sensorFrameRate = 10.0 / imgStepWidth; // frames per second for Lidar and camera
  int dataBufferSize = 2;       // no. of images which are held in memory (ring buffer) at the same time
  vector<DataFrame> dataBuffer; // list of data frames which are held in memory at the same time
  bool bVis = false;            // visualize results
  bool bLimitKpts = false;

  // Detector Parameters
  vector<cv::KeyPoint> keypoints; // create empty feature list for current image
  string detectorName = "SHITOMASI"; // SHITOMASI, HARRIS, FAST, BRISK, ORB, AKAZE, SIFT
  // std::cout << "(SHITOMASI, HARRIS, FAST, BRISK, ORB, AKAZE, SIFT)"<< '\n' << "Enter detector of choice: " << '\n';
  // std::cin >> detectorName;
  if (argc > 1) {
    detectorName = argv[1];
    // cout << "Setting the keypoint detector type based on the command line: " << detectorName << "\n";
  }

  // Descriptor Parameters
  cv::Mat descriptors;
  string descriptorName = "BRISK"; // BRISK, BRIEF, ORB, FREAK, AKAZE, SIFT
  // std::cout << "(BRISK, BRIEF, ORB, FREAK, AKAZE, SIFT)"<< '\n' << "Enter descriptor of choice: " << '\n';
  // std::cin >> descriptorName;
  if (argc > 2) {
      descriptorName = argv[2];
      // cout << "Setting the descriptor type based on the command line: " << descriptorName << "\n";
  }
  // check for AKAZE detector - ensure AKAZE keypoints
  if (descriptorName.compare("AKAZE") == 0 && detectorName.compare("AKAZE") != 0) {
    std::cerr << "Warning: Need AKAZE keypoints for AKAZA detector" << '\n';
    return 0;
  }

  // Matching Parameters
  vector<cv::DMatch> matches;
  string matcherType = "MAT_BF";        // MAT_BF, MAT_FLANN
  string descriptorType = "DES_BINARY"; // DES_BINARY, DES_HOG
  string selectorType = "SEL_KNN";       // SEL_NN, SEL_KNN

  // Performance evaluators
  double detectorTime = 0, descriptorTime = 0;
  vector<int> noKeypoints;
  vector<int> noMatches;
  vector<float> listDetectorTime;
  vector<float> listDescriptorTime;

  std::vector<double> CamTTC_Array;
  std::vector<double> TTCLidar_dist;
  std::vector<double> LidarTTC_Array;
  std::vector<double> TTCLidar_delta_d;
  std::vector<double> TTCLidar_vel;

// ################################################ PARAMETERS SET & DATA LOADED FROM KITTI ################################################ //

  /* MAIN LOOP OVER ALL IMAGES */
  int image = 1;
  for (size_t imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex; imgIndex+=imgStepWidth) {
    /* LOAD IMAGE INTO BUFFER */

    // assemble filenames for current index
    ostringstream imgNumber;
    imgNumber << setfill('0') << setw(imgFillWidth) << imgStartIndex + imgIndex;
    string imgFullFilename = imgBasePath + imgPrefix + imgNumber.str() + imgFileType;

    // load image from file
    cv::Mat img = cv::imread(imgFullFilename);

    // push image into data frame buffer
    DataFrame frame;
    frame.cameraImg = img;
    dataBuffer.push_back(frame);
    if(dataBuffer.size()>dataBufferSize){
      dataBuffer.erase(dataBuffer.begin());
    }
    std::cout << "\n" << "\n" << "IMAGE " << image << '\n';
    cout << "#1 : LOAD IMAGE INTO BUFFER: Done" << endl;
    /* USE YOLO TO DETECT & CLASSIFY OBJECTS */
    float confThreshold = 0.2;
    float nmsThreshold = 0.4;
    bVis = false;
    detectObjects((dataBuffer.end() - 1)->cameraImg, (dataBuffer.end() - 1)->boundingBoxes, confThreshold, nmsThreshold,
                  yoloBasePath, yoloClassesFile, yoloModelConfiguration, yoloModelWeights, bVis);

    cout << "#2 : DETECT & CLASSIFY OBJECTS WITH YOLO: Done" << endl;
    /* CROP LIDAR POINTS */
    // load 3D Lidar points from file
    string lidarFullFilename = imgBasePath + lidarPrefix + imgNumber.str() + lidarFileType;
    std::vector<LidarPoint> lidarPoints;
    loadLidarFromFile(lidarPoints, lidarFullFilename);

    // remove Lidar points based on distance properties
    float minZ = -1.5, maxZ = -0.9, minX = 2.0, maxX = 20.0, maxY = 2.0, minR = 0.1; // focus on ego lane
    cropLidarPoints(lidarPoints, minX, maxX, maxY, minZ, maxZ, minR);

    (dataBuffer.end() - 1)->lidarPoints = lidarPoints;
    cout << "#3 : CROP LIDAR POINTS: Done" << endl;

    /* CLUSTER LIDAR POINT CLOUD */

    // associate Lidar points with camera-based ROI
    float shrinkFactor = 0.10; // shrinks each bounding box by the given percentage to avoid 3D object merging at the edges of an ROI
    clusterLidarWithROI((dataBuffer.end()-1)->boundingBoxes, (dataBuffer.end() - 1)->lidarPoints, shrinkFactor, P_rect_00, R_rect_00, RT);

    cout << "#4 : CLUSTER LIDAR POINT CLOUD done" << endl;
    // REMOVE THIS LINE BEFORE PROCEEDING WITH THE FINAL PROJECT
    // continue; // skips directly to the next image without processing what comes beneath

// ################################################ VISUAL CAMERA ANALYSIS ################################################ //

    /* DETECT IMAGE KEYPOINTS */
    // convert current image to grayscale
    cv::Mat imgGray;
    double detectorTime = 0, descriptorTime = 0;
    cv::cvtColor((dataBuffer.end()-1)->cameraImg, imgGray, cv::COLOR_BGR2GRAY);
    bVis = false;
    if (detectorName.compare("SHITOMASI") == 0) {
      detKeypointsShiTomasi(keypoints, imgGray, detectorTime, bVis);
    }
    else if (detectorName.compare("HARRIS") == 0) {
      detKeypointsHarris(keypoints, imgGray, detectorTime, bVis);
    }
    else if (detectorName.compare("ORB") == 0) {
      detKeypointsOrb(keypoints, imgGray, detectorTime, bVis);
    }
    else if (detectorName.compare("FAST") == 0) {
      detKeypointsFast(keypoints, imgGray, detectorTime, bVis);
    }
    else if (detectorName.compare("BRISK") == 0) {
      detKeypointsBrisk(keypoints, imgGray, detectorTime, bVis);
    }
    else if (detectorName.compare("AKAZE") == 0) {
      detKeypointsAkaze(keypoints, imgGray, detectorTime, bVis);
    }
    else if (detectorName.compare("SIFT") == 0) {
      detKeypointsSift(keypoints, imgGray, detectorTime, bVis);
    }
    else {
      std::cerr << "Warning: Unkown keypoint detector provided" << '\n';
      return 0;
    }

    std::cout << detectorName << " detector used with " << keypoints.size() << " found in " << detectorTime << "ms."<< '\n';

    if (bLimitKpts) { // optional : limit number of keypoints (helpful for debugging and learning)
        int maxKeypoints = 50;

        if (detectorName.compare("SHITOMASI") == 0) {
          // there is no response info, so keep the first 50 as they are sorted in descending quality order
          keypoints.erase(keypoints.begin() + maxKeypoints, keypoints.end());
        }
        cv::KeyPointsFilter::retainBest(keypoints, maxKeypoints);
        cout << " NOTE: Keypoints have been limited!" << endl;
    }

    // push keypoints and descriptor for current frame to end of data buffer
    (dataBuffer.end() - 1)->keypoints = keypoints;

    cout << "#5 : DETECT KEYPOINTS done" << endl;

    /* EXTRACT KEYPOINT DESCRIPTORS */
    descKeypoints((dataBuffer.end() - 1)->keypoints, (dataBuffer.end() - 1)->cameraImg, descriptors, descriptorName, descriptorTime);
    // push descriptors for current frame to end of data buffer
    (dataBuffer.end() - 1)->descriptors = descriptors;

    std::cout << descriptorName << " descriptor used. " << descriptorTime<< "ms taken." << '\n';

    cout << "#6 : EXTRACT DESCRIPTORS done" << endl;

    noKeypoints.push_back(keypoints.size());
    listDetectorTime.push_back(detectorTime);
    listDescriptorTime.push_back(descriptorTime);

    if (dataBuffer.size() > 1) // wait until at least two images have been processed
    {
      /* MATCH KEYPOINT DESCRIPTORS */
      matchDescriptors((dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints,
                       (dataBuffer.end() - 2)->descriptors, (dataBuffer.end() - 1)->descriptors,
                       matches, descriptorName, descriptorType, matcherType, selectorType);

      noMatches.push_back(matches.size());
      // store matches in current data frame
      (dataBuffer.end() - 1)->kptMatches = matches;

      // std::cout << matches.size() << " matches found." << '\n';

      bVis = false;
      if (bVis) { // Visulise keypoint mathces
          cv::Mat matchImg = ((dataBuffer.end() - 1)->cameraImg).clone();
          cv::drawMatches((dataBuffer.end() - 2)->cameraImg, (dataBuffer.end() - 2)->keypoints,
                          (dataBuffer.end() - 1)->cameraImg, (dataBuffer.end() - 1)->keypoints,
                          matches, matchImg,
                          cv::Scalar::all(-1), cv::Scalar::all(-1),
                          vector<char>(), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

          string windowName = "Matching keypoints between two camera images";
          cv::namedWindow(windowName, 7);
          cv::imshow(windowName, matchImg);
          cout << "Press key to continue to next image" << endl;
          cv::waitKey(0); // wait for key to be pressed
      }
      bVis = false;

      cout << "#7 : MATCH KEYPOINT DESCRIPTORS done" << endl;
      /* TRACK 3D OBJECT BOUNDING BOXES */

// ################################################ END: VISUAL CAMERA ANALYSIS ################################################ //

      //// STUDENT ASSIGNMENT
      //// TASK FP.1 -> match list of 3D objects (vector<BoundingBox>) between current and previous frame (implement ->matchBoundingBoxes)
      map<int, int> bbBestMatches;
      matchBoundingBoxes(matches, bbBestMatches, *(dataBuffer.end()-2), *(dataBuffer.end()-1)); // associate bounding boxes between current and previous frame using keypoint matches
      //// EOF STUDENT ASSIGNMENT

      // store matches in current data frame
      (dataBuffer.end()-1)->bbMatches = bbBestMatches;

      cout << "#8 : TRACK 3D OBJECT BOUNDING BOXES done" << endl;


      /* COMPUTE TTC ON OBJECT IN FRONT */

      // loop over all BB match pairs
      for (auto it1 = (dataBuffer.end() - 1)->bbMatches.begin(); it1 != (dataBuffer.end() - 1)->bbMatches.end(); ++it1) {
        // find bounding boxes associates with current match
        BoundingBox *prevBB, *currBB;
        for (auto it2 = (dataBuffer.end() - 1)->boundingBoxes.begin(); it2 != (dataBuffer.end() - 1)->boundingBoxes.end(); ++it2) {
          if (it1->second == it2->boxID) { // check wether current match partner corresponds to this BB
            currBB = &(*it2);
          }
        }
        for (auto it2 = (dataBuffer.end() - 2)->boundingBoxes.begin(); it2 != (dataBuffer.end() - 2)->boundingBoxes.end(); ++it2) {
          if (it1->first == it2->boxID) { // check wether current match partner corresponds to this BB
            prevBB = &(*it2);
          }
        }

// ################################################ TTC CALCUALTIONS ################################################ //
        // std::cout << "number of lidar points:" << currBB->lidarPoints.size() << " and " << prevBB->lidarPoints.size() << '\n';

        if( currBB->lidarPoints.size()>0 && prevBB->lidarPoints.size()>0 ) { // only compute TTC if we have Lidar points
          //// STUDENT ASSIGNMENT
          //// TASK FP.2 -> compute time-to-collision based on Lidar data (implement -> computeTTCLidar)
          double ttcLidar;
          computeTTCLidar(prevBB->lidarPoints, currBB->lidarPoints, sensorFrameRate, ttcLidar, LidarTTC_Array, TTCLidar_delta_d, TTCLidar_dist, TTCLidar_vel, image);
          //// EOF STUDENT ASSIGNMENT

          bVis = false;
          if(bVis) {
            show3DObjects((dataBuffer.end()-1)->boundingBoxes, cv::Size(4.0, 20.0), cv::Size(2000, 2000), true);
          }
          bVis = false;

          //// STUDENT ASSIGNMENT
          //// TASK FP.3 -> assign enclosed keypoint matches to bounding box (implement -> clusterKptMatchesWithROI)
          //// TASK FP.4 -> compute time-to-collision based on camera (implement -> computeTTCCamera)
          double ttcCamera;
          clusterKptMatchesWithROI(*currBB, (dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints, (dataBuffer.end() - 1)->kptMatches);
          computeTTCCamera((dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints, currBB->kptMatches, sensorFrameRate, ttcCamera, CamTTC_Array, image);
          //// EOF STUDENT ASSIGNMENT

          bVis = false;
          if (bVis) {
            cv::Mat visImg = (dataBuffer.end() - 1)->cameraImg.clone();
            showLidarImgOverlay(visImg, currBB->lidarPoints, P_rect_00, R_rect_00, RT, &visImg);
            cv::rectangle(visImg, cv::Point(currBB->roi.x, currBB->roi.y), cv::Point(currBB->roi.x + currBB->roi.width, currBB->roi.y + currBB->roi.height), cv::Scalar(0, 255, 0), 2);

            char str[200];
            sprintf(str, "TTC Lidar : %f s, TTC Camera : %f s", ttcLidar, ttcCamera);
            putText(visImg, str, cv::Point2f(80, 50), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(0,0,255));

            string windowName = "Final Results : TTC";
            cv::namedWindow(windowName, 4);
            cv::imshow(windowName, visImg);
            cout << "Press key to continue to next frame" << endl;
            cv::waitKey(0);
          }
          bVis = false;

        } // eof TTC computation
      } // eof loop over all BB matches
    }
    ++image;
  }

  std::cout << '\n'<< '\n' << "TTC Lidar Evaluation" << '\n';
  image = 1;
  std::vector<double>::const_iterator it1 = LidarTTC_Array.begin();
  std::vector<double>::const_iterator it2 = TTCLidar_delta_d.begin();
  std::vector<double>::const_iterator it3 = TTCLidar_vel.begin();
  std::vector<double>::const_iterator it4 = TTCLidar_dist.begin();
  for (; it1 != LidarTTC_Array.end(); ++it1, ++it2, ++it3, ++it4) {
    std::cout << image << "      TTC: " << setprecision(4) << *it1 << "       delta d: " << setprecision(2) << *it2 << "      min dist: " << setprecision(4) << *it4 << "      vel: " << setprecision(2) << *it3 << '\n';
    ++image;
  }

  std::cout << '\n'<< '\n' << "TTC Camera Evaluation" << '\n';
  image = 1;
  for (std::vector<double>::const_iterator i = CamTTC_Array.begin(); i != CamTTC_Array.end(); ++i) {
    std::cout << image << ": " << setprecision(4) << *i << '\n';
    ++image;
  }

    // Evaluation of system
  // const string *path="/Users/arjun/Documents/GitHub/Udacity_Sensor_Fusion/SFND_3D_Object_Tracking/dat/camera_data/" + detectorName + "_" + descriptorName + ".txt";
  ofstream file("/Users/arjun/Documents/GitHub/Udacity_Sensor_Fusion/SFND_3D_Object_Tracking/dat/camera_data/" + detectorName + "_" + descriptorName + ".txt"); //open in constructor
  if (file.is_open()) {
    file << "     TTC Camera Evaluation" << '\n';
    image = 1;
    for (std::vector<double>::const_iterator i = CamTTC_Array.begin(); i != CamTTC_Array.end(); ++i) {
      file << setprecision(4) << *i << '\n';
      ++image;
    }

    file << '\n'<< '\n' << "     TTC Lidar Evaluation" << '\n';
    std::vector<double>::const_iterator it1 = LidarTTC_Array.begin();
    std::vector<double>::const_iterator it2 = TTCLidar_delta_d.begin();
    std::vector<double>::const_iterator it3 = TTCLidar_vel.begin();
    std::vector<double>::const_iterator it4 = TTCLidar_dist.begin();
    image = 1;
    file << " TTC: " << '\n';
    for (; it1 != LidarTTC_Array.end(); ++it1) {
      file << setprecision(4) << *it1 << '\n';
      ++image;
    }
    image = 1;
    file << " delta d: " << '\n';
    for (; it2 != TTCLidar_delta_d.end(); ++it2) {
      file << setprecision(2) << *it2 << '\n';
      ++image;
    }
    image = 1;
    file << " vel: " << '\n';
    for (; it3 != TTCLidar_vel.end(); ++it3) {
      file << setprecision(2) << *it3 << '\n';
      ++image;
    }
    image = 1;
    file << " min dist: " << '\n';
    for (; it4 != TTCLidar_dist.end(); ++it4) {
      file << setprecision(2) << *it4 << '\n';
      ++image;
    }
  }
  // eof loop over all images
  return 0;
}
