#include <unordered_set>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;


// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT) {
  // loop over all Lidar points and associate them to a 2D bounding box
  cv::Mat X(4, 1, cv::DataType<double>::type);
  cv::Mat Y(3, 1, cv::DataType<double>::type);

  for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1) {
    // assemble vector for matrix-vector-multiplication
    X.at<double>(0, 0) = it1->x;
    X.at<double>(1, 0) = it1->y;
    X.at<double>(2, 0) = it1->z;
    X.at<double>(3, 0) = 1;

    // project Lidar point into camera
    Y = P_rect_xx * R_rect_xx * RT * X;
    cv::Point pt;
    // pixel coordinates
    pt.x = Y.at<double>(0, 0) / Y.at<double>(2, 0);
    pt.y = Y.at<double>(1, 0) / Y.at<double>(2, 0);

    vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
    for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2) {
      // shrink current bounding box slightly to avoid having too many outlier points around the edges
      cv::Rect smallerBox;
      smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
      smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
      smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
      smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

        // check wether point is within current bounding box
        if (smallerBox.contains(pt)) {
          enclosingBoxes.push_back(it2);
        }

      } // eof loop over all bounding boxes

      // check wether point has been enclosed by one or by multiple boxes
      if (enclosingBoxes.size() == 1) {
        // add Lidar point to bounding box
        enclosingBoxes[0]->lidarPoints.push_back(*it1);
      }

  } // eof loop over all Lidar points
}

/*
* The show3DObjects() function below can handle different output image sizes, but the text output has been manually tuned to fit the 2000x2000 size.
* However, you can make this function work for other sizes too.
* For instance, to use a 1000x1000 size, adjusting the text positions by dividing them by 2.
*/
void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait) {
  // create topview image
  cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

  for(auto it1=boundingBoxes.begin(); it1!=boundingBoxes.end(); ++it1)
  {
    // create randomized color for current 3D object
    cv::RNG rng(it1->boxID);
    cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150), rng.uniform(0, 150));

    // plot Lidar points into top view image
    int top=1e8, left=1e8, bottom=0.0, right=0.0;
    float xwmin=1e8, ywmin=1e8, ywmax=-1e8;
    for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
    {
      // world coordinates
      float xw = (*it2).x; // world position in m with x facing forward from sensor
      float yw = (*it2).y; // world position in m with y facing left from sensor
      xwmin = xwmin<xw ? xwmin : xw;
      ywmin = ywmin<yw ? ywmin : yw;
      ywmax = ywmax>yw ? ywmax : yw;

      // top-view coordinates
      int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
      int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

      // find enclosing rectangle
      top = top<y ? top : y;
      left = left<x ? left : x;
      bottom = bottom>y ? bottom : y;
      right = right>x ? right : x;

      // draw individual point
      cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
    }

    // draw enclosing rectangle
    cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),cv::Scalar(0,0,0), 2);

    // augment object with some key data
    char str1[200], str2[200];
    sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
    putText(topviewImg, str1, cv::Point2f(left-250, bottom+50), cv::FONT_ITALIC, 2, currColor);
    sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax-ywmin);
    putText(topviewImg, str2, cv::Point2f(left-250, bottom+125), cv::FONT_ITALIC, 2, currColor);
  }

  // plot distance markers
  float lineSpacing = 2.0; // gap between distance markers
  int nMarkers = floor(worldSize.height / lineSpacing);
  for (size_t i = 0; i < nMarkers; ++i)
  {
    int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
    cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
  }

  // display image
  string windowName = "3D Objects";
  cv::namedWindow(windowName, 1);
  cv::imshow(windowName, topviewImg);

  if(bWait)
  {
    cv::waitKey(0); // wait for key to be pressed
  }
}

// associates given bounding boxes and the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches) {
  double eucli_diff_mean = 0.0;
  int N = 0;
  for (const cv::DMatch &match : kptMatches) {
    // query => source (prev), train => reference (curr)
    const cv::KeyPoint &kpts_prevFrame = kptsPrev[match.queryIdx];
    const cv::KeyPoint &kpts_currFrame = kptsCurr[match.trainIdx];
    if (boundingBox.roi.contains(kpts_currFrame.pt)) {
      N++;
      double x_diff = kpts_currFrame.pt.x - kpts_prevFrame.pt.x;
      double y_diff = kpts_currFrame.pt.y - kpts_prevFrame.pt.y;
      double eucli_diff = std::hypot(x_diff, y_diff);
      eucli_diff_mean += eucli_diff;
    }
  }
  eucli_diff_mean /= N;

  double eucli_diff_std = 0.0;
  for (const cv::DMatch &match : kptMatches) {
    const cv::KeyPoint &kpts_prevFrame = kptsPrev[match.queryIdx];
    const cv::KeyPoint &kpts_currFrame = kptsCurr[match.trainIdx];
    if (boundingBox.roi.contains(kpts_currFrame.pt)) {
      double x_diff = kpts_currFrame.pt.x - kpts_prevFrame.pt.x;
      double y_diff = kpts_currFrame.pt.y - kpts_prevFrame.pt.y;
      double eucli_diff = std::hypot(x_diff, y_diff);
      eucli_diff_std += (eucli_diff - eucli_diff_mean) * (eucli_diff - eucli_diff_mean);
    }
  }
  eucli_diff_std = std::sqrt( eucli_diff_std / (N - 1) );

  for (const cv::DMatch &match : kptMatches) {
    const cv::KeyPoint &kpts_prevFrame = kptsPrev[match.queryIdx];
    const cv::KeyPoint &kpts_currFrame = kptsCurr[match.trainIdx];
    if (boundingBox.roi.contains(kpts_currFrame.pt)) {
      double x_diff = kpts_currFrame.pt.x - kpts_prevFrame.pt.x;
      double y_diff = kpts_currFrame.pt.y - kpts_prevFrame.pt.y;
      double eucli_diff = std::hypot(x_diff, y_diff);
      if (std::abs(eucli_diff - eucli_diff_mean) < 2.0 * eucli_diff_std) {
        boundingBox.kptMatches.push_back(match);
      }
    }
  }
}

// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, std::vector<double> &TTC_Array, int image, cv::Mat *visImg) {
  int N = 0;
  double sum_kpt_ratios;
  vector<double> kpt_ratios;
  for(const cv::DMatch &ground_kpt : kptMatches) {
    const cv::KeyPoint ground_kpt_prev = kptsPrev[ground_kpt.queryIdx];
    const cv::KeyPoint ground_kpt_curr = kptsCurr[ground_kpt.trainIdx];
    for (auto end_kpt = kptMatches.begin() + 1; end_kpt != kptMatches.end(); ++end_kpt) {

      const cv::KeyPoint end_kpt_prev = kptsPrev[end_kpt->queryIdx];
      const cv::KeyPoint end_kpt_curr = kptsCurr[end_kpt->trainIdx];

      double diff_kpt_prev_x = end_kpt_prev.pt.x - ground_kpt_prev.pt.x;
      double diff_kpt_prev_y = end_kpt_prev.pt.y - ground_kpt_prev.pt.y;
      double diff_kpt_curr_x = end_kpt_curr.pt.x - ground_kpt_curr.pt.x;
      double diff_kpt_curr_y = end_kpt_curr.pt.y  - ground_kpt_curr.pt.y;
      double diff_kpt_prev = std::hypot(diff_kpt_prev_x, diff_kpt_prev_y);
      double diff_kpt_curr = std::hypot(diff_kpt_curr_x, diff_kpt_curr_y);

      double diff_kpt_ratio = diff_kpt_curr/diff_kpt_prev;

      // std::cout << "end_kpt_prev.pt.x:     " << end_kpt_prev.pt.x << "     " << "ground_kpt_prev.pt.x:     " <<ground_kpt_prev.pt.x << "\n";
      // std::cout << "end_kpt_prev.pt.y:     " << end_kpt_prev.pt.y << "     " << "ground_kpt_prev.pt.y:     " <<ground_kpt_prev.pt.y << "\n";
      // std::cout << "end_kpt_curr.pt.x:     " << end_kpt_curr.pt.x << "     " << "ground_kpt_curr.pt.x:     " <<ground_kpt_curr.pt.x << "\n";
      // std::cout << "end_kpt_curr.pt.y:     " << end_kpt_curr.pt.y << "     " << "ground_kpt_curr.pt.y:     " <<ground_kpt_curr.pt.y << "\n" << "\n";
      //
      // std::cout << "raw data (prev): " << diff_kpt_prev_x << "    " << diff_kpt_prev_y << "    " << "Hypot: " << diff_kpt_prev << "\n";
      // std::cout << "raw data (curr): " << diff_kpt_curr_x << "    " << diff_kpt_curr_y << "    " << "Hypot: " << diff_kpt_curr << "\n";
      // std::cout << "diff_kpt_ratio: " <<diff_kpt_ratio << '\n' <<"\n";

      if (isnan(diff_kpt_ratio) || isinf(diff_kpt_ratio)) {
        // std::cout << "ERROR" << '\n';
        // std::cout << "raw data (prev):" << diff_kpt_prev_x << "    " << diff_kpt_prev_y << "    " << diff_kpt_prev << "\n";
        // std::cout << "raw data (curr):" << diff_kpt_curr_x << "    " << diff_kpt_curr_y << "    " << diff_kpt_curr << "\n";
        // std::cout << "diff_kpt_ratio: " <<diff_kpt_ratio << '\n';
        // cv::waitKey(0);
        continue;
      }
      kpt_ratios.push_back(diff_kpt_ratio);
      sum_kpt_ratios += diff_kpt_ratio;
      N++;
    }
  }

  double mean_sum_kpt_ratio = sum_kpt_ratios/ N;
  // std::cout << mean_sum_kpt_ratio << "   " << sum_kpt_ratios << "   "<< N << '\n';
  TTC = (-1.0/frameRate)/(1.0-mean_sum_kpt_ratio);
  TTC_Array.push_back(TTC);
  std::cout << "      Camera TTC: " << TTC << '\n';
}

void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev, std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC, std::vector<double> &TTC_Array, std::vector<double> &TTCLidar_delta_d, std::vector<double> &TTCLidar_dist, std::vector<double> &TTCLidar_velocity, int image) {
  double dT = 1/frameRate;
  double laneWidth = 2.0;
  double min_d0 = 1e9, min_d1 = 1e9;
  std::vector<LidarPoint> plane_1 = PlaneRansac(lidarPointsCurr, 300, 0.03, image);

  if (TTCLidar_dist.size() == 0) {
    std::vector<LidarPoint> plane_0 = PlaneRansac(lidarPointsPrev, 300, 0.03, image);
    for (auto it = plane_0.begin(); it != plane_0.end(); ++it) {
      if (abs(it->y) <= laneWidth / 2.0) {
        min_d0 = min_d0 > it->x ? it->x : min_d0; // replace if true
      }
    }
  }
  else {
      min_d0 = TTCLidar_dist.back();
  }
  for (auto it = plane_1.begin(); it != plane_1.end(); ++it) {
    if (abs(it->y) <= laneWidth / 2.0) {
      min_d1 = min_d1 > it->x ? it->x : min_d1;
    }
  }
  TTC = min_d1 * (dT / (min_d0-min_d1));
  double velocity = (min_d0-min_d1)/dT;
  TTC_Array.push_back(TTC);
  TTCLidar_dist.push_back(min_d1);
  TTCLidar_delta_d.push_back(min_d0-min_d1);
  TTCLidar_velocity.push_back(velocity);

  // Evaluation
  // std::cerr << "No. Plane 0: " << plane_0.size() << '\n';
  std::cerr << "No. Plane 1: " << plane_1.size() << '\n';
  std::cout << "      min distance prev: " << min_d0 << '\n';
  std::cout << "      min distance curr: " << min_d1 << '\n';
  std::cout << "      delta dist: " << (min_d0-min_d1) << '\n';
  std::cout << "      velocity: " << (min_d0-min_d1) / dT << '\n';
  std::cout << "      Lidar TTC: " << TTC << '\n';
}

// Matches bounding boxes to their corresponding boxes. Sets pairs
void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame) {

  // Create matrix to track the number of matches between a boxes found in prev and curr frame
  // Note: cv::Size(cols, rows)
  // Matching bounding boxes using matches from cv system.
  cv::Mat keypnt_box_match_matrix = cv::Mat::zeros(cv::Size(currFrame.boundingBoxes.size(), prevFrame.boundingBoxes.size()), CV_32FC1);

  // for (const cv::DMatch &it_match : matches) {
  for(auto it_match = matches.begin(); it_match != matches.end(); ++it_match) {
    // trying different ways of looping for practise
    for (const BoundingBox &it_Bbox_prev : prevFrame.boundingBoxes) {
      int it_Bbox_prev_id = it_Bbox_prev.boxID;
      // const cv::Point2f &keypnt_prev = prevFrame.keypoints[it_match->queryIdx].pt; // Query is prev | Train is curr
      // checking for keypnt in prev before looping curr cuts time

      if (it_Bbox_prev.roi.contains(prevFrame.keypoints[it_match->queryIdx].pt)) {
      // if (it_Bbox_prev.roi.contains(cv::Point2i((int) keypnt_prev.x, (int) keypnt_prev.y))) {
        for (const BoundingBox &bbox_curr_frame : currFrame.boundingBoxes) {
          int bbox_curr_frame_id = bbox_curr_frame.boxID;
          // use y coord of keypnt in the prev frame as the y coord rather than the y coord in the curr frame to improve quality of keypnt??
          // if (bbox_curr_frame.roi.contains(cv::Point2i((int) keypnt_curr.x, (int) keypnt_prev.y))) {
          if (bbox_curr_frame.roi.contains(currFrame.keypoints[it_match->trainIdx].pt)) {
            keypnt_box_match_matrix.at<float>(it_Bbox_prev_id, bbox_curr_frame_id) += 1.0f;
          }
        }
      }
    }
  }

  auto matrix_print = [](const cv::Mat &mat) { //lambda fx for printing matrix | 'const' keyword to ensure
      for (int row = 0; row < mat.rows; row++) {
          for (int col = 0; col < mat.cols; col++) {
              std::cout << mat.at<float>(row, col) << ", ";
          }
          std::cout << "\n";
      }
  };
  std::cout << "Final match-keypoint matrix:\n";
  matrix_print(keypnt_box_match_matrix);

  auto row_col_zero = [](cv::Mat &mat, int row, int col) {
      for (int col = 0; col < mat.cols; col++) { //zero row
          mat.at<float>(row, col) = 0.0f;
      }
      for (int row = 0; row < mat.rows; row++) { // zero col
          mat.at<float>(row, col) = 0.0f;
      }
  };

  while (true) {
    double minVal;
    double maxVal;
    cv::Point minLoc;
    cv::Point maxLoc;
    cv::minMaxLoc(keypnt_box_match_matrix, &minVal, &maxVal, &minLoc, &maxLoc);
    if ((maxVal) < 1.0f) { // all values zeroed as max is zero
      break;
    }
    int it_Bbox_prev_id = maxLoc.y;
    // std::cout << "Best match found" << "\n Previous frame box ID " << maxLoc.y
      // << "\n Current frame box ID " << maxLoc.x << "\n No. keypoint matches: " << maxVal <<"\n";

    // (prev, curr) in matches map
    bbBestMatches.insert(std::pair<int, int>(maxLoc.y, maxLoc.x));
    row_col_zero(keypnt_box_match_matrix, maxLoc.y, maxLoc.x);
    // std::cout << " zeroed keypoint matix:\n";
    // matrix_print(keypnt_box_match_matrix);
  }
}

std::vector<LidarPoint> PlaneRansac(std::vector<LidarPoint> lidarPoints, int maxIterations, float distanceTol, int image) {
  bool save = false;
  std::vector<LidarPoint> inliersResult;
	srand(time(NULL));

	while (maxIterations--) {
		// Randomly sample subset and fit line
		std::vector<LidarPoint> inliers;
    std::unordered_set<int> samples;
		while (samples.size() < 3) {
			samples.insert(rand() % (lidarPoints.size()));
		}
		float x1, y1, z1, x2, y2, z2, x3, y3, z3;
		auto itr = samples.begin();
		x1 = lidarPoints[*itr].x;
		y1 = lidarPoints[*itr].y;
		z1 = lidarPoints[*itr].z;
		itr++;
		x2 = lidarPoints[*itr].x;
		y2 = lidarPoints[*itr].y;
		z2 = lidarPoints[*itr].z;
		itr++;
		x3 = lidarPoints[*itr].x;
		y3 = lidarPoints[*itr].y;
		z3 = lidarPoints[*itr].z;
		// Generate plane between these points
		float a = (y2 - y1)*(z3 - z1)-(y3 - y1)*(z2 - z1);
		float b = (z2 - z1)*(x3 - x1)-(z3 - z1)*(x2 - x1);
		float c = (x2 - x1)*(y3 - y1)-(x3 - x1)*(y2 - y1);
		float d = -(a*x1 + b*y1 + c*z1);

		for (int index = 0; index < lidarPoints.size(); index++) {
			float x4 = lidarPoints[index].x;
			float y4 = lidarPoints[index].y;
			float z4 = lidarPoints[index].z;
			// Measure distance between every point and fitted line
			float dist = (fabs(a*x4 + b*y4 + c*z4 + d))/(sqrt(a*a + b*b + c*c)); //use fabs() instead of abs() so as to account for the floats
			// If distance is smaller than threshold count it as inlier
			if (dist <= distanceTol) {
				inliers.push_back(lidarPoints[index]);
			}
		}
    // Return indicies of inliers from fitted line with most inliers
    if (inliers.size() > inliersResult.size()) {
      inliersResult = inliers;
    }
    inliers.clear();
	}

  // ransac visualize
  bool bVis = false;
  if (bVis) {
    cv::Size worldSize = cv::Size(4.0, 20.0);
    cv::Size imageSize = cv::Size(2000, 2000);
    bool bWait = true;
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));
    // create randomized color for current 3D object
    // cv::RNG rng(it1->boxID);
    // cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150), rng.uniform(0, 150));

    // plot Lidar points into top view image
    int top=1e8, left=1e8, bottom=0.0, right=0.0;
    float xwmin=1e8, ywmin=1e8, ywmax=-1e8;
    for (auto it2 = inliersResult.begin(); it2 != inliersResult.end(); ++it2) {
      // world coordinates
      float xw = (*it2).x; // world position in m with x facing forward from sensor
      float yw = (*it2).y; // world position in m with y facing left from sensor
      xwmin = xwmin<xw ? xwmin : xw;
      ywmin = ywmin<yw ? ywmin : yw;
      ywmax = ywmax>yw ? ywmax : yw;

      // top-view coordinates
      int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
      int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

      // find enclosing rectangle
      top = top<y ? top : y;
      left = left<x ? left : x;
      bottom = bottom>y ? bottom : y;
      right = right>x ? right : x;

      // draw individual point
      cv::circle(topviewImg, cv::Point(x, y), 4, 5, -1);
    }

    // draw enclosing rectangle
    cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),cv::Scalar(0,0,0), 2);

    // augment object with some key data
    char str1[200], str2[200];
    sprintf(str1, "#pts=%d", (int)inliersResult.size());
    putText(topviewImg, str1, cv::Point2f(left-250, bottom+50), cv::FONT_ITALIC, 2, 5);
    sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax-ywmin);
    putText(topviewImg, str2, cv::Point2f(left-250, bottom+125), cv::FONT_ITALIC, 2, 5);

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects - RANSAC";
    cv::namedWindow(windowName, 1);
    cv::imshow(windowName, topviewImg);
    if (save) {
      string source = "/Users/arjun/Documents/GitHub/Udacity_Sensor_Fusion/SFND_3D_Object_Tracking/images/top_view/RANSAC_topview_" + std::to_string(image-1) + ".jpg";
      cv::imwrite(source, topviewImg);
    }

    if(bWait) {
      cv::waitKey(0); // wait for key to be pressed
    }
  }
  bVis = false;
  return inliersResult;
}
