
#include <iostream>
#include <algorithm>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;


// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        pt.x = Y.at<double>(0, 0) / Y.at<double>(0, 2); // pixel coordinates
        pt.y = Y.at<double>(1, 0) / Y.at<double>(0, 2);

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        { 
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    } // eof loop over all Lidar points
}


void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
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


// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    std::vector <double> ec;
    double maxdistance=2.0;
    //std::cout<<"clusterKptMatchesWithROI beign "<<std::endl;
    for (auto match :kptMatches)
    {
        cv::KeyPoint preKepoint=kptsCurr.at(match.trainIdx);
        cv::KeyPoint CurrKepoint=kptsPrev.at(match.queryIdx);
        
        if (boundingBox.roi.contains(CurrKepoint.pt))
        {
          double ecdistance=cv::norm(preKepoint.pt-CurrKepoint.pt);
          ec.push_back(ecdistance);
        }
    }
    double ecmid;
    int index=floor(ec.size()/2);
    std::sort(ec.begin(),ec.end());
     // std::cout<<"clusterKptMatchesWithROI mid "<<std::endl;
    if (ec.size()%2==0)
    {
        
        ecmid=ec[index];
    }
    else
    {
       ecmid=(ec[index]+ec[index-1])/2;
    }
       for (auto match :kptMatches)
    {
        cv::KeyPoint preKepoint=kptsCurr.at(match.trainIdx);
        cv::KeyPoint CurrKepoint=kptsPrev.at(match.queryIdx);
        if (boundingBox.roi.contains(CurrKepoint.pt))
        {
          double ecdistance=cv::norm(preKepoint.pt-CurrKepoint.pt);
          if ((ecdistance>(ecdistance-maxdistance))&&(ecdistance<(ecdistance+maxdistance)))
          {
              boundingBox.keypoints.push_back(CurrKepoint);
              boundingBox.kptMatches.push_back(match);
          }
        }
    }
    //std::cout<<"clusterKptMatchesWithROI end "<<boundingBox.kptMatches.size()<<std::endl;
    // ...
}


// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
     vector<double> r;
     double min_distance=50.0;
     double max_distance=150.0;
//std::cout<<"computeTTCCamera beign "<<kptMatches.size()<<std::endl;
    for (auto match=kptMatches.begin();match!=kptMatches.end()-1;match++)
    {
 
        cv::KeyPoint OuterCurrKepoint=kptsCurr.at(match->trainIdx);
        cv::KeyPoint OuterprevKepoint=kptsPrev.at(match->queryIdx);
        //std::cout<<"outer current dev "<<std::endl;
        for (auto match1:kptMatches)
        {
               cv::KeyPoint InnerCurrKepoint=kptsCurr.at(match1.trainIdx);
               cv::KeyPoint InnerprevKepoint=kptsPrev.at(match1.queryIdx);
               double currdev=cv::norm(OuterCurrKepoint.pt-InnerCurrKepoint.pt);
               double prevdev=cv::norm(OuterprevKepoint.pt-InnerprevKepoint.pt);
               //std::cout<<"current dev "<<currdev<<std::endl;
               if ((currdev>=min_distance)&&(currdev<=max_distance)&&(prevdev> std::numeric_limits<double>::epsilon()))
               {
                   double ratio=currdev  / prevdev;
                   r.push_back(ratio);
               }
        }

    }
    if (r.size()==0)
    {
        std::cout<<"computeTTCCamera free "<<std::endl;
        TTC=NAN;
        return;
    }
    double ecmid;
   
    std::sort(r.begin(),r.end());
     int index=floor(r.size()/2);
    if (r.size()%2==0)
    {
        
        ecmid=r[index];
    }
    else
    {
       ecmid=(r[index]+r[index-1])/2;
    }
    double f=1.0/frameRate ;
    TTC=-f/(1.0-ecmid);
    std::cout<<"Camera TCC : "<<  TTC << "msec"<<std::endl;
    // ...
} 


void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
    double lanewidth=0.4;
    double minprev=1e9;
    double mincurr=1e9;
    for(auto it=lidarPointsPrev.begin();it!=lidarPointsPrev.end();it++)
    {
        double y=2*std::abs(it->y);
        if (y < lanewidth)
        {
           minprev = minprev > it->x ? it->x : minprev; 
        }
    }
    for (auto it1=lidarPointsCurr.begin();it1!=lidarPointsCurr.end();it1++)
    {
        double y1=2*std::abs(it1->y);
        if (y1 < lanewidth)
        {
           mincurr = mincurr > it1->x ? it1->x : mincurr; 
        }

    }

     TTC= mincurr / (frameRate*(minprev-mincurr));
     std::cout<<"Lidar TCC : "<<  TTC << "msec"<<std::endl;
    // ...
}

bool sortmatch(const pair<int,int> a ,const pair<int,int> b)
{
    return (a.second>b.second);;
}
void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{
    cv::KeyPoint current_keyPoint ,prev_keyPoint;
   

        for (auto it1 = prevFrame.boundingBoxes.begin(); it1 != prevFrame.boundingBoxes.end(); ++it1)
        {
           std::vector <std::pair<int ,int >> prevMatch;  
          for (auto it3 = currFrame.boundingBoxes.begin(); it3 != currFrame.boundingBoxes.end(); ++it3)  
          {
            int count=0;
            for (auto it=matches.begin();it!=matches.end();it++)
            {
                current_keyPoint=currFrame.keypoints[it->trainIdx];
                prev_keyPoint =prevFrame.keypoints[it->queryIdx];
                if ((it1->roi.contains(prev_keyPoint.pt))&& (it3->roi.contains(current_keyPoint.pt)))
                {
                    count++;
                }
            }
            prevMatch.push_back(std::make_pair(it3->boxID,count));
             //std::cout<<"Prev match : "<<  it3->boxID << " count: " <<count <<std::endl;
          } 
          sort(prevMatch.begin(),prevMatch.end(),sortmatch);
          bbBestMatches.insert(std::make_pair(it1->boxID,prevMatch[0].first));   
          //std::cout<<"Best match :"<<  it1->boxID << " " <<prevMatch[0].first <<std::endl; 
        }
    
    // ...
}
