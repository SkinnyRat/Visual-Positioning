//
//  Camera.m
//  CV Navigation
//

#import <opencv2/opencv.hpp>
#import <opencv2/videoio/cap_ios.h>
#import <opencv2/imgcodecs/ios.h>

#import <Foundation/Foundation.h>

#include "Camera.h"

template<typename Out> void Split(const std::string & s, char delim, Out result);
std::vector<std::string>    Split(const std::string & s, char delim);

/*
 Done:
 - Akaze detection.
 - Reference image.
 - Depth map.
 - Save images.
 - Calibration.
 - Pose computation.
 
 Next:
 - Cloud Service!!!
 - Real-life Trials!!!
 */


@interface Camera()<CvVideoCameraDelegate>
@end

@implementation Camera
{
    UIViewController<CameraDelegate> * delegate;
    UIImageView   * imageView;
    int           * bPressed;
    Float32       * mainTitle;
    CvVideoCamera * videoCamera;
    
    int BCount;
    cv::Mat Ref, ref_desc;
    cv::Ptr<cv::AKAZE> akaze;
    cv::Ptr<cv::DescriptorMatcher> matcher;
    std::vector<cv::KeyPoint> ref_kp;
    std::vector<std::vector<std::string>> Coords;
}

- (id)initWithController:(UIViewController<CameraDelegate>*)c andImageView:(UIImageView*)iv titleText:(Float32*)theTitle bPressed: (int*)btnPressed
{
    delegate  = c;
    imageView = iv;
    mainTitle = theTitle;
    bPressed  = btnPressed;
    
    videoCamera = [[CvVideoCamera alloc] initWithParentView:imageView];
    videoCamera.defaultAVCaptureDevicePosition      = AVCaptureDevicePositionBack;
    videoCamera.defaultAVCaptureVideoOrientation    = AVCaptureVideoOrientationPortrait;
    videoCamera.rotateVideo                         = NO;
    videoCamera.defaultFPS                          = 8;
    videoCamera.delegate                            = self;
    
    //UIImage * tplImg  = [UIImage imageNamed:@"Reference"];
    NSData  * imageData = [[NSData alloc] initWithContentsOfURL: [NSURL URLWithString: @"http://laoahpek.azurewebsites.net/ReferenceFrame.jpg"]];
    UIImage * yucks = [UIImage imageWithData: imageData];
    NSLog(@"Downloaded JPG  = %f %f", yucks.size.height, yucks.size.width);
    cv::Mat temp;
    UIImageToMat(yucks, temp);
    cv::cvtColor(temp, Ref, cv::COLOR_BGRA2RGB);
    
    const double akaze_thresh = 2e-3;
    akaze   = cv::AKAZE::create(); akaze->setThreshold(akaze_thresh);
    matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
    akaze->detectAndCompute(Ref, cv::noArray(), ref_kp, ref_desc);
    
    //NSString * path   = [[NSBundle mainBundle] pathForResource:@"Coords" ofType:@"txt"];
    //NSString * coords = [NSString stringWithContentsOfFile:path encoding:NSUTF8StringEncoding error:nil];
    NSURL    * url    = [NSURL URLWithString:@"http://laoahpek.azurewebsites.net/DepthMapData.dat"];
    NSError  * error  = nil;
    NSString * coords = [[NSString alloc] initWithContentsOfURL: url encoding: NSUTF8StringEncoding error: &error];
    std::string rawcd = std::string([coords UTF8String]);
    NSLog(@"Downloaded Data = %ul", rawcd.length());
    
    std::stringstream ss(rawcd); std::string line;
    while (std::getline(ss, line))
    {
        std::stringstream sline(line);
        std::vector<std::string> points{std::istream_iterator<std::string>{sline}, std::istream_iterator<std::string>{}};
        Coords.push_back(points);
    }
    BCount = (*btnPressed);
    return self;
}

- (void)processImage:(cv::Mat &)img
{
    const double S          = 0.32f;    // Const scaling factor; 1.622f;
    const double nn_match   = 0.8f;     // Nearest-neighbour matching ratio
    float CamMat[3][3]      = { {660.3314718048218f, 0.0f, 240.0f}, {0.0f, 660.3314718048218f, 320.0f}, {0.0f, 0.0f, 1.0f } };
    
    cv::Mat Img, Cam = cv::Mat(3, 3, CV_32FC1, &CamMat);
    cv::cvtColor(img, Img, cv::COLOR_BGR2RGB);
    
    cv::Mat img_desc;
    std::vector<cv::KeyPoint> img_kp;
    std::vector<std::vector<cv::DMatch>> matches;
    akaze->detectAndCompute(Img, cv::noArray(), img_kp, img_desc);
    matcher->knnMatch(ref_desc, img_desc, matches, 2);
    
    std::map<std::string, int>  Lop,  Imp;
    std::vector<cv::Point3f>    locpoints;
    std::vector<cv::Point2f>    imgpoints;
    int MIN = 4, num = 0;
/*
    int RH  = Coords.size(),    RW = Coords[0].size();
    int CW  = img.cols,         CH = img.rows;
    NSLog(@"Text  = %d %d", RW, RH);
    NSLog(@"Image = %d %d", CW, CH);
*/
    for (unsigned i = 0; i < matches.size() && matches.size() >= MIN; i++)
    {
        if (matches[i][0].distance < nn_match * matches[i][1].distance && img_kp.size() >= MIN)
        {
            int qi = matches[i][0].queryIdx, ti = matches[i][0].trainIdx;
            int ux = (int)(ref_kp[qi].pt.x), vy = (int)(ref_kp[qi].pt.y);
            
            std::vector<std::string> xyz = Split(Coords[vy][ux], ',');
            cv::Point3f wpoint = {(float)std::atof(xyz[0].c_str()), (float)std::atof(xyz[1].c_str()), (float)std::atof(xyz[2].c_str())};
            if (wpoint.x == 0 && wpoint.y == 0 && wpoint.z == 0)    continue;
            
            std::string L = std::to_string((int)(wpoint.x*20))           + "_" + std::to_string((int)(wpoint.y*20));
            std::string I = std::to_string((int)(img_kp[ti].pt.x/20))    + "_" + std::to_string((int)(img_kp[ti].pt.y/20));
            
            if (!Lop.count(L) && !Imp.count(I))
            {
                num++;
                Lop[L] = 1;
                Imp[I] = 1;
                locpoints.push_back(wpoint);
                imgpoints.push_back(img_kp[ti].pt);
                cv::circle(img,  img_kp[ti].pt, 4,      cv::Scalar(0,0,255,255), 2);
                cv::putText(img, std::to_string(num),   img_kp[ti].pt, cv::FONT_HERSHEY_PLAIN, 1.5f, cv::Scalar(255,0,0,255));
                
                //std::string WP = std::to_string(wpoint.x) + ',' + std::to_string(wpoint.y) + ',' + std::to_string(wpoint.z);
                //cv::circle(Ref,  ref_kp[qi].pt, 4, cv::Scalar(0,255,0,255), 2);
                //cv::putText(Ref, std::to_string(num), ref_kp[qi].pt, cv::FONT_HERSHEY_PLAIN, 1.5f, cv::Scalar(255,0,0,255));
            }
        }
    }
    if (num < MIN)
    {
        *mainTitle = -99.0f; // Operating range -10m < distance < 10m.
        return;
    }
    cv::Mat Rvec, Tvec, R,  T, Inliers;
    cv::solvePnP(locpoints, imgpoints, Cam, cv::noArray(), Rvec, Tvec);
    cv::Rodrigues(Rvec, R); R = R.t(); T = -R * Tvec;
    float sx = T.at<float>(0,0)*S , sy = T.at<float>(1,0)*S , sz = T.at<float>(2,0)*S;
    
    if (BCount != *bPressed)
    {
        //cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
        //UIImage *toSave = MatToUIImage(img);
        //UIImageWriteToSavedPhotosAlbum(toSave, nil, nil, nil);
        //NSLog(@"Image saved %d time(s).", BCount);
        
        NSLog(@"Num = %d; Pose = %f %f %f", num, sx, sy, sz);
        BCount = *bPressed;
    }
    *mainTitle = std::sqrt(sx*sx + sy*sy + sz*sz);
}

- (void)start
{
    [videoCamera start];
}

- (void)stop
{
    [videoCamera stop];
}

@end


template<typename Out>
void Split(const std::string & s, char delim, Out result)
{
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim))
        if (item.size() && item[0] != delim)
            *(result++) = item;
}

std::vector<std::string> Split(const std::string & s, char delim)
{
    std::vector<std::string> elems;
    Split(s, delim, std::back_inserter(elems));
    return elems;
}




