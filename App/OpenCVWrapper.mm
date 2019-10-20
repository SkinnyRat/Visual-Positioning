//
//  OpenCVWrapper.m
//  CV
//
//  Created by ThisUser on 8/6/19.
//  Copyright © 2019 ThisUser. All rights reserved.
//
#import <opencv2/opencv.hpp>

#import "OpenCVWrapper.h"

@implementation OpenCVWrapper

+ (NSString *)openCVVersionString
{
    return [NSString stringWithFormat:@"OpenCV => v%s",  CV_VERSION];
}

@end
