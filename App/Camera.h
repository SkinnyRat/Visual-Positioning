//
//  Camera.h
//  CV
//
//  Created by ThisUser on 22/6/19.
//  Copyright © 2019 ThisUser. All rights reserved.
//

#ifndef Camera_h
#define Camera_h

#import <Foundation/Foundation.h>
#import <UIKit/UIKit.h>


// Protocol for callback action
@protocol CameraDelegate <NSObject>

@end

// Public interface for camera. ViewController only needs to init, start and stop.
@interface Camera : NSObject

-(id) initWithController: (UIViewController<CameraDelegate>*)c andImageView: (UIImageView*)iv titleText: (Float32*)mainTitle bPressed: (int*)btnPressed;
-(void)start;
-(void)stop;

@end

#endif /* Camera_h */

