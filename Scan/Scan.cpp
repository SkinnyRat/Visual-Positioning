// Reference: https://github.com/IntelRealSense/librealsense/wiki/API-How-To 

#include <chrono> 
#include <string> 
#include <fstream> 
#include <sstream> 
#include <iostream> 

#include <librealsense2/rs.hpp> 
#include <librealsense2/rsutil.h> 
#include <opencv2/opencv.hpp> 
#include <opencv2/imgproc/imgproc.hpp> 

#include "Header.hpp" 

int Fcount = 0; 


int main(int argc, char * argv[]) try
{
	rs2::context Ctx;
	auto Rlist = Ctx.query_devices();
	if (Rlist.size() == 0) throw std::runtime_error("No device detected! Is it plugged in?");
	std::cout << "CONNECTED => " << Rlist[0].get_info(RS2_CAMERA_INFO_NAME) << ".\n";

	rs2::config	cfg;
	cfg.enable_stream(RS2_STREAM_COLOR, W, H, RS2_FORMAT_RGB8, 30);
	cfg.enable_stream(RS2_STREAM_DEPTH, W, H, RS2_FORMAT_Z16,  30);

	rs2::colorizer			Cmap;
	rs2::pipeline			pipe;
	rs2::pipeline_profile	profile  = pipe.start(cfg);
	rs2_stream				align_to = AlignStream(profile.get_streams());
	rs2::align				align(align_to);
	auto Scale				= DepthScale(profile.get_device());
	auto stream				= profile.get_stream(RS2_STREAM_DEPTH).as<rs2::video_stream_profile>();
	auto intrinsics			= stream.get_intrinsics();
	
	// Capture 30 frames to give autoexposure, etc. a chance to settle... 
	for (auto i = 0; i < 30; ++i) pipe.wait_for_frames();
	ready = true; 
	
	Window app(1366, 768, "RealSense"); 
	while (app) 
	{
		rs2::frameset processed			= align.process(pipe.wait_for_frames()); 
		rs2::video_frame video_frame	= processed.first(align_to); 
		rs2::depth_frame depth_frame	= processed.get_depth_frame(); 
		const uint16_t * Dframe			= reinterpret_cast<const uint16_t*>(depth_frame.get_data()); 

		app.show(video_frame, Rec[1]); 
		app.show(Cmap.process(depth_frame), Rec[0]); 

		if (!ready) 
		{
			Fcount++; 
			std::string FC		= std::to_string(Fcount); 
			SavePNG("Color"		+ FC, W, H,   video_frame); 
			SavePNG("Depth"		+ FC, W, H,   Cmap.process(depth_frame)); 
			SaveDAT("RGBD"		+ FC, Dframe, intrinsics, 0.5f, 10.0f, Scale); 
			app.title("Frame "	+ FC); 
			ready = true; 
		}
	}
	return EXIT_SUCCESS;
}
catch (const rs2::error & E)
{
	std::cerr << "RealSense error at " << E.get_failed_function() << "(" << E.get_failed_args() << ") \n    " << E.what() << "\n";
	return EXIT_FAILURE;
}
catch (const std::exception & E)
{
	std::cerr << E.what() << "\n";
	return EXIT_FAILURE;
}



