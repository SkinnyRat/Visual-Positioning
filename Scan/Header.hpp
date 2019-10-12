#pragma once

#include <map>
#include <cmath>
#include <string>
#include <iomanip>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <functional>
#include <thread>

#define GLFW_INCLUDE_GLU
#include <glfw3.h>
#include <librealsense2/rs.hpp>


#ifndef PI
const double	PI = 3.14159265358979323846;
#endif
const size_t	IMU_FRAME_WIDTH  = 640;
const size_t	IMU_FRAME_HEIGHT = 480;

const int		CORES = 4; 
const int		W     = 640, H   = 480; 
const			std::string  DIR = "Data/Directory"; 

struct float3 
{
	float x, y, z;
	float3 operator*(float t)
	{
		return { x * t, y * t, z * t };
	}
	float3 operator-(float t)
	{
		return { x - t, y - t, z - t };
	}
	void operator*=(float t)
	{
		x = x * t;
		y = y * t;
		z = z * t;
	}
	void operator=(float3 other)
	{
		x = other.x;
		y = other.y;
		z = other.z;
	}
	void add(float t1, float t2, float t3)
	{
		x += t1;
		y += t2;
		z += t3;
	}
};
struct float2	{ float x, y; };
struct rect		{ float x, y; float w, h; };

const rect		Rec[] = { {140, 40, 480, 640}, {700, 40, 480, 640} }; 
const int		AW = 480, AH = 640; 
bool  ready; 


float DepthScale(rs2::device dev)
{
	for (rs2::sensor & sensor : dev.query_sensors())
		if (rs2::depth_sensor dpt = sensor.as<rs2::depth_sensor>())
			return dpt.get_depth_scale();
	throw std::runtime_error("Device does not have depth sensor!");
}

rs2_stream AlignStream(const std::vector<rs2::stream_profile> & streams)
{
	rs2_stream align_to = RS2_STREAM_ANY;
	bool depth_stream_found = false;
	bool color_stream_found = false;
	for (rs2::stream_profile sp : streams)
	{
		rs2_stream profile_stream = sp.stream_type();
		if (profile_stream != RS2_STREAM_DEPTH)
		{
			if (!color_stream_found)					align_to = profile_stream;
			if (profile_stream == RS2_STREAM_COLOR)		color_stream_found = true;
		}
		else	depth_stream_found = true;
	}
	if (!depth_stream_found)							throw std::runtime_error("No depth stream available!");
	if (align_to == RS2_STREAM_ANY)						throw std::runtime_error("Unable to find stream to align with!");
	return align_to;
}


void SavePNG(const std::string & Filename, int W, int H, const rs2::video_frame & frame)
{
	cv::Mat PNG; 
	cv::Mat Out(H, W, CV_8UC3, (uchar*)frame.get_data()); 
	cv::rotate(Out, Out, cv::ROTATE_90_COUNTERCLOCKWISE); 
	cv::cvtColor(Out, PNG, cv::COLOR_BGR2RGB); 
	cv::imwrite(DIR + Filename + ".png", PNG); 
	std::cout << Filename << " data: " << W << "x" << H << "x" << frame.get_bytes_per_pixel() << "\n"; 
}

void SaveDAT(const std::string & Filename, const uint16_t * Dframe, rs2_intrinsics & Itr, float Min, float Max, float Scale)
{
	auto St = std::chrono::high_resolution_clock::now();
	std::stringstream SSS; SSS << std::fixed << std::setprecision(2);

	std::vector<std::vector<std::string>> Coords; 
	for (int i=0; i<W; i++) 
	{
		std::vector<std::string> Temp; Temp.resize(W); 
		Coords.push_back(Temp); 
	}
	// Extract depth data & rotate 90' counterclockwise. 
	for (int y = 0; y < H; y++)
	{
		auto index = y * W;
		for (int x = 0; x < W; x++, ++index)
		{
			float point[3] = { 0.f,0.f,0.f }; float pixel[2] = { x,y }; float depth = Dframe[index] * Scale;
			if (depth > Min && depth < Max)
				rs2_deproject_pixel_to_point(point, &Itr, pixel, depth);
			SSS << point[0] << "," << point[1] << "," << point[2];
			Coords[x][y] = SSS.str();
			SSS.str(std::string());
		}
	}
	std::ofstream DAT(DIR + Filename + ".dat"); 
	for (int i=0; i<W; i++) 
	{
		for (int j=0; j<H; j++) 
			DAT << Coords[W-1-i][j] << " "; 
		DAT << "\n"; 
	}
	auto Ex = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - St).count();
	std::cout << "Depth data saved in " << Ex / 1000.f << " s.\n";
}



class Texture
{
public:

	void upload(const rs2::video_frame& frame)
	{
		if (!frame)			return;
		if (!_gl_handle)	glGenTextures(1, &_gl_handle);
		GLenum err			= glGetError();
		auto width			= frame.get_width();
		auto height			= frame.get_height();
		auto format			= frame.get_profile().format();
		_stream_type		= frame.get_profile().stream_type();
		_stream_index		= frame.get_profile().stream_index();

		cv::Mat Img(H, W, CV_8UC3, (uchar*)frame.get_data());
		cv::rotate(Img, Img, cv::ROTATE_90_COUNTERCLOCKWISE);
		glBindTexture(GL_TEXTURE_2D, _gl_handle);
		switch (format)
		{
		case RS2_FORMAT_RGB8:
		case RS2_FORMAT_BGR8:
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB,  height, width, 0, GL_RGB,		GL_UNSIGNED_BYTE, Img.ptr());
			break;
		case RS2_FORMAT_RGBA8:
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, height, width, 0, GL_RGBA,		GL_UNSIGNED_BYTE, Img.ptr());
			break;
		case RS2_FORMAT_Y8:
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB,  height, width, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, Img.ptr());
			break;
		default:
			throw std::runtime_error("The requested format is not supported!");
		}
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
		glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
		glBindTexture(GL_TEXTURE_2D, 0);
	}

	void show(const rect& r, float alpha = 1.f) const
	{
		if (!_gl_handle) return;
		glViewport((int)r.x, (int)r.y, (int)r.w, (int)r.h);
		glLoadIdentity();
		glMatrixMode(GL_PROJECTION);
		glOrtho(0, r.w, r.h, 0, -1, +1);
		glBindTexture(GL_TEXTURE_2D, _gl_handle);
		glColor4f(1.0f, 1.0f, 1.0f, alpha);

		glEnable(GL_TEXTURE_2D);
		glBegin(GL_QUADS);
		glTexCoord2f(0, 0); glVertex2f(0, 0);
		glTexCoord2f(0, 1); glVertex2f(0, r.h);
		glTexCoord2f(1, 1); glVertex2f(r.w, r.h);
		glTexCoord2f(1, 0); glVertex2f(r.w, 0);
		glEnd();
		glDisable(GL_TEXTURE_2D);
		glBindTexture(GL_TEXTURE_2D, 0);
	}

	GLuint get_gl_handle() { return _gl_handle; }

	void render(const rs2::frame& frame, const rect& rect, float alpha = 1.f)
	{
		if (auto vf = frame.as<rs2::video_frame>())
		{
			upload(vf);
			show(rect, alpha);
		}
		else
		{
			throw std::runtime_error("Rendering is currently supported for video frames only!");
		}
	}

private:
	GLuint          _gl_handle   = 0;
	rs2_stream      _stream_type = RS2_STREAM_ANY;
	int             _stream_index{};
};


class Window
{
public:
	std::function<void(bool)>           on_left_mouse	= [](bool) {};
	std::function<void(double, double)> on_mouse_scroll = [](double, double) {};
	std::function<void(double, double)> on_mouse_move	= [](double, double) {};
	std::function<void(int)>            on_key_release	= [](int) {};

	Window(int width, int height, const char* title) : _width(width), _height(height)
	{
		glfwInit();
		win = glfwCreateWindow(width, height, title, nullptr, nullptr);
		if (!win)
			throw std::runtime_error("Could not open OpenGL window!");
		glfwMakeContextCurrent(win);

		glfwSetWindowUserPointer(win,   this);
		glfwSetMouseButtonCallback(win, [](GLFWwindow * w, int button, int action, int mods)
		{
			auto s = (Window*)glfwGetWindowUserPointer(w);
			if (button == 0) s->on_left_mouse(action == GLFW_PRESS);
		});

		glfwSetScrollCallback(win, [](GLFWwindow * w, double xoffset, double yoffset)
		{
			auto s = (Window*)glfwGetWindowUserPointer(w);
			s->on_mouse_scroll(xoffset, yoffset);
		});

		glfwSetCursorPosCallback(win, [](GLFWwindow * w, double x, double y)
		{
			auto s = (Window*)glfwGetWindowUserPointer(w);
			s->on_mouse_move(x, y);
		});

		glfwSetKeyCallback(win, [](GLFWwindow * w, int key, int scancode, int action, int mods)
		{
			auto s = (Window*)glfwGetWindowUserPointer(w);
			if (0 == action)  // on key release 
			{
				s->on_key_release(key); 
				if (key == GLFW_KEY_ENTER) 
				{
					if (ready) 
					{
						ready = false; 
						glfwSetWindowTitle(w, "RealSense: SAVING..."); 
					}
				}
			}
		});
	}
	~Window()
	{
		glfwDestroyWindow(win);
		glfwTerminate();
	}

	void close()	{ glfwSetWindowShouldClose(win, 1); }

	float width()	const { return float(_width);  }
	float height()	const { return float(_height); }

	operator bool()
	{
		glPopMatrix();
		glfwSwapBuffers(win);
		auto res = !glfwWindowShouldClose(win);

		glfwPollEvents();
		glfwGetFramebufferSize(win, &_width, &_height);
		glClear(GL_COLOR_BUFFER_BIT);
		glViewport(0, 0, _width, _height);
		glPushMatrix();
		glfwGetWindowSize(win, &_width, &_height);
		glOrtho(0, _width, _height, 0, -1, +1);
		return res;
	}

	void title(std::string text)
	{
		std::string out = "RealSense: " + text; 
		glfwSetWindowTitle(win,		out.c_str()); 
	}

	void show(rs2::frame frame) { show(frame, { 0, 0, AW, AH }); }

	void show(const rs2::frame& frame, const rect& rect)
	{
		if (auto fs = frame.as<rs2::frameset>())
			render_frameset(fs, rect);
		if (auto vf = frame.as<rs2::video_frame>())
			render_video_frame(vf, rect);
	}
	
	operator GLFWwindow*() { return win; }

private:
	GLFWwindow *  win;
	std::map<int, Texture> _textures;
	int  _width,  _height;

	void render_video_frame(const rs2::video_frame& f, const rect& r)
	{
		auto& t = _textures[f.get_profile().unique_id()];
		t.render(f, r);
	}
	
	void render_frameset(const rs2::frameset& frames, const rect& r)
	{
		std::vector<rs2::frame> supported_frames;
		for (auto f : frames)
			if (can_render(f))
				supported_frames.push_back(f);
		if (supported_frames.empty())  return;

		std::sort(supported_frames.begin(), supported_frames.end(), [](rs2::frame first, rs2::frame second)
		{ 
			return first.get_profile().stream_type() < second.get_profile().stream_type(); 
		});
		std::vector<rect> image_grid = { Rec[0], Rec[1] };
		int image_index = 0;
		for (auto f : supported_frames)
		{
			auto r = image_grid.at(image_index);
			show(f, r);
			image_index++;
		}
	}

	bool can_render(const rs2::frame& f) const
	{
		auto format = f.get_profile().format();
		switch (format)
		{
		case RS2_FORMAT_RGB8:
		case RS2_FORMAT_RGBA8:
		case RS2_FORMAT_Y8:
		case RS2_FORMAT_MOTION_XYZ32F:
			return true;
		default:
			return false;
		}
	}
};














