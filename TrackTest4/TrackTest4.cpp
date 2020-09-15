﻿#include <iostream>
#include <algorithm>
#include <Kinect.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "stdafx.h"
#include "kcftracker.hpp"

//#pragma comment(lib, "opencv_world310d.lib")

using namespace std;
using namespace cv;

class BoxExtractor {
public:
	Rect2d extract(Mat img);
	Rect2d extract(const std::string& windowName, Mat img, bool showCrossair = true);

	struct handlerT {
		bool isDrawing;
		Rect2d box;
		Mat image;

		// initializer list
		handlerT() : isDrawing(false) {};
	}params;

private:
	static void mouseHandler(int event, int x, int y, int flags, void *param);
	void opencv_mouse_callback(int event, int x, int y, int, void *param);
};

int main(int argc, char* argv[]) {

	bool HOG = true;
	bool FIXEDWINDOW = false;
	bool MULTISCALE = true;
	bool SILENT = true;
	bool LAB = false;

	IKinectSensor* kinectSensor;
	IColorFrameSource* colorFrameSource = NULL;
	IDepthFrameSource* depthFrameSource = NULL;
	IColorFrameReader* colorFrameReader;
	IDepthFrameReader* depthFrameReader;
	BOOLEAN isAvailable = 0;
	
	GetDefaultKinectSensor(&kinectSensor);
	kinectSensor->Open();
	Sleep(1000);
	kinectSensor->get_IsAvailable(&isAvailable);
	bool a = isAvailable;
	if (a == 1) {
		cout << "Kinect开启成功" << endl;

		kinectSensor->get_ColorFrameSource(&colorFrameSource);
		kinectSensor->get_DepthFrameSource(&depthFrameSource);

		colorFrameSource->OpenReader(&colorFrameReader);
		depthFrameSource->OpenReader(&depthFrameReader);

		IColorFrame* colorFrameInit = nullptr;
		IDepthFrame* depthFrameInit = nullptr;

		HRESULT hResult = colorFrameReader->AcquireLatestFrame(&colorFrameInit);

		while (SUCCEEDED(hResult) == 0) {
			hResult = colorFrameReader->AcquireLatestFrame(&colorFrameInit);
		}
		//color图像尺寸
		int width = 1920;
		int height = 1080;
		unsigned int bufferSize = width * height * 4 * sizeof(unsigned char);

		//1920 * 1080的Mat，4通道
		Mat frameC4Init(height, width, CV_8UC4);

		if (SUCCEEDED(hResult)) {
			//Kinect只能获取4通道的Bgra图像，需要通道转换
			colorFrameInit->CopyConvertedFrameDataToArray(bufferSize, reinterpret_cast<BYTE*>(frameC4Init.data), ColorImageFormat_Bgra);
		}

		
		SafeRelease(colorFrameInit);
		
		//将4通道Kinect图像转换为3通道bgr
		Mat frameC3Init(frameC4Init.size(), CV_8UC3);
		Mat alphaInit(frameC4Init.size(), CV_8UC1);
		Mat outInit[] = { frameC3Init,alphaInit };
		//设置dot的对应关系，去掉4通道每个像素最后的255无效值
		int frome_toInit[] = { 0,0,1,1,2,2,3,3 };
		//进行通道转换。
		mixChannels(&frameC4Init, 1, outInit, 2, frome_toInit, 4);
		cout << frameC3Init.rows << endl;
		cout << frameC3Init.cols << endl;

		KCFTracker tracker(HOG, FIXEDWINDOW, MULTISCALE, LAB);

		BoxExtractor box;

		Rect2d roi = box.extract("tracker", frameC3Init);

		if (roi.width == 0 || roi.height == 0)
			return 0;

		tracker.init(roi, frameC3Init);
		rectangle(frameC3Init, roi, Scalar(0, 255, 255), 1, 8);
		Rect result;

		//1920 * 1080的Mat，4通道
		Mat frameC4(height, width, CV_8UC4);
		//3通道转换准备
		Mat frameC3(frameC4.size(), CV_8UC3);
		Mat alpha(frameC4.size(), CV_8UC1);
		Mat out[] = { frameC3,alpha };
		//设置dot的对应关系，去掉4通道每个像素最后的255无效值
		int frome_to[] = { 0,0,1,1,2,2,3,3 };
		//进行通道转换。
		while (1)
		{
			IColorFrame* colorFrame = nullptr;
			HRESULT sign = colorFrameReader->AcquireLatestFrame(&colorFrame);
			//判断Kinect图像帧是否获取成功，如果失败则跳出本次循环，成功则进行KCF处理
			if (!SUCCEEDED(sign)) {
				continue;
			}
			if (SUCCEEDED(sign)) {
				colorFrame->CopyConvertedFrameDataToArray(bufferSize, reinterpret_cast<BYTE*>(frameC4.data), ColorImageFormat_Bgra);
			}
			SafeRelease(colorFrame);

			//转换为3通道
			mixChannels(&frameC4, 1, out, 2, frome_to, 4);

			if (frameC3.rows == 0 || frameC3.cols == 0)
				break;

			result = tracker.update(frameC3);

			rectangle(frameC3, Point(result.x, result.y), Point(result.x + result.width, result.y + result.height), Scalar(0, 255, 255), 1, 8);
			imshow("Image", frameC3);
			
			if (waitKey(1) == 27)break;
		}

	}

	else {
		cout << "Kinect开启失败" << endl;
	}
}

void BoxExtractor::mouseHandler(int event, int x, int y, int flags, void *param) {
	BoxExtractor *self = static_cast<BoxExtractor*>(param);
	self->opencv_mouse_callback(event, x, y, flags, param);
}

void BoxExtractor::opencv_mouse_callback(int event, int x, int y, int, void *param) {
	handlerT * data = (handlerT*)param;
	switch (event) {
		// update the selected bounding box
	case EVENT_MOUSEMOVE:
		if (data->isDrawing) {
			data->box.width = x - data->box.x;
			data->box.height = y - data->box.y;
		}
		break;

		// start to select the bounding box
	case EVENT_LBUTTONDOWN:
		data->isDrawing = true;
		data->box = cvRect(x, y, 0, 0);
		break;

		// cleaning up the selected bounding box
	case EVENT_LBUTTONUP:
		data->isDrawing = false;
		if (data->box.width < 0) {
			data->box.x += data->box.width;
			data->box.width *= -1;
		}
		if (data->box.height < 0) {
			data->box.y += data->box.height;
			data->box.height *= -1;
		}
		break;
	}
}

Rect2d BoxExtractor::extract(Mat img) {
	return extract("Bounding Box Extractor", img);
}

Rect2d BoxExtractor::extract(const std::string& windowName, Mat img, bool showCrossair) {

	int key = 0;

	// show the image and give feedback to user
	imshow(windowName, img);
	//printf("Select an object to track and then press SPACE/BACKSPACE/ENTER button!\n");

	// copy the data, rectangle should be drawn in the fresh image
	params.image = img.clone();

	// select the object
	setMouseCallback(windowName, mouseHandler, (void *)&params);

	// end selection process on SPACE (32) BACKSPACE (27) or ENTER (13)
	while (!(key == 32 || key == 27 || key == 13)) {
		// draw the selected object
		rectangle(
			params.image,
			params.box,
			Scalar(255, 0, 0), 2, 1
		);

		// draw cross air in the middle of bounding box
		if (showCrossair) {
			// horizontal line
			line(
				params.image,
				Point((int)params.box.x, (int)(params.box.y + params.box.height / 2)),
				Point((int)(params.box.x + params.box.width), (int)(params.box.y + params.box.height / 2)),
				Scalar(255, 0, 0), 2, 1
			);

			// vertical line
			line(
				params.image,
				Point((int)(params.box.x + params.box.width / 2), (int)params.box.y),
				Point((int)(params.box.x + params.box.width / 2), (int)(params.box.y + params.box.height)),
				Scalar(255, 0, 0), 2, 1
			);
		}

		// show the image bouding box
		imshow(windowName, params.image);

		// reset the image
		params.image = img.clone();

		//get keyboard event
		key = waitKey(1);
	}


	return params.box;
}





