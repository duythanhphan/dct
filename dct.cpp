#include <stdlib.h>
#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

using namespace cv; 
using namespace std;

const double DISPLAY_SCALE = 10;
const unsigned int AXIS_PADDING = 50;
const unsigned int TICK_SIZE = 5;
const double MINOR_SCALE = 10;
const double MAJOR_SCALE = 50;
const Scalar LINE_COLOR(0, 0, 0);
const int FONT_FACE = FONT_HERSHEY_PLAIN;
const double MAJOR_TICK_FONT_SCALE = 0.8;

void original_mouse_callback(int event, int x, int y, int flags, void* param) {
    int img_x = x - AXIS_PADDING;
    int img_y = y - AXIS_PADDING;
    stringstream mouse_pos;
    mouse_pos << "Mouse at (" << img_x << ", " << img_y << ")";
    const char* mouse_pos_str = mouse_pos.str().c_str();
    double font_thickness = 1;
    int baseline;
    Size text_size = getTextSize(mouse_pos_str, FONT_FACE, MAJOR_TICK_FONT_SCALE, font_thickness, &baseline);
    rectangle(*(Mat*)param, Point(0, 0), Point(((Mat*)param)->cols, 2*text_size.height+10),
              Scalar(255, 255, 255), CV_FILLED);
    if (img_x >= 0 && img_y >= 0 &&
        img_x <= ((Mat*)param)->cols - 2*AXIS_PADDING && img_y <= ((Mat*)param)->rows - 2*AXIS_PADDING) {
      putText(*(Mat*)param, mouse_pos_str, Point(10, 10+text_size.height),
              FONT_FACE, MAJOR_TICK_FONT_SCALE, Scalar(0, 0, 0), font_thickness, CV_AA);
    }
    imshow("Input", *(Mat*)param);
}

void dct_mouse_callback(int event, int x, int y, int flags, void* param) {
    int img_x = (int)(x - AXIS_PADDING)/DISPLAY_SCALE;
    int img_y = (int)(y - AXIS_PADDING)/DISPLAY_SCALE;
    stringstream mouse_pos;
    mouse_pos << "Mouse at (" << img_x << ", " << img_y << ")";
    const char* mouse_pos_str = mouse_pos.str().c_str();
    double font_thickness = 1;
    int baseline;
    Size text_size = getTextSize(mouse_pos_str, FONT_FACE, MAJOR_TICK_FONT_SCALE, font_thickness, &baseline);
    rectangle(*(Mat*)param, Point(0, 0), Point(((Mat*)param)->cols, 2*text_size.height+10),
              Scalar(255, 255, 255), CV_FILLED);
    if (img_x >= 0 && img_y >= 0 &&
        img_x*DISPLAY_SCALE <= ((Mat*)param)->cols - 2*AXIS_PADDING &&
        img_y*DISPLAY_SCALE <= ((Mat*)param)->rows - 2*AXIS_PADDING) {
      putText(*(Mat*)param, mouse_pos_str, Point(10, 10+text_size.height),
              FONT_FACE, MAJOR_TICK_FONT_SCALE, Scalar(0, 0, 0), font_thickness, CV_AA);
    }
    imshow("DCT", *(Mat*)param);
}

int main(int argc, char** argv) {
  if (argc != 2) {
    fprintf(stdout, "usage: %s image\n", argv[0]);
    return 1;
  }
  stringstream ss, ss1;
  ss << argv[1];
  string input_location = ss.str();
  string filename = input_location.substr(6);
  ss1 << "output/" << filename;
  const char* output_location = ss1.str().c_str();
  Mat img = imread(input_location, CV_LOAD_IMAGE_GRAYSCALE);

  vector<Mat> planes;
  split(img, planes);
  vector<Mat> outplanes(planes.size());
  for (size_t ii = 0; ii < planes.size(); ii++) {
    planes[ii].convertTo(planes[ii], CV_32FC1);
    dct(planes[ii], outplanes[ii]);
  }
  Mat merged;
  merge(outplanes, merged);

  // generate a display input image (axis)
  Rect input_rect(AXIS_PADDING, AXIS_PADDING, img.cols, img.rows);
  Mat original_output(img.rows+2*AXIS_PADDING, img.cols+2*AXIS_PADDING, img.type(), Scalar(255,255,255));
  img.copyTo(original_output(input_rect));

  // generate a display DCT image (zoomed, cropped, axis)
  Rect dct_rect(AXIS_PADDING, AXIS_PADDING, merged.cols, merged.rows);
  Rect original_dct_size(0, 0, merged.cols, merged.rows);
  Mat dct_output;
  merged.convertTo(dct_output, CV_8UC1);
  Mat output(dct_output.rows+2*AXIS_PADDING, dct_output.cols+2*AXIS_PADDING, dct_output.type(), Scalar(255,255,255));
  resize(dct_output, dct_output, Size(dct_output.cols*DISPLAY_SCALE, dct_output.rows*DISPLAY_SCALE));
  dct_output = dct_output(original_dct_size);
  dct_output.copyTo(output(dct_rect));

  for (size_t x = 0; x < 1000; ++x) {
    if (MINOR_SCALE*x > merged.cols) break;
    line(output, Point(MINOR_SCALE*x+AXIS_PADDING, AXIS_PADDING - TICK_SIZE),
                 Point(MINOR_SCALE*x+AXIS_PADDING, AXIS_PADDING), LINE_COLOR);
  }
  for (size_t x = 0; x < 1000; ++x) {
    line(output, Point(MAJOR_SCALE*x+AXIS_PADDING, AXIS_PADDING - 2*TICK_SIZE),
                 Point(MAJOR_SCALE*x+AXIS_PADDING, AXIS_PADDING), LINE_COLOR);
    if (MAJOR_SCALE/DISPLAY_SCALE*x <= merged.cols) {
      line(original_output, Point(MAJOR_SCALE/DISPLAY_SCALE*x+AXIS_PADDING, AXIS_PADDING - 2*TICK_SIZE),
                   Point(MAJOR_SCALE/DISPLAY_SCALE*x+AXIS_PADDING, AXIS_PADDING), LINE_COLOR);
    }
    stringstream num_ss;
    num_ss << MAJOR_SCALE/DISPLAY_SCALE*x;
    const char* num = num_ss.str().c_str();
    double font_thickness = 0.5;
    Size text_size;
    int baseline;
    text_size = getTextSize(num, FONT_FACE, MAJOR_TICK_FONT_SCALE, font_thickness, &baseline);
    if (MAJOR_SCALE*x <= merged.cols) {
      putText(output, num, Point(MAJOR_SCALE*x+AXIS_PADDING - text_size.width/2, AXIS_PADDING-TICK_SIZE*3),
              FONT_FACE, MAJOR_TICK_FONT_SCALE, Scalar(0, 0, 0), font_thickness, CV_AA);
    }
  }

  for (size_t y = 0; y < 1000; ++y) {
    if (MINOR_SCALE*y > merged.rows) break;
    line(output, Point(AXIS_PADDING - TICK_SIZE, MINOR_SCALE*y+AXIS_PADDING),
                 Point(AXIS_PADDING, MINOR_SCALE*y+AXIS_PADDING), LINE_COLOR);
  }
  for (size_t y = 0; y < 1000; ++y) {
    line(output, Point(AXIS_PADDING - 2*TICK_SIZE, MAJOR_SCALE*y+AXIS_PADDING),
                 Point(AXIS_PADDING, MAJOR_SCALE*y+AXIS_PADDING), LINE_COLOR);
    if (MAJOR_SCALE/DISPLAY_SCALE*y <= merged.rows) {
      line(original_output, Point(AXIS_PADDING - 2*TICK_SIZE, MAJOR_SCALE/DISPLAY_SCALE*y+AXIS_PADDING),
                   Point(AXIS_PADDING, MAJOR_SCALE/DISPLAY_SCALE*y+AXIS_PADDING), LINE_COLOR);
    }
    stringstream num_ss;
    num_ss << MAJOR_SCALE/DISPLAY_SCALE*y;
    const char* num = num_ss.str().c_str();
    double font_thickness = 0.5;
    Size text_size;
    int baseline;
    text_size = getTextSize(num, FONT_FACE, MAJOR_TICK_FONT_SCALE, font_thickness, &baseline);
    if (MAJOR_SCALE*y <= merged.rows) {
      putText(output, num, Point(AXIS_PADDING-TICK_SIZE*3-text_size.width, MAJOR_SCALE*y+AXIS_PADDING+text_size.height/2),
              FONT_FACE, MAJOR_TICK_FONT_SCALE, Scalar(0, 0, 0), font_thickness, CV_AA);
    }
  }

  namedWindow("Input", CV_WINDOW_AUTOSIZE);
  moveWindow("Input", 1000, 0);
  setMouseCallback("Input", original_mouse_callback, &original_output);
  imshow("Input", original_output);
  namedWindow("DCT", CV_WINDOW_AUTOSIZE);
  moveWindow("DCT", 1000+original_output.cols, 0);
  setMouseCallback("DCT", dct_mouse_callback, &output);
  imshow("DCT", output);
  waitKey(0);

  return 0;
}
