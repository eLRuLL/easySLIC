#include <cv.h>
#include <highgui.h>
#include <iostream>
#include "slic.h"

using namespace cv;
using namespace std;

/// Matrices to store images



int main( int argc, char** argv )
{
  /// Read image ( same size, same type )
  Mat algo = imread(argv[1]);
  IplImage *image = new IplImage(algo);
  IplImage *lab_image = cvCloneImage(image);

  //if( !image->data ) { printf("Error loading image \n"); return -1; }

  /// Create Windows
  cvCvtColor(image, lab_image, CV_BGR2Lab);
  int w = image->width, h = image->height;
  int nr_superpixels = atoi(argv[2]);
  int nc = atoi(argv[3]);

  double step = sqrt((w * h) / (double) nr_superpixels);
  Slic slic;
  slic.generate_superpixels(lab_image, step, nc);
  slic.create_connectivity(lab_image);

  /* Display the contours and show the result. */
  slic.display_contours(image, CV_RGB(255,0,0));
  cvShowImage("result", image);
  cvWaitKey(0);
  cvSaveImage(argv[4], image);
  //imshow("SLIC",lab_image);
  //imshow("SLIC",image);

  /// Wait until user press some key
  //waitKey(0);
  return 0;
}
