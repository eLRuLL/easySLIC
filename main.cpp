#include <highgui.h>
#include <iostream>
#include "slic.h"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

/// Matrices to store images



// int main( int argc, char** argv )
// {
//   /// Read image ( same size, same type )
//   Mat *image;
//   image = new Mat(imread(argv[1]));
//   Mat *lab_image = new Mat(Mat::zeros(image->rows,image->cols,CV_8UC3));

//   if( !image->data ){
//    cout << "Error loading image \n"; 
//    return -1; 
//   }

//   /// Create Windows
//   cvtColor(*image, *lab_image, CV_BGR2Lab);
//   int w = image->cols, h = image->rows;
  
//   int nr_superpixels = atoi(argv[2]);
//   int nc = atoi(argv[3]);

//   int step = sqrt((w * h) / (double) nr_superpixels);
//   cout<<"step: "<<step<<endl;
//   cout << lab_image->at<Vec3b>(0,0) << endl;
//   Slic slic;
//   slic.generate_superpixels(lab_image, step, nc);
//   // slic.create_connectivity(lab_image);
//   // slic.colour_with_cluster_means(lab_image);

//   /* Display the contours and show the result. */
//   slic.display_contours(image, CV_RGB(255,0,0));
//   imshow("result", *image);
//   waitKey(0);
//   imshow("result", *lab_image);
//   waitKey(0);
//   // cvSaveImage(argv[4], image);

//   /// Wait until user press some key
//   //waitKey(0);
//   return 0;

//   delete image;
//   delete lab_image;
// }

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

  int step = sqrt((w * h) / (double) nr_superpixels);
  cout<<"step: "<<step<<endl;
  Slic slic;
  slic.generate_superpixels(lab_image, step, nc);
  slic.create_connectivity(lab_image);

  /* Display the contours and show the result. */
  slic.display_contours(image, CV_RGB(255,0,0));
  cvShowImage("result", image);
  cvWaitKey(0);
  // cvSaveImage(argv[4], image);
  // imshow("SLIC",lab_image);
  //imshow("SLIC",image);

  /// Wait until user press some key
  //waitKey(0);
  return 0;
}
