#define GL_GLEXT_PROTOTYPES
#include <GL/glut.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#define DIM 512
#include "superpixels.h"
#include "slic.h"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

GLuint in_buffer;
GLuint out_buffer;
GLuint cluster_buffer;
GLuint distance_buffer;
GLuint centercolor_buffer;
GLuint centercoord_buffer;
GLuint centercount_buffer;
int M,N;

static void key_func( unsigned char key, int x, int y ) {
	switch (key) {
	case 27:
	// clean up OpenGL and CUDA
	// interop_cleanup();
	glBindBuffer( GL_PIXEL_UNPACK_BUFFER_ARB, 0 );
	glDeleteBuffers( 1, &in_buffer );
	glDeleteBuffers( 1, &out_buffer );
	exit(0);
	}
}

int main( int argc, char **argv ) {

 //    Mat* img;
 //    img = new Mat(imread(argv[1]));
	// unsigned char *imgdata = (unsigned char*)(img->data);
 //    M = img->rows;
 //    N = img->cols;
 //    int C = img->channels();

	VideoCapture cap(argv[1]);
    Mat* img;
    img = new Mat(Mat::zeros(cap.get(CV_CAP_PROP_FRAME_WIDTH),cap.get(CV_CAP_PROP_FRAME_HEIGHT),CV_8UC3));
	unsigned char *imgdata;

    M = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
    N = cap.get(CV_CAP_PROP_FRAME_WIDTH);
    int C = img->channels();

	// these GLUT calls need to be made before the other GL calls
	glutInit( &argc, argv );
	glutInitDisplayMode( GLUT_DOUBLE | GLUT_RGBA );
	glutInitWindowSize( N, M );
	glutCreateWindow( "bitmap" );


	int nr_superpixels = 100;
	int nc = 40;
  	int step = sqrt((M * N) / (double) nr_superpixels);

  	int hor = (N - step/2)/step;
  	int vert = (M - step/2)/step;
  	int ncenters = hor*vert;

  	cout << "Clusters: " << ncenters << endl;

	glGenBuffers( 1, &in_buffer );
	glBindBuffer( GL_PIXEL_UNPACK_BUFFER_ARB, in_buffer );
	glBufferData( GL_PIXEL_UNPACK_BUFFER_ARB, M * N * C,NULL, GL_STREAM_DRAW_ARB );

	glGenBuffers( 1, &out_buffer );
	glBindBuffer( GL_PIXEL_UNPACK_BUFFER_ARB, out_buffer );
	glBufferData( GL_PIXEL_UNPACK_BUFFER_ARB, M * N * C,NULL, GL_STREAM_DRAW_ARB );

	glGenBuffers( 1, &cluster_buffer );
	glBindBuffer( GL_PIXEL_UNPACK_BUFFER_ARB, cluster_buffer );
	glBufferData( GL_PIXEL_UNPACK_BUFFER_ARB, M * N * sizeof(int),NULL, GL_DYNAMIC_DRAW_ARB );

	glGenBuffers( 1, &distance_buffer );
	glBindBuffer( GL_PIXEL_UNPACK_BUFFER_ARB, distance_buffer );
	glBufferData( GL_PIXEL_UNPACK_BUFFER_ARB, M * N * sizeof(float),NULL, GL_DYNAMIC_DRAW_ARB );

	glGenBuffers( 1, &centercolor_buffer );
	glBindBuffer( GL_PIXEL_UNPACK_BUFFER_ARB, centercolor_buffer );
	glBufferData( GL_PIXEL_UNPACK_BUFFER_ARB, ncenters * 3 * sizeof(float),NULL, GL_DYNAMIC_DRAW_ARB );

	glGenBuffers( 1, &centercoord_buffer );
	glBindBuffer( GL_PIXEL_UNPACK_BUFFER_ARB, centercoord_buffer );
	glBufferData( GL_PIXEL_UNPACK_BUFFER_ARB, ncenters * 2 * sizeof(int),NULL, GL_DYNAMIC_DRAW_ARB );

	glGenBuffers( 1, &centercount_buffer );
	glBindBuffer( GL_PIXEL_UNPACK_BUFFER_ARB, centercount_buffer );
	glBufferData( GL_PIXEL_UNPACK_BUFFER_ARB, ncenters * sizeof(int),NULL, GL_DYNAMIC_DRAW_ARB );

  	interop_setup(M,N, ncenters, step, nc);
	interop_register_buffer(in_buffer, out_buffer, cluster_buffer, distance_buffer, centercolor_buffer, centercoord_buffer, centercount_buffer);

	// int ticks = 0;
	// Slic slic;
    while(1)
    {
        bool bSuccess = cap.read(*img);

        if (!bSuccess) //if not success, break loop
        {
            cout << "Cannot read the frame from video file" << endl;
            break;
        }

        cvtColor(*img, *img, CV_BGR2Lab);
        imgdata = (unsigned char*)(img->data);
		glBindBuffer( GL_PIXEL_UNPACK_BUFFER_ARB, in_buffer );
        glBufferSubData( GL_PIXEL_UNPACK_BUFFER_ARB, 0,  M * N * C,imgdata);

        interop_map();
        interop_run(M,N,hor,vert, ncenters, imgdata);

        // imshow("MyWindow", *image);
        // waitKey(0);

        // Flip Upside Down (warning: deprecated functions)
		glRasterPos2f(-1,1);
		glPixelZoom( 1, -1 );

		// glDrawPixels is also deprecated, use textures instead
		// glBindBuffer( GL_PIXEL_UNPACK_BUFFER_ARB, out_buffer );
		glDrawPixels( N, M, GL_BGR, GL_UNSIGNED_BYTE, 0 );
		glutSwapBuffers();

    }
	// set up GLUT and kick off main loop
	// glutKeyboardFunc( key_func );
	// glutMainLoop();
	// glutDisplayFunc( draw_func );
	exit(0);
    delete img;
    interop_cleanup();

	return 0;
}