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

GLuint in_buffer;
GLuint out_buffer;
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
	// IplImage *image = new IplImage(*img);
	// IplImage *lab_image = cvCloneImage(image);

    M = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
    N = cap.get(CV_CAP_PROP_FRAME_WIDTH);
    int C = img->channels();



	// these GLUT calls need to be made before the other GL calls
	glutInit( &argc, argv );
	glutInitDisplayMode( GLUT_DOUBLE | GLUT_RGBA );
	glutInitWindowSize( N, M );
	glutCreateWindow( "bitmap" );



	glGenBuffers( 1, &in_buffer );
	glBindBuffer( GL_PIXEL_UNPACK_BUFFER_ARB, in_buffer );
	glBufferData( GL_PIXEL_UNPACK_BUFFER_ARB, M * N * C,NULL, GL_STREAM_DRAW_ARB );

	glGenBuffers( 1, &out_buffer );
	glBindBuffer( GL_PIXEL_UNPACK_BUFFER_ARB, out_buffer );
	glBufferData( GL_PIXEL_UNPACK_BUFFER_ARB, M * N * C,NULL, GL_STREAM_DRAW_ARB );

	int nr_superpixeles = 100;
	int m_variable = 40;
	interop_setup(M,N, nr_superpixeles, m_variable);
	interop_register_buffer(in_buffer, out_buffer);

	// int nr_superpixels = 100;
	// int nc = 40;
 //  	int step = sqrt((M * N) / (double) nr_superpixels);

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

        // image = new IplImage(*img);
        // cvtColor(image, lab_image, CV_BGR2Lab);
        // unsigned char *imgdata = (unsigned char*)(IplImage(img)->data);

        imgdata = (unsigned char*)(img->data);
		glBindBuffer( GL_PIXEL_UNPACK_BUFFER_ARB, in_buffer );
        glBufferSubData( GL_PIXEL_UNPACK_BUFFER_ARB, 0,  M * N * C,imgdata);

		// slic.generate_superpixels(lab_image, step, nc);
		  // slic.create_connectivity(lab_image);
		  // slic.display_contours(image, CV_RGB(255,0,0));

        interop_map();
        interop_run(M,N);

        // imshow("MyWindow", *image);
        // waitKey(0);

        // if(waitKey(1) == 27) //wait for 'esc' key press for 30 ms. If 'esc' key is pressed, break loop
        // {
        //     cout << "esc key is pressed by user" << endl;
        //     break;
        // }

        		// Flip Upside Down (warning: deprecated functions)
		glRasterPos2f(-1,1);
		glPixelZoom( 1, -1 );

		// glDrawPixels is also deprecated, use textures instead
		glBindBuffer( GL_PIXEL_UNPACK_BUFFER_ARB, out_buffer );
		glDrawPixels( N, M, GL_BGR, GL_UNSIGNED_BYTE, 0 );
		glutSwapBuffers();

    }
	// set up GLUT and kick off main loop
	// glutKeyboardFunc( key_func );
	glutMainLoop();
	// glutDisplayFunc( draw_func );

    delete img;

	return 0;
}
