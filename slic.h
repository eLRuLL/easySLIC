#ifndef SLIC_H
#define SLIC_H

/* slic.h.
 *
 * Written by: eLRuLL y Juanca el bacan
 *
 */

#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <stdio.h>
#include <math.h>
#include <vector>
#include <float.h>
using namespace std;
using namespace cv;

/* 2d matrices are handled by 2d vectors. */
#define vec2dd vector<vector<double> >
#define vec2di vector<vector<int> >
#define vec2db vector<vector<bool> >
/* The number of iterations run by the clustering algorithm. */
#define NR_ITERATIONS 10

class Slic {
    private:
        /* The cluster assignments and distance values for each pixel. */
        vec2di clusters;
        vec2dd distances;

        /* The LAB and xy values of the centers. */
        vec2dd centers;
        /* The number of occurences of each center. */
        vector<int> center_counts;

        /* The step size per cluster, and the colour (nc) and distance (ns)
         * parameters. */
        int step, nc, ns;
        int N;

        /* Compute the distance between a center and an individual pixel.
          Equation 1
        */
        double compute_dist(int ci, CvPoint pixel, CvScalar colour){
          double dc = sqrt(pow(centers[ci][0] - colour.val[0], 2) + pow(centers[ci][1]
          - colour.val[1], 2) + pow(centers[ci][2] - colour.val[2], 2));
          double ds = sqrt(pow(centers[ci][3] - pixel.x, 2) + pow(centers[ci][4] - pixel.y, 2));

          //return sqrt(pow(dc / nc, 2) + pow(ds / ns, 2));

          double m_value = 40.0;
          //  S = sqrt(N/K)
          //    N = number of pixels.
          //    K = number of superpixels.
          double K = 100;
          double S_value = sqrt(N/K);
          return dc + (m_value/S_value)*ds;

          //double w = m_value / (pow(ns / nc, 2));
          //return sqrt(dc) + sqrt(ds * w);
        }
        /* Find the pixel with the lowest gradient in a 3x3 surrounding. */
        CvPoint find_local_minimum(IplImage *image, CvPoint center){
          int i,j;
          double min_grad = FLT_MAX;
          CvPoint loc_min = cvPoint(center.x, center.y);

          for (i = center.x-1; i < center.x+2; ++i) {
            for (j = center.y-1; j < center.y+2; ++j) {
              CvScalar c1 = cvGet2D(image, j+1, i);
              CvScalar c2 = cvGet2D(image, j, i+1);
              CvScalar c3 = cvGet2D(image, j, i);
              /* get L values. */
              double i1 = c1.val[0];
              double i2 = c2.val[0];
              double i3 = c3.val[0];
              /*double i1 = c1.val[0] * 0.11 + c1.val[1] * 0.59 + c1.val[2] * 0.3;
              double i2 = c2.val[0] * 0.11 + c2.val[1] * 0.59 + c2.val[2] * 0.3;
              double i3 = c3.val[0] * 0.11 + c3.val[1] * 0.59 + c3.val[2] * 0.3;*/

              /* Compute horizontal and vertical gradients and keep track of the
              minimum. */
              if (sqrt(pow(i1 - i3, 2)) + sqrt(pow(i2 - i3,2)) < min_grad) {
                min_grad = fabs(i1 - i3) + fabs(i2 - i3);
                loc_min.x = i;
                loc_min.y = j;
              }
            }
          }

          return loc_min;
        }

        /* Remove and initialize the 2d vectors. */
        void clear_data(){
          clusters.clear();
          distances.clear();
          centers.clear();
          center_counts.clear();
        }

        /* Initializes:
          * centers
          * center_counts
          * clusters
          * distances
          */
        void init_data(IplImage *image){
          int i,j;
          /* Initialize the cluster and distance matrices with default
             values for each pixel. */
          for (i = 0; i < image->width; ++i){
            vector<int> cr(image->height, -1);
            vector<double> dr(image->height, FLT_MAX);
            clusters.push_back(cr);
            distances.push_back(dr);
          }

          /* Initialize the centers and counters. */
          for (i = step; i < image->width - step/2; i += step) {
            for (j = step; j < image->height - step/2; j += step) {
              vector<double> center;
              /* Find the local minimum (gradient-wise on 3x3 space). */
              CvPoint nc = find_local_minimum(image, cvPoint(i,j));
              CvScalar colour = cvGet2D(image, nc.y, nc.x);

              /* Generate the center vector. */
              center.push_back(colour.val[0]);
              center.push_back(colour.val[1]);
              center.push_back(colour.val[2]);
              center.push_back(nc.x);
              center.push_back(nc.y);

              /* Append to vector of centers. */
              centers.push_back(center);
              center_counts.push_back(0);
            }
          }
        }

    public:
        Slic(){};
        ~Slic(){clear_data();}

        /* Generate an over-segmentation for an image. */
        void generate_superpixels(IplImage *image, int step, int nc){
          //K = nr_superpixels;
          N = image->width * image->height;
          int i,j,k,l;
          this->step = step;
          this->nc = nc;
          this->ns = step;

          // Clear previous data (if any), and re-initialize it.
          clear_data();
          init_data(image);

          /* Run for 10 iterations (as described by the algorithm). */
          for (i = 0; i < NR_ITERATIONS; ++i) {

            /* Reset distance values. */
            for (j = 0; j < image->width; ++j) {
              for (k = 0;k < image->height; ++k) {
                distances[j][k] = FLT_MAX;
              }
            }
            for (j = 0; j < (int) centers.size(); ++j) {
              /* Only comparing to pixels in a 2 x step by 2 x step region. */
              for (k = centers[j][3] - step; k < centers[j][3] + step; ++k) {
                for (l = centers[j][4] - step; l < centers[j][4] + step; ++l) {

                  if (k >= 0 && k < image->width && l >= 0 && l < image->height) {
                    CvScalar colour = cvGet2D(image, l, k);
                    double d = compute_dist(j, cvPoint(k,l), colour);

                    /* Update cluster allocation if the cluster minimizes the
                    distance. */
                    if (d < distances[k][l]) {
                      distances[k][l] = d;
                      clusters[k][l] = j;
                    }
                  }
                }
              }
            }

            /* Clear the center values. */
            for (j = 0; j < (int) centers.size(); ++j) {
              centers[j][0] = centers[j][1] = centers[j][2] = centers[j][3] = centers[j][4] = 0;
              center_counts[j] = 0;
            }

            /* Compute the new cluster centers. */
            for (j = 0; j < image->width; ++j) {
              for (k = 0; k < image->height; ++k) {
                int c_id = clusters[j][k];

                if (c_id != -1) {
                  CvScalar colour = cvGet2D(image, k, j);

                  centers[c_id][0] += colour.val[0];
                  centers[c_id][1] += colour.val[1];
                  centers[c_id][2] += colour.val[2];
                  centers[c_id][3] += j;
                  centers[c_id][4] += k;

                  center_counts[c_id] += 1;
                }
              }
            }

            /* Normalize the clusters. */
            for (j = 0; j < (int) centers.size(); ++j) {
              centers[j][0] /= center_counts[j];
              centers[j][1] /= center_counts[j];
              centers[j][2] /= center_counts[j];
              centers[j][3] /= center_counts[j];
              centers[j][4] /= center_counts[j];
            }
          }
        }
        /* Enforce connectivity for an image. */
        void create_connectivity(IplImage *image){
          int label = 0, adjlabel = 0;
          const int lims = (image->width * image->height) / ((int)centers.size());

          const int dx4[4] = {-1,  0,  1,  0};
          const int dy4[4] = { 0, -1,  0,  1};

          /* Initialize the new cluster matrix. */
          vec2di new_clusters;
          for (int i = 0; i < image->width; i++) {
            vector<int> nc;
            for (int j = 0; j < image->height; j++) {
              nc.push_back(-1);
            }
            new_clusters.push_back(nc);
          }

          for (int i = 0; i < image->width; i++) {
            for (int j = 0; j < image->height; j++) {
              if (new_clusters[i][j] == -1) {
                vector<CvPoint> elements;
                elements.push_back(cvPoint(i, j));

                /* Find an adjacent label, for possible use later. */
                for (int k = 0; k < 4; k++) {
                  int x = elements[0].x + dx4[k], y = elements[0].y + dy4[k];

                  if (x >= 0 && x < image->width && y >= 0 && y < image->height) {
                    if (new_clusters[x][y] >= 0) {
                      adjlabel = new_clusters[x][y];
                    }
                  }
                }

                int count = 1;
                for (int c = 0; c < count; c++) {
                  for (int k = 0; k < 4; k++) {
                    int x = elements[c].x + dx4[k], y = elements[c].y + dy4[k];

                    if (x >= 0 && x < image->width && y >= 0 && y < image->height) {
                      if (new_clusters[x][y] == -1 && clusters[i][j] == clusters[x][y]) {
                        elements.push_back(cvPoint(x, y));
                        new_clusters[x][y] = label;
                        count += 1;
                      }
                    }
                  }
                }

                /* Use the earlier found adjacent label if a segment size is
                smaller than a limit. */
                if (count <= lims >> 2) {
                  for (int c = 0; c < count; c++) {
                    new_clusters[elements[c].x][elements[c].y] = adjlabel;
                  }
                  label -= 1;
                }
                label += 1;
              }
            }
          }
        }

        /* Draw functions. Resp. displayal of the centers and the contours. */
        void display_center_grid(IplImage *image, CvScalar colour){
          for (int i = 0; i < (int) centers.size(); i++) {
            cvCircle(image, cvPoint(centers[i][3], centers[i][4]), 2, colour, 2);
          }
        }
        void display_contours(IplImage *image, CvScalar colour){
          const int dx8[8] = {-1, -1,  0,  1, 1, 1, 0, -1};
          const int dy8[8] = { 0, -1, -1, -1, 0, 1, 1,  1};

          /* Initialize the contour vector and the matrix detailing whether a pixel
          * is already taken to be a contour. */
          vector<CvPoint> contours;
          vec2db istaken;
          for (int i = 0; i < image->width; i++) {
            vector<bool> nb;
            for (int j = 0; j < image->height; j++) {
              nb.push_back(false);
            }
            istaken.push_back(nb);
          }

          /* Go through all the pixels. */
          for (int i = 0; i < image->width; i++) {
            for (int j = 0; j < image->height; j++) {
              int nr_p = 0;

              /* Compare the pixel to its 8 neighbours. */
              for (int k = 0; k < 8; k++) {
                int x = i + dx8[k], y = j + dy8[k];

                if (x >= 0 && x < image->width && y >= 0 && y < image->height) {
                  if (istaken[x][y] == false && clusters[i][j] != clusters[x][y]) {
                    nr_p += 1;
                  }
                }
              }

              /* Add the pixel to the contour list if desired. */
              if (nr_p >= 2) {
                contours.push_back(cvPoint(i,j));
                istaken[i][j] = true;
              }
            }
          }

          /* Draw the contour pixels. */
          for (int i = 0; i < (int)contours.size(); i++) {
            cvSet2D(image, contours[i].y, contours[i].x, colour);
          }
        }
        void colour_with_cluster_means(IplImage *image){
          vector<CvScalar> colours(centers.size());

          /* Gather the colour values per cluster. */
          for (int i = 0; i < image->width; i++) {
            for (int j = 0; j < image->height; j++) {
              int index = clusters[i][j];
              CvScalar colour = cvGet2D(image, j, i);

              colours[index].val[0] += colour.val[0];
              colours[index].val[1] += colour.val[1];
              colours[index].val[2] += colour.val[2];
            }
          }

          /* Divide by the number of pixels per cluster to get the mean colour. */
          for (int i = 0; i < (int)colours.size(); i++) {
            colours[i].val[0] /= center_counts[i];
            colours[i].val[1] /= center_counts[i];
            colours[i].val[2] /= center_counts[i];
          }

          /* Fill in. */
          for (int i = 0; i < image->width; i++) {
            for (int j = 0; j < image->height; j++) {
              CvScalar ncolour = colours[clusters[i][j]];
              cvSet2D(image, j, i, ncolour);
            }
          }
        }
};

#endif
