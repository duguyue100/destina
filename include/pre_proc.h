// Author : Hu Yuhuang
// Date   : 2014-04-25

/*
 * This file contains a set of preprocessing tools on image and signal.
 */

#ifndef PRE_PROC_H
#define PRE_PROC_H

#include "destina_sys_lib.h"

using namespace ProcTool;

namespace PreProc
{

/*
 * Function: this function generate gabor wavelet dictionary
 *
 * INPUT
 * M : Number of scales.
 * N : Number of rotations.
 * X : Rows of output gabor filter.
 * Y : Cols of output gabor filter.
 * K : Constant, K=pi or K~2.5.
 *
 * OUTPUT
 * GWD : generated gabor wavelet dictionary.
 *       each column of GWD is a item of gabor wavelet.
 */
void generateGaborWavelet(int M, int N, Size S, double K, cv::Mat & GWD)
{
    const double pi=atan(1)*4;

    for (int i=1; i<=M; i++)
        for (int j=1; j<=N; j++)
        {
            double sigma=K*(double)i/sqrt(2);
            double lambda=sigma/0.56;
            cv::Mat GW=getGaborKernel(S, sigma, j*pi/(double)N, lambda, 0.5, CV_64FC1);

            if (S.height%2==0) GW.pop_back();

            if (S.width%2==0)
            {
                GW=GW.t();
                GW.pop_back();
                GW=GW.t();
            }

            GWD.push_back(GW.reshape(0, 1));

            GW.release();
        }
    GWD=GWD.t();

    GWD.convertTo(GWD, CV_64FC1);
}


/*
 * Function: reorganize Gabor wavelet dictionary to visualize
 *
 * INPUT
 * M  : Number of scales.
 * N  : Number of rotations.
 * S  : Size of wavelet.
 * in : Gabor wavelet dictionary.
 *
 * OUTPUT
 * out : Reorganized dictionary.
 */
void reorganizeGaborDictionary(int M, int N, Size S, cv::Mat in, cv::Mat & out)
{
    cv::Mat In=in.t();

    for (int i=0; i<M; i++)
    {
        cv::Mat temp;
        for (int j=0; j<N; j++)
            temp.push_back(In.row(i*N+j).reshape(0, S.height));

        if (i==0) out=temp;
        else cv::hconcat(out, temp, out);

        temp.release();
    }

    In.release();
}


/*
 * Function: preProcImage to valid size and format to appropriate size
 *
 * INPUT
 * in   : input image
 * S    : output size
 * gray : if convert to gray form
 *
 * OUTPUT
 * out : output image in double form (rescaled to (0,1))
 */
void preProcImage(cv::Mat in, Size S, bool gray, cv::Mat & out)
{
    cv::Mat temp;
    if (gray==true)
        cv::cvtColor(in, temp, CV_BGR2GRAY);

    cv::resize(temp, temp, S);

    // change to double format

    temp.convertTo(out, CV_64FC1);
    out/=255.0;

    temp.release();
}

/*
 * Function: this function receives a original grayscaled image and
 *           return ZCA whitened image
 *
 * INPUT
 * in      : original image
 * epsilon : constant (0.1)
 *
 * OUTPUT
 * out : ZCA whitened image
 *
 */
void whiteningImage(cv::Mat in, double epsilon, cv::Mat & out)
{
    cv::Mat temp;
    // split image to batches

    splitImageToPatches(in, temp);

    // rescale mean to zero

    Mat mean=temp.col(0);
    for (int i=1; i<temp.cols; i++) mean+=temp.col(i);
    mean/=(double)temp.cols;
    cv::repeat(mean, 1, temp.cols, mean);
    temp-=mean;

    // calculate necessary eigen values and eigen vectors

    Mat sigma=temp*temp.t();
    sigma/=temp.cols;

    PCA pca(sigma, Mat(), CV_PCA_DATA_AS_COL);
    // every row is a eigen vector.

    cv::Mat eigenValues=pca.eigenvalues;
    cv::Mat eigenVectors=pca.eigenvectors.t();

    cv::Mat eigenSqrt;
    cv::sqrt(Mat::diag(eigenValues+epsilon), eigenSqrt);
    eigenSqrt=1/eigenSqrt;

    Mat whitenedImage=eigenVectors*eigenSqrt*eigenVectors.t()*temp;

    // restore image

    reorganizePatchesToImage(whitenedImage, out);
}

}

#endif // PRE_PROC_H
