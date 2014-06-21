// Author : Hu Yuhuang
// Date   : 2014-04-22

/*
 * this file contains some useful tools on image and signal
 */

#ifndef PROC_TOOLS_H
#define PROC_TOOLS_H

#include "destina_sys_lib.h"

namespace ProcTool
{

/*
 * Function: split one image to 16*2^(2n-4) patches.
 *
 * INPUT
 * in        : input image in 2^n*2^n form.
 * patchSize : size of each patch.
 *
 * OUTPUT
 * out : output image, each column is a patch
 *       noted that after this process, the Mat turns to float.
 */
void splitImageToPatches(cv::Mat in, int patchSize, cv::Mat & out)
{
    // To-do: check channel of input image
    //        should be CV_64FC1

    int noInRows=in.rows/patchSize;
    int noInCols=in.cols/patchSize;

    for (int i=0; i<noInRows; i++)
        for (int j=0; j<noInCols; j++)
        {
            cv::Mat temp=in(cv::Rect(i*patchSize, j*patchSize, patchSize, patchSize)).clone();
            temp=Mat_<double>(temp);
            out.push_back(temp.reshape(0, 1));
        }

    out=out.t();
}

/*
 * Function: for 4*4 patches only.
 */
void splitImageToPatches(cv::Mat in, cv::Mat & out)
{
    splitImageToPatches(in, 4, out);
}


/*
 * Function: reorganize patches to image
 *
 * INPUT
 * in        : input image, each cloumn is a patch.
 * patchSize : size of each patch.
 *
 * OUTPUT
 * out : output image.
 */
void reorganizePatchesToImage(cv::Mat in, int patchSize, cv::Mat & out)
{
    int noInRows=(int)sqrt(in.cols);
    int noInCols=noInRows;

    for (int i=0; i<noInRows; i++)
    {
        cv::Mat temp;
        for (int j=0; j<noInCols; j++)
        {
            cv::Mat tCol=in.col(i*noInCols+j);
            tCol=tCol.t();
            tCol=tCol.reshape(0, patchSize);
            if (j==0) temp=tCol;
            else cv::hconcat(temp, tCol, temp);
        }

        out.push_back(temp);

        temp.release();
    }

    out=out.t();
}

void reorganizePatchesToImage(cv::Mat in, cv::Mat & out)
{
    reorganizePatchesToImage(in, 4, out);
}

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

/*
 * Function: read batch of CIFAR images.
 *
 * INPUT
 * filename         : CIFAR bin file.
 * number_of_images : How many images want to load.
 *
 * OUTPUT
 * vec   : a vector contains CIFAR image
 * label : image data format.
 */
void readCIFARBatch(string filename, int number_of_images, vector<Mat> &vec, Mat &label)
{
    std::ifstream file(filename.c_str(), ios::binary);
    if (file.is_open() && number_of_images!=0)
    {
        int n_rows = 32;
        int n_cols = 32;
        for(int i = 0; i < number_of_images; ++i)
        {
            unsigned char tplabel = 0;
            file.read((char*) &tplabel, sizeof(tplabel));
            vector<Mat> channels;
            Mat fin_img = Mat::zeros(n_rows, n_cols, CV_8UC3);
            for(int ch = 0; ch < 3; ++ch){
                Mat tp = Mat::zeros(n_rows, n_cols, CV_8UC1);
                for(int r = 0; r < n_rows; ++r){
                    for(int c = 0; c < n_cols; ++c){
                        unsigned char temp = 0;
                        file.read((char*) &temp, sizeof(temp));
                        tp.at<uchar>(r, c) = (int) temp;
                    }
                }
                channels.push_back(tp);
            }
            merge(channels, fin_img);
            vec.push_back(fin_img);
            label.at<double>(0, i) = (double)tplabel;
        }
    }
}

/*
 * Function: process CIFAR image batch to signal.
 *
 * INPUT
 * batch : batch of CIFAR images.
 *
 * OUTPUT
 * processedBatch : batch of processed CIFAR image signals.
 */
void processCIFARBatch(vector<cv::Mat> batch, vector<cv::Mat> & processedBatch)
{
    Size ksize; ksize.height=32; ksize.width=32;

    cv::Mat temp;
    for (int i=0; i<batch.size(); i++)
    {
        preProcImage(batch[i], ksize, true, temp);
        splitImageToPatches(temp, temp);
        processedBatch.push_back(temp);
    }

    temp.release();
}

} // end of identifier

#endif // PROC_TOOLS_H
