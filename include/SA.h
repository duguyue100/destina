// MnistClassify.cpp
//
// Author: Eric Yuan
// Blog: http://eric-yuan.me
// You are FREE to use the following code for ANY purpose.
//
// Altered by:
// Hu Yuhuang
// 2014-06-02

/*
 * This file contains a set of preprocessing tools on image and signal.
 */

#ifndef SA_H
#define SA_H

#include "destina_sys_lib.h"

namespace SparseAE
{

#define IS_TEST 0
#define IS_TEST_SA 0
#define IS_TEST_SMR 0
#define IS_TEST_FT 0

#define ATD at<double>
#define elif else if

int batch;

typedef struct SparseAutoencoder{
    Mat W1;
    Mat W2;
    Mat b1;
    Mat b2;
    Mat W1grad;
    Mat W2grad;
    Mat b1grad;
    Mat b2grad;
    double cost;
}SA;

typedef struct SparseAutoencoderActivation{
    Mat aInput;
    Mat aHidden;
    Mat aOutput;
}SAA;

Mat concatenateMat(vector<Mat> &vec){

    int height = vec[0].rows;
    int width = vec[0].cols;
    Mat res = Mat::zeros(height * width, vec.size(), CV_64FC1);
    for(int i=0; i<vec.size(); i++){
        Mat img(height, width, CV_64FC1);

        vec[i].convertTo(img, CV_64FC1);
        // reshape(int cn, int rows=0), cn is num of channels.
        Mat ptmat = img.reshape(0, height * width);
        Rect roi = cv::Rect(i, 0, ptmat.cols, ptmat.rows);
        Mat subView = res(roi);
        ptmat.copyTo(subView);
    }
    divide(res, 255.0, res);
    return res;
}

int ReverseInt (int i){
    unsigned char ch1, ch2, ch3, ch4;
    ch1 = i & 255;
    ch2 = (i >> 8) & 255;
    ch3 = (i >> 16) & 255;
    ch4 = (i >> 24) & 255;
    return((int) ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

/*
 * Function: the function calculates sigmoid function of a matrix in element
 *           wise. The function normalizes values to (0,1).
 *           f(x)=1/(1+exp(-x))
 *
 * INPUT
 * M : the input matrix.
 *
 * OUTPUT
 * Mat : the output matrix.
 */
Mat sigmoid(Mat &M){
    Mat temp;
    exp(-M, temp);
    return 1.0 / (temp + 1.0);
}

/*
 * Function: the first derivative of sigmoid function. g(x)=f(x)(1-f(x)).
 *
 * INPUT
 * a : the input matrix.
 *
 * OUTPUT
 * Mat : the output matrix.
 */
Mat dsigmoid(Mat &a){
    Mat res = 1.0 - a;
    res = res.mul(a);
    return res;
}

/*
 * Function: this function initializes weights in an autoencoder.
 *
 * INPUT:
 * inputsize  : size of input layer.
 * hiddensize : size of hidden layer.
 * nsamples   : number of samples.
 * epilson    : constant.
 *
 * OUTPUT
 * sa : output initialized autoencoder
 */
void weightRandomInit(SA &sa, int inputsize, int hiddensize, int nsamples, double epsilon){

    double *pData;
    sa.W1 = Mat::ones(hiddensize, inputsize, CV_64FC1);
    for(int i=0; i<hiddensize; i++){
        pData = sa.W1.ptr<double>(i);
        for(int j=0; j<inputsize; j++){
            pData[j] = randu<double>();
        }
    }
    sa.W1 = sa.W1 * (2 * epsilon) - epsilon;

    sa.W2 = Mat::ones(inputsize, hiddensize, CV_64FC1);
    for(int i=0; i<inputsize; i++){
        pData = sa.W2.ptr<double>(i);
        for(int j=0; j<hiddensize; j++){
            pData[j] = randu<double>();
        }
    }
    sa.W2 = sa.W2 * (2 * epsilon) - epsilon;

    sa.b1 = Mat::ones(hiddensize, 1, CV_64FC1);
    for(int j=0; j<hiddensize; j++){
        sa.b1.ATD(j, 0) = randu<double>();
    }
    sa.b1 = sa.b1 * (2 * epsilon) - epsilon;

    sa.b2 = Mat::ones(inputsize, 1, CV_64FC1);
    for(int j=0; j<inputsize; j++){
        sa.b2.ATD(j, 0) = randu<double>();
    }
    sa.b2 = sa.b2 * (2 * epsilon) - epsilon;

    sa.W1grad = Mat::zeros(hiddensize, inputsize, CV_64FC1);
    sa.W2grad = Mat::zeros(inputsize, hiddensize, CV_64FC1);
    sa.b1grad = Mat::zeros(hiddensize, 1, CV_64FC1);
    sa.b2grad = Mat::zeros(inputsize, 1, CV_64FC1);
    sa.cost = 0.0;
}

/*
 * Function: this function get autoencoder activation from input data.
 *
 * INPUT:
 * sa   : an autoencoder object.
 * data : input data.
 *
 * OUTPUT
 * SAA : SAA object which stores activations.
 */
SAA getSparseAutoencoderActivation(SA &sa, Mat &data){
    SAA acti;
    data.copyTo(acti.aInput);
    acti.aHidden = sa.W1 * acti.aInput + repeat(sa.b1, 1, data.cols);
    acti.aHidden = sigmoid(acti.aHidden);
    acti.aOutput = sa.W2 * acti.aHidden + repeat(sa.b2, 1, data.cols);
    acti.aOutput = sigmoid(acti.aOutput);
    return acti;
}

/*
 * Function: this function calculates costs for given sparse autoencoder.
 *
 * INPUT
 * sa            : given autoencoder.
 * data          : input data.
 * lambda        : constant.
 * sparsityParam : sparsity parameter.
 * beta          : constant.
 *
 * OUTPUT
 * sa : the cost is updated in given autoencoder object.
 */
void sparseAutoencoderCost(SA &sa, Mat &data, double lambda, double sparsityParam, double beta){

    int nfeatures = data.rows;
    int nsamples = data.cols;
    SAA acti = getSparseAutoencoderActivation(sa, data);

    Mat errtp = acti.aOutput - data;
    pow(errtp, 2.0, errtp);
    errtp /= 2.0;
    double err = sum(errtp)[0] / nsamples;
    // now calculate pj which is the average activation of hidden units
    Mat pj;
    reduce(acti.aHidden, pj, 1, CV_REDUCE_SUM);
    pj /= nsamples;
    // the second part is weight decay part
    double err2 = sum(sa.W1)[0] + sum(sa.W2)[0];
    err2 *= (lambda / 2.0);
    // the third part of overall cost function is the sparsity part
    Mat err3;
    Mat temp;
    temp = sparsityParam / pj;
    log(temp, temp);
    temp *= sparsityParam;
    temp.copyTo(err3);
    temp = (1 - sparsityParam) / (1 - pj);
    log(temp, temp);
    temp *= (1 - sparsityParam);
    err3 += temp;
    sa.cost = err + err2 + sum(err3)[0] * beta;

    // following are for calculating the grad of weights.
    Mat delta3 = -(data - acti.aOutput);
    delta3 = delta3.mul(dsigmoid(acti.aOutput));
    Mat temp2 = -sparsityParam / pj + (1 - sparsityParam) / (1 - pj);
    temp2 *= beta;
    Mat delta2 = sa.W2.t() * delta3 + repeat(temp2, 1, nsamples);
    delta2 = delta2.mul(dsigmoid(acti.aHidden));
    Mat nablaW1 = delta2 * acti.aInput.t();
    Mat nablaW2 = delta3 * acti.aHidden.t();
    Mat nablab1, nablab2;
    delta3.copyTo(nablab2);
    delta2.copyTo(nablab1);
    sa.W1grad = nablaW1 / nsamples + lambda * sa.W1;
    sa.W2grad = nablaW2 / nsamples + lambda * sa.W2;
    reduce(nablab1, sa.b1grad, 1, CV_REDUCE_SUM);
    reduce(nablab2, sa.b2grad, 1, CV_REDUCE_SUM);
    sa.b1grad /= nsamples;
    sa.b2grad /= nsamples;
}

/*
 * Function: this function run gradient checking in numerical way.
 *           (do not enable while real task is running).
 *
 * INPUT
 * sa            : given autoencoder.
 * data          : input data.
 * lambda        : constant.
 * sparsityParam : sparsity parameter.
 * beta          : constant.
 *
 * OUTPUT
 * void : this function will print checking on terminal.
 */
void gradientChecking(SA &sa, Mat &data, double lambda, double sparsityParam, double beta){

    sparseAutoencoderCost(sa, data, lambda, sparsityParam, beta);
    Mat w1g(sa.W1grad);
    cout<<"test sparse autoencoder !!!!"<<endl;
    double epsilon = 1e-4;
    for(int i=0; i<sa.W1.rows; i++){
        for(int j=0; j<sa.W1.cols; j++){
            double memo = sa.W1.ATD(i, j);
            sa.W1.ATD(i, j) = memo + epsilon;
            sparseAutoencoderCost(sa, data, lambda, sparsityParam, beta);
            double value1 = sa.cost;
            sa.W1.ATD(i, j) = memo - epsilon;
            sparseAutoencoderCost(sa, data, lambda, sparsityParam, beta);
            double value2 = sa.cost;
            double tp = (value1 - value2) / (2 * epsilon);
            cout<<i<<", "<<j<<", "<<tp<<", "<<w1g.ATD(i, j)<<", "<<w1g.ATD(i, j) / tp<<endl;
            sa.W1.ATD(i, j) = memo;
        }
    }
}

/*
 * Function: this function control the main training for a given autoencoder.
 *
 * INPUT
 * sa            : given autoencoder (make sure already initialized).
 * data          : given training dataset.
 * hiddenSize    : hidden layer size.
 * lambda        : constant.
 * sparsityParam : sparsity parameter.
 * beta          : constant.
 * lrate         : constant.
 * MaxIter       : maximum number of iterations.
 *
 * OUTPUT
 * sa : trained autoencoder.
 */
void trainSparseAutoencoder(SA &sa, Mat &data, int hiddenSize, double lambda, double sparsityParam, double beta, double lrate, int MaxIter){

    int nfeatures = data.rows;
    int nsamples = data.cols;

    //weightRandomInit(sa, nfeatures, hiddenSize, nsamples, 0.12);

    if (IS_TEST_SA){
        gradientChecking(sa, data, lambda, sparsityParam, beta);
    }else{
        int converge = 0;
        double lastcost = 0.0;
        cout<<"Sparse Autoencoder Learning: "<<endl;
        while(converge < MaxIter){

            int randomNum = abs(((long)rand() + (long)rand()) % (data.cols - batch));
            Rect roi = Rect(randomNum, 0, batch, data.rows);

            Mat batchX;
            if (batch!=0) batchX = data(roi);
            else batchX = data;


            sparseAutoencoderCost(sa, batchX, lambda, sparsityParam, beta);
            cout<<"learning step: "<<converge<<", Cost function value = "<<sa.cost<<", randomNum = "<<randomNum<<endl;
            if(fabs((sa.cost - lastcost) ) <= 5e-5 && converge > 0) break;
            if(sa.cost <= 0.0) break;
            lastcost = sa.cost;
            sa.W1 -= lrate * sa.W1grad;
            sa.W2 -= lrate * sa.W2grad;
            sa.b1 -= lrate * sa.b1grad;
            sa.b2 -= lrate * sa.b2grad;
            ++ converge;
        }
    }
}

}

#endif // SA_H
