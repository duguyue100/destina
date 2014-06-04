// Author : Hu Yuhuang
// Date   : 2014-04-25

/*
 * This file contains implementaion of information filter based
 * sparse coding.
 */

#ifndef IFSC_H
#define IFSC_H

#include "destina_sys_lib.h"

namespace IFSC
{

/*
 * Function: calculate covariance for matrix.
 *
 * INPUT
 * X : input matrix, each column is a sample.
 *
 * OUTPUT
 * covMat : output covariance matrix.
 */
void covariance(cv::Mat X, cv::Mat & covMat)
{
    float N=(float)X.rows;

    Mat mean=X.row(0);
    for (int i=1; i<X.rows; i++) mean+=X.row(i);
    mean/=N;

    cv::repeat(mean, X.rows, 1, mean);
    cv::Mat zX=X-mean;

    covMat=(1/(N-1))*zX*zX.t();
}

/*
 * Function: generate Gaussian Noise covariance matrix by given size.
 *
 * INPUT
 * noRow       : Number of rows.
 * noCol       : Number of columns.
 * sigma_gamma : variance of the noise.
 *
 * OUTPUT
 * covMat : noise covariance matrix.
 */
void generateNoiseCovariance(int noRow, int noCol, float sigma_gamma, cv::Mat & covMat)
{
    cv::Mat temp, tt;
    for (int i=0;i<=noCol;i++)
    {
        cv::Mat tp=Mat::zeros(noRow, 1, CV_32F);
        cv::randn(tp, 0, sqrt(sigma_gamma));
        if (i!=0) cv::hconcat(temp, tp, temp);
        else temp=tp;
    }
    //cv::calcCovarMatrix(temp.t(), covMat, tt, CV_COVAR_COLS, CV_32F);
    covariance(temp, covMat);

    temp.release();
    tt.release();
}

/*
 * Function: information filter based improved sparse coding for 1-D signal.
 *
 * INPUT
 * signal  : 1-D signal.
 * dict    : corresponding dictionary.
 * Q       : noise covariance matrix.
 * delta_w : noise variance.
 *
 * OUTPUT
 * SigmaNew : information matrix.
 * PsiNew   : information vector, representation z=Simga*Psi.
 */
void learnRepresentation(cv::Mat signal, cv::Mat dict, cv::Mat Q, float delta_w, cv::Mat Sigma, cv::Mat Psi, cv::Mat & SigmaNew, cv::Mat & PsiNew)
{
    int dictSize=dict.cols;

    // innovation
    cv::Mat SigmaTemp=Sigma+(1/delta_w)*dict.t()*dict;
    cv::Mat PsiTemp=Psi+(1/delta_w)*dict.t()*signal;

    // prediction
    cv::Mat SigmaTempInv=SigmaTemp+Q.inv();
    cv::Mat A=Mat::eye(dictSize, dictSize, CV_32F)-SigmaTemp*SigmaTempInv.inv();

    SigmaNew=A*SigmaTemp;
    PsiNew=A*PsiTemp;

    // release memory
    SigmaTemp.release();
    SigmaTempInv.release();
    PsiTemp.release();
    A.release();
}

/*
 * Function: compute sparse representation using information filter.
 *
 * INPUT
 * signal     : a 1-d signal.
 * dict       : a dictionary for the signal.
 * interation : how many iteration apply.
 *
 * OUTPUT
 * representation : the sparse representation for the signal.
 */
void computeSignalRepresentation(cv::Mat signal, cv::Mat dict, int iteration, cv::Mat & representation)
{
    cv::Mat Sigma=Mat::zeros(dict.cols, dict.cols, CV_32F);
    cv::Mat Psi=Mat::zeros(dict.cols, 1, CV_32F);
    cv::Mat Q;
    generateNoiseCovariance(dict.cols, iteration, 0.05, Q);

    for (int i=1; i<=iteration; i++)
    {
        //cout << i << endl;
        learnRepresentation(signal, dict, Q, 0.1, Sigma, Psi, Sigma, Psi);
    }

    representation=Sigma.inv()*Psi;
}


/*
 * Function: compute sparse represenation for the image
 *
 * INPUT
 * SIG  : image, each column is a image patch.
 * dict : dictionary.
 *
 * OUTPUT
 * REP : sparse representation for the image.
 */
void computeRepresentation(cv::Mat SIG, cv::Mat dict, cv::Mat & REP)
{
    int noSignal=SIG.cols;
    cv::Mat tempRep;

    for (int i=0; i<noSignal; i++)
    {
        cout << i << endl;
        computeSignalRepresentation(SIG.col(i), dict, 10, tempRep);

        if (i!=0) cv::hconcat(REP, tempRep, REP);
        else REP=tempRep;
    }

    tempRep.release();
}


/*
 * Function: restore signal based on dictionary and sparse represenation
 *
 * INPUT
 * REP  : Sparse represenation of the signal.
 * dict : The corresponding dictionary.
 *
 * OUTPUT
 * SIG : restored signal.
 */
void restoreSignal(cv::Mat REP, cv::Mat dict, cv::Mat & SIG)
{
    SIG=dict*REP;
}

}

#endif // IFSC_H
