// Author: Hu Yuhuang
// Date  : 2014-04-25

/*
 * This file contains a test for information filter based sparse coding.
 * Basically, this file computes sparse represenation and then
 * restores signal by using learned represenation. A comparision between
 * two images are performed also.
 */

#include "destina_sys_lib.h"
#include "destina_lib.h"

const double PI=atan(1)*4;

int main(void)
{
    Size imSize; imSize.height=64; imSize.width=64;
    cv::Mat im=imread("../../resources/lenna.png"); // original image
    cv::Mat imout, imWhite; // variable for whitening
    cv::Mat imageSig, restoredImageSig; // image signal
    cv::Mat restoredImage; // restored image
    cv::Mat imageRep; // image signal representation

    // pre process image and whitening
    PreProc::preProcImage(im, imSize, true, imout);
    PreProc::whiteningImage(imout, 0.1, imWhite); // imWhite in float
    ProcTool::splitImageToPatches(imWhite, imageSig);

    // generate Gabor wavelet dictionary
    Size gwSize; gwSize.height=4; gwSize.width=4;
    int M=4, N=64;
    cv::Mat GWD;
    PreProc::generateGaborWavelet(M, N, gwSize, PI, GWD);

    // compute represenation
    IFSC::computeSignalRepresentation(imageSig.col(3), GWD, 10, imageRep);

    cout << imageSig.col(2) << endl;
    cout << GWD*imageRep << endl;
    IFSC::computeRepresentation(imageSig, GWD, imageRep);

    // restore signal
    IFSC::restoreSignal(imageRep, GWD, restoredImageSig);
    ProcTool::reorganizePatchesToImage(restoredImageSig, restoredImage);

    // display result
    imshow("Whitening Test", imWhite);
    waitKey(0);
    imshow("Restored Image", restoredImage);

    waitKey(0);
    return 0;
}
