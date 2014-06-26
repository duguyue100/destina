// Author : Hu Yuhuang
// Date   : 2014-06-21

/*
 * This file contains a test for CIFAR-10
 */

#include "destina_sys_lib.h"
#include "destina_lib.h"

int main(void)
{
    Size ksize; ksize.height=32; ksize.width=32;
    vector<Mat> images;
    vector<Mat> imSignals;
    Mat label = Mat::zeros(1, 10000, CV_64FC1);

    string filename="../../cifar-10-batches-bin/data_batch_1.bin";
    ProcTool::readCIFARBatch(filename, 10, images, label);
    ProcTool::processCIFARBatch(images, imSignals);

    cv::Mat imout, imSig, imWhite;
    ProcTool::preProcImage(images[1], ksize, true, imout);
    ProcTool::splitImageToPatches(imout, imSig);

    //ProcTool::contrastNormalization(imSig, 0.04, imNormal);
    ProcTool::whitening(imSig, 0.1, imWhite);

    double min, max;
    cv::minMaxLoc(imWhite, &min, &max);
    imWhite-=min;
    cv::minMaxLoc(imWhite, &min, &max);
    imWhite/=max;

    cv::Mat recon;

    ProcTool::reorganizePatchesToImage(imSignals[1], recon);

    cv::imshow("test", recon);
    cv::waitKey(0);

    return 0;
}
