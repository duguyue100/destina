// Author : Hu Yuhuang
// Date   : 2014-06-04

/*
 * This file contains a test of destin network
 */

#include "destina_sys_lib.h"
#include "destina_lib.h"

int main(void)
{
    Size ksize; ksize.height=32; ksize.width=32;
    cv::Mat im=imread("../../resources/lenna.png");
    cv::Mat imout, imSig;
    Mat trainX;

    PreProc::preProcImage(im, ksize, true, imout);
    ProcTool::splitImageToPatches(imout, imSig);

    int centroids[]={256, 128, 64, 32};
    DestinNetwork * network=new DestinNetwork(4, centroids);

    network->train(imSig);
    //network->trainSingleLayer(imSig, 0);

    SparseAE::SAA saa=SparseAE::getSparseAutoencoderActivation(network->dict[0], imSig);

    cv::Mat recon;
    ProcTool::reorganizePatchesToImage(saa.aOutput, recon);

    cv::imshow("test", recon);
    waitKey(0);

    return 0;
}
