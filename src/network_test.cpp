// Author : Hu Yuhuang
// Date   : 2014-06-04

/*
 * This file contains a test of destin network
 */

#include "destina_sys_lib.h"
#include "destina_lib.h"

int main(void)
{
    Size ksize; ksize.height=256; ksize.width=256;
    cv::Mat im=imread("../../resources/lenna.png");
    cv::Mat imout, imSig, imNormal, imWhite;

    ProcTool::preProcImage(im, ksize, true, imout);
    ProcTool::splitImageToPatches(imout, imSig);

    //ProcTool::contrastNormalization(imSig, 0.04, imNormal);
    ProcTool::whitening(imSig, 0.1, imWhite);

    double min, max;
    cv::minMaxLoc(imWhite, &min, &max);
    imWhite-=min;
    cv::minMaxLoc(imWhite, &min, &max);
    imWhite/=max;

//    cv::Mat reconWhite;
//    ProcTool::reorganizePatchesToImage(imWhite, reconWhite);
//    imshow("test3", reconWhite);
//    cv::waitKey(0);


    int centroids[]={128, 64, 32, 32};
    DestinNetwork * network=new DestinNetwork(4, centroids);


    for (int i=1; i<20; i++)
    {
        cout << "[TRAINING]" << i << endl;
        network->pretrain(imWhite);
    }


    /*
    for (int i=1; i<20; i++)
    {
        cout << "[TRAINING]" << i << endl;
        network->train(imSig);
    }
    */

    /*
    for (int i=0;i<4;i++)
    {
        cout << network->feature[i] << endl;
    }
    */

    //cout << network->feature[0] << endl;



    //SparseAE::SAA saa=SparseAE::getSparseAutoencoderActivation(network->dict[0], imSig);

    cv::Mat recon;

    network->getRecontruction(1, recon);
    //ProcTool::reorganizePatchesToImage(saa.aOutput, recon);

    cv::imshow("test1", imout);
    waitKey(0);
    cv::imshow("test", recon);
    waitKey(0);

    return 0;
}
