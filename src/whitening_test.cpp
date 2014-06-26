// Author : Hu Yuhuang
// Date   : 2014-04-22

/*
 * This file contains a test for image ZCA whitening.
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

    cv::Mat recon;

    ProcTool::reorganizePatchesToImage(imWhite, recon);

    imshow("Whitening Test", recon);

    waitKey(0);

    return 0;
}
