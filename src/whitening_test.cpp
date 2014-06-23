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
    cv::Mat imout;
    cv::Mat imWhite;
    cv::Mat imNormal;

    ProcTool::preProcImage(im, ksize, true, imout);

    ProcTool::contrastNormalizedImage(imout, 0.04, imNormal);
    ProcTool::whiteningImage(imNormal, 0.1, imWhite);

    imshow("Whitening Test", imWhite);
    cout << imWhite << endl;
    waitKey(0);

    return 0;
}
