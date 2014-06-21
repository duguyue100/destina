// Author : Hu Yuhuang
// Date   : 2014-04-22

/*
 * This file contains a test for image ZCA whitening.
 */

#include "destina_sys_lib.h"
#include "destina_lib.h"

int main(void)
{
    Size ksize; ksize.height=512; ksize.width=512;
    cv::Mat im=imread("../../resources/lenna.png");
    cv::Mat imout;
    cv::Mat imWhite;

    ProcTool::preProcImage(im, ksize, true, imout);
    ProcTool::whiteningImage(imout, 0.1, imWhite);

    imshow("Whitening Test", imWhite);
    waitKey(0);

    return 0;
}
