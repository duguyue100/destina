// Author : Hu Yuhuang
// Date   : 2014-04-22

/*
 * This file contains a test for generating Gabor wavelet dictionary.
 */

#include "destina_sys_lib.h"
#include "destina_lib.h"

const double PI=atan(1)*4;

int main(void)
{
    Size ksize; ksize.height=32; ksize.width=32;
    int M=20;
    int N=36;

    cv::Mat GWD;
    cv::Mat GWDOut;
    ProcTool::generateGaborWavelet(M, N, ksize, PI, GWD);
    ProcTool::reorganizeGaborDictionary(M, N, ksize, GWD, GWDOut);

    imshow("Gabor Test", GWDOut.t());

    waitKey(0);

    return 0;
}
