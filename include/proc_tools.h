// Author : Hu Yuhuang
// Date   : 2014-04-22

/*
 * this file contains some useful tools on image and signal
 */

#ifndef PROC_TOOLS_H
#define PROC_TOOLS_H

#include "destina_sys_lib.h"

namespace ProcTool
{

/*
 * Function: split one image to 16*2^(2n-4) patches.
 *
 * INPUT
 * in        : input image in 2^n*2^n form.
 * patchSize : size of each patch.
 *
 * OUTPUT
 * out : output image, each column is a patch
 *       noted that after this process, the Mat turns to float.
 */
void splitImageToPatches(cv::Mat in, int patchSize, cv::Mat & out)
{
    // To-do: check channel of input image
    //        should be CV_64FC1

    int noInRows=in.rows/patchSize;
    int noInCols=in.cols/patchSize;

    for (int i=0; i<noInRows; i++)
        for (int j=0; j<noInCols; j++)
        {
            cv::Mat temp=in(cv::Rect(i*patchSize, j*patchSize, patchSize, patchSize)).clone();
            temp=Mat_<double>(temp);
            out.push_back(temp.reshape(0, 1));
        }

    out=out.t();
}

/*
 * Function: for 4*4 patches only.
 */
void splitImageToPatches(cv::Mat in, cv::Mat & out)
{
    splitImageToPatches(in, 4, out);
}


/*
 * Function: reorganize patches to image
 *
 * INPUT
 * in        : input image, each cloumn is a patch.
 * patchSize : size of each patch.
 *
 * OUTPUT
 * out : output image.
 */
void reorganizePatchesToImage(cv::Mat in, int patchSize, cv::Mat & out)
{
    int noInRows=(int)sqrt(in.cols);
    int noInCols=noInRows;

    for (int i=0; i<noInRows; i++)
    {
        cv::Mat temp;
        for (int j=0; j<noInCols; j++)
        {
            cv::Mat tCol=in.col(i*noInCols+j);
            tCol=tCol.t();
            tCol=tCol.reshape(0, patchSize);
            if (j==0) temp=tCol;
            else cv::hconcat(temp, tCol, temp);
        }

        out.push_back(temp);

        temp.release();
    }

    out=out.t();
}

void reorganizePatchesToImage(cv::Mat in, cv::Mat & out)
{
    reorganizePatchesToImage(in, 4, out);
}

}

#endif // PROC_TOOLS_H
