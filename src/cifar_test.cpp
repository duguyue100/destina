// Author : Hu Yuhuang
// Date   : 2014-06-21

/*
 * This file contains a test for CIFAR-10
 */

#include "destina_sys_lib.h"
#include "destina_lib.h"

int main(void)
{
    vector<Mat> images;
    vector<Mat> imSignals;
    Mat label = Mat::zeros(1, 10000, CV_64FC1);

    string filename="../../cifar-10-batches-bin/data_batch_1.bin";
    ProcTool::readCIFARBatch(filename, 10, images, label);
    ProcTool::processCIFARBatch(images, imSignals);

    cv::imshow("test", images[1]);
    cv::waitKey(0);

    return 0;
}
