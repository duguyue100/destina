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

    Mat C = (Mat_<double>(3,3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);

    cout << C.reshape(0, 9) << endl;

    Mat mean;
    Mat stddev;
    cv::meanStdDev(C, mean, stddev);

    cout << mean << endl;
    cout << stddev << endl;
    Mat A=Mat::zeros(2,3,CV_64FC1)+1;
    Mat B=Mat::ones(2,3, CV_64FC1)*2;
    cv::divide(A.row(0), B.row(0), A.row(0));
    cout << A << endl;

    cv::imshow("test", images[1]);
    cv::waitKey(0);

    return 0;
}
