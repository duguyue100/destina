// Author : Hu Yuhuang
// Date   : 2014-06-21

/*
 * This file contains a test for CIFAR-10
 */

#include "destina_sys_lib.h"
#include "destina_lib.h"

int main(void)
{
    int centroids[]={256, 128, 64, 32};
    DestinNetwork * network=new DestinNetwork(4, centroids);

    vector<cv::Mat> images;
    vector<cv::Mat> imSignals;
    cv::Mat label = Mat::zeros(1, 10000, CV_64FC1);

    string filename="../../cifar-10-batches-bin/data_batch_1.bin";
    ProcTool::readCIFARBatch(filename, 200, images, label);
    ProcTool::processCIFARBatch(images, imSignals);

    cout << "[MESSAGE] Images processed" << endl;

    int split=100;
    int noIteration=8;
    cv::Mat preTrainSignal=imSignals[0];

    for (int i=1; i<split; i++)
        cv::hconcat(preTrainSignal, imSignals[i], preTrainSignal);

    cout << "[MESSAGE][PRETRAINING] Pretraining signal constructed" << endl;

    // train batch

    for (int i=1; i<=1; i++)
    {
        network->pretrain(preTrainSignal);
        cout << "[MESSAGE][PRETRAINING] " << i << "th iteration completed" << "\r";
        cout.flush();
    }

    cout << endl << "[MESSAGE][PRETRAINING] Pretraining completed" << endl;

    // train each sample
    for (int i=split; i<imSignals.size(); i++)
    {
        for (int j=0; i<noIteration; j++)
        {
            network->train(imSignals[i]);
        }
        cout << "[MESSAGE][TRAINING] " << i << "th image completed" << "\r";
        cout.flush();
    }

    cout << endl << "[MESSAGE][TRAINING] Training completed" << endl;

    // extracting feature
    cout << "[MESSAGE][TESTING] Prepare testing" << endl;


    vector<cv::Mat> observedF;
    for (int i=0; i<imSignals.size(); i++)
    {
        vector<cv::Mat> f;
        network->initFeature(f, 4);
        for (int j=0; j<noIteration; j++)
        {
            network->observe(imSignals[i], f);
        }

        // extract feature;
        observedF.push_back(f[3]);

        cout << "[MESSAGE][TESTING] Feature of " << i << "th image extracted" << "\r";
        cout.flush();
    }

    cout << endl << "[MESSAGE] Feature extracted" << endl;

    cout << observedF.size() << endl;
    cout << observedF[0].rows << endl;
    cout << observedF[0].cols << endl;

    return 0;
}
