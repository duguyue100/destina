#include "destina_sys_lib.h"
#include "destina_lib.h"

using namespace SparseAE;

int main(int argc, char** argv)
{
    long start, end;
    start = clock();

    Size ksize; ksize.height=512; ksize.width=512;
    cv::Mat im=imread("../../resources/lenna.png");
    cv::Mat imout, imSig;
    Mat trainX;

    PreProc::preProcImage(im, ksize, true, imout);
    ProcTool::splitImageToPatches(imout, imSig);

    //imSig.convertTo(trainX, CV_64FC1);
    trainX=imSig;

    batch = trainX.cols / 100;

    vector<SA> HiddenLayers;
    vector<Mat> Activations;
    int SparseAutoencoderLayers=1;
    for(int i=0; i<SparseAutoencoderLayers; i++){
        Mat tempX;
        if(i == 0) trainX.copyTo(tempX); else Activations[Activations.size() - 1].copyTo(tempX);
        SA tmpsa;
        trainSparseAutoencoder(tmpsa, tempX, 600, 3e-3, 0.1, 3, 2e-2, 80000);
        Mat tmpacti = tmpsa.W1 * tempX + repeat(tmpsa.b1, 1, tempX.cols);
        tmpacti = sigmoid(tmpacti);
        HiddenLayers.push_back(tmpsa);
        Activations.push_back(tmpacti);
    }

    SAA saa=getSparseAutoencoderActivation(HiddenLayers[0], trainX);

    cout << saa.aOutput.cols << endl;
    cout << saa.aOutput.rows << endl;

    cv::Mat recon;
    ProcTool::reorganizePatchesToImage(saa.aOutput, recon);


    end = clock();
    cout<<"Totally used time: "<<((double)(end - start)) / CLOCKS_PER_SEC<<" second"<<endl;

    imshow("Sparse Autoencoder Reconstruction", recon);
    waitKey(0);

    return 0;
}
