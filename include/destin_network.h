// Author : Hu Yuhuang
// Date   : 2014-04-27

/*
 * This file contains a class for creating DeSTIN network.
 */

#ifndef DESTIN_NETWORK_H
#define DESTIN_NETWORK_H

#include "destina_sys_lib.h"
#include "destina_lib.h"

class DestinNetwork
{
public:
    // VARIABLES

    int nLayer; // number of layers.
    int * dictSize; // dictionary size for each dictionary.
    //vector<cv::Mat> dict; // container for dictionary.

    /* sparse autoencoder */

    vector<SparseAE::SA> dict;
    vector<Mat> feature;

public:
    DestinNetwork()
    {

    }

    DestinNetwork(int nlayer, int * DictSize)
    {
        initNetwork(nlayer, DictSize);
    }

    void initNetwork(int nlayer, int * DictSize)
    {
        nLayer=nlayer;
        dictSize=DictSize;
        initFeature(feature, nLayer);
    }

    /*
     * Function: this function initialize features for each layer.
     *           there are different methods available,
     *           currently, the simplest method is implemented.
     *
     * f: features are going to be initialized.
     *
     */
    void initFeature(vector<cv::Mat> & f, int nlayer)
    {
        for (int i=0; i<nlayer; i++)
        {
            int nNodes=0;
            calNodesInLayer(i, nNodes);
            cv::Mat temp;
            createInitNodesSimple(nNodes, i, temp);
            f.push_back(temp);
            temp.release();
        }
    }

    /*
     * Function: this function calculates number of nodes given
     *           a layer.
     *
     * INPUT
     * layer : given layer.
     *
     * OUTPUT
     * nnodes : number of nodes which belongs to such layer.
     */
    void calNodesInLayer(int layer, int & nnodes)
    {
        int exp=nLayer-2*layer+2;

        nnodes=(int)pow(2.0, (double)exp);
    }

    /*
     * Function: this function calculates the initialization of the nodes
     *           here the simplest method is implemented.
     *
     * INPUT
     * nnodes : number of nodes in such layer.
     * layer  : given layer.
     *
     * OUTPUT
     * initf  : initial setting for the layer, each column is a sample.
     */
    void createInitNodesSimple(int nnodes, int layer, cv::Mat & initf)
    {
        int ds=dictSize[layer];
        double p=1.0/(double)ds;

        initf=Mat::ones(ds, nnodes, CV_64FC1)*p;
    }

    /*
     * Function: this function organize representation from lower
     *           layer to upper layer, in principle, should be
     *           4 nodes to 1.
     *
     * INPUT
     * in : lower layer representation, samples in columns.
     *
     * OUTPUT
     * out : upper layer representation.
     */
    void organizeRepresentation(cv::Mat in, cv::Mat & out)
    {
        int noRegionNodes=4;
        int noNodes=in.cols;
        int noNewNodes=noNodes/noRegionNodes;
        int sideOldLength=(int)sqrt((double)noNodes);
        int sideNewLength=(int)sqrt((double)noNewNodes);

        for (int i=0; i<sideNewLength; i++)
            for (int j=0; j<sideNewLength; j++)
            {
                cv::Mat temp=in.col((2*i-2)*sideOldLength+(2*j-1));
                temp.push_back(in.col((2*i-2)*sideOldLength+(2*j)));
                temp.push_back(in.col((2*i-1)*sideOldLength+(2*j-1)));
                temp.push_back(in.col((2*i-1)*sideOldLength+(2*j)));

                if (i*sideNewLength+j!=0) cv::hconcat(out, temp, out);
                else out=temp;

                temp.release();
            }
    }

    /*
     * Function: this function reorganize represenation from upper layer
     *           to lower layer, in principle, should be 1 node to 4 nodes.
     *
     * INPUT
     * in : upper layer representation, samples in columns.
     *
     * OUTPUT
     * out : lower layer representation.
     */
    void reorganizeRepresentation(cv::Mat in, cv::Mat & out)
    {
        int noRegionNodes=4;
        int noNewNodes=in.cols;
        int noNodes=noNodes*noRegionNodes;
        int sideOldLength=(int)sqrt((double)noNodes);
        int sideNewLength=(int)sqrt((double)noNewNodes);
        int repLength=in.rows/noRegionNodes;

        out=Mat::zeros(repLength, noNodes, CV_32F);
        for (int i=0; i<sideNewLength; i++)
            for (int j=0; j<sideNewLength; j++)
            {
                cv::Mat temp=in.col(i*sideNewLength+j);
                temp(Rect(0,0,1,repLength)).copyTo(out.col((2*i-2)*sideOldLength+(2*j-1)));
                temp(Rect(repLength,0,1,repLength)).copyTo(out.col((2*i-2)*sideOldLength+(2*j)));
                temp(Rect(repLength*2,0,1,repLength)).copyTo(out.col((2*i-1)*sideOldLength+(2*j-1)));
                temp(Rect(repLength*3,0,1,repLength)).copyTo(out.col((2*i-1)*sideOldLength+(2*j)));

                temp.release();
            }
    }


    /*
     * Function: train a image
     *
     * INPUT
     * image : input image in patches form.
     */
    void train(cv::Mat image)
    {

    }

    /*
     * Function: train a single layer based on input to such layer
     *
     * INPUT
     * sig   : input for such layer
     * layer : layer indicator
     */
    void trainSingleLayer(cv::Mat sig, int layer)
    {
        /* this function hear demonstrate sparse autoencoder */

        cv::Mat trainX;
        sig.convertTo(trainX, CV_64FC1);

        SparseAE::batch = trainX.cols / 100;

        vector<Mat> Activations;
        //Mat tempX;
        //if(i == 0) trainX.copyTo(tempX); else Activations[Activations.size() - 1].copyTo(tempX);
        //SA tmpsa;
        SparseAE::trainSparseAutoencoder(dict[layer], trainX, dictSize[layer], 3e-3, 0.1, 3, 2e-2, 80000);
        /*
        Mat tmpacti = tmpsa.W1 * tempX + repeat(dictionary[layer].b1, 1, trainX.cols);
        tmpacti = sigmoid(tmpacti);
        Activations.push_back(tmpacti);

        SAA saa=getSparseAutoencoderActivation(HiddenLayers[0], trainX);
        */
    }

    /*
     * Function: update dictionary based on input samples.
     *
     * INPUT
     * samples : signals in column form.
     * layer   : which layer is training.
     */
    void clustering(cv::Mat samples, int layer)
    {

    }

    /*
     * Function: extract representation of a image.
     *
     * INPUT
     * image : input image.
     *
     * OUTPUT
     * rep : extract corresponding representation.
     */
    void extractRepresentation(cv::Mat image, cv::Mat & rep)
    {

    }

};

#endif // DESTIN_NETWORK_H
