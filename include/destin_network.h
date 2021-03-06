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
        for (int i=0; i<nLayer; i++)
        {
            SparseAE::SA temp;
            // init dictionary
            if (i==0)
                SparseAE::weightRandomInit(temp, 16 ,dictSize[i], feature[i].cols, 0.12);
            else SparseAE::weightRandomInit(temp, feature[i-1].rows*4 ,dictSize[i], feature[i].cols, 0.12);
            dict.push_back(temp);
        }
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
                cv::Mat temp=in.col((2*i)*sideOldLength+(2*j));
                temp.push_back(in.col((2*i)*sideOldLength+(2*j+1)));
                temp.push_back(in.col((2*i+1)*sideOldLength+(2*j)));
                temp.push_back(in.col((2*i+1)*sideOldLength+(2*j+1)));

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
        int noNodes=noNewNodes*noRegionNodes;
        int sideOldLength=(int)sqrt((double)noNodes);
        int sideNewLength=(int)sqrt((double)noNewNodes);
        int repLength=in.rows/noRegionNodes;

        out=Mat::zeros(repLength, noNodes, CV_64FC1);
        for (int i=0; i<sideNewLength; i++)
            for (int j=0; j<sideNewLength; j++)
            {
                cv::Mat temp=in.col(i*sideNewLength+j);
                temp(Rect(0,0,1,repLength)).copyTo(out.col((2*i)*sideOldLength+(2*j)));
                temp(Rect(0,repLength,1,repLength)).copyTo(out.col((2*i)*sideOldLength+(2*j+1)));
                temp(Rect(0,repLength*2,1,repLength)).copyTo(out.col((2*i+1)*sideOldLength+(2*j)));
                temp(Rect(0,repLength*3,1,repLength)).copyTo(out.col((2*i+1)*sideOldLength+(2*j+1)));

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
        for (int i=nLayer-1; i>=0; i--)
        {
            // construct input
            cv::Mat inputFeature;
            if (i!=0)
            {
                organizeRepresentation(feature[i-1], inputFeature);
            }
            else
            {
                inputFeature=image;
            }

            // training

            trainSingleLayer(inputFeature, i);

        }
    }

    /*
     * Function: this function performs a pre-training in bottom
     *           up fashion.
     *
     * INPUT
     * image : input image in patches form.
     */
    void pretrain(cv::Mat image)
    {
        for (int i=0; i<nLayer;i++)
        {
            // construct input
            cv::Mat inputFeature;
            if (i!=0)
            {
                organizeRepresentation(feature[i-1], inputFeature);
            }
            else
            {
                inputFeature=image;
            }

            // training

            trainSingleLayer(inputFeature, i);
        }
    }

    /*
     * Function: train a single layer based on input to such layer
     *
     * INPUT
     * sig     : input for such layer
     * layer   : layer indicator
     * noBatch : number of batches
     */
    void trainSingleLayer(cv::Mat sig, int layer, int noBatch)
    {
        /* this function hear demonstrate sparse autoencoder */

        SparseAE::batch = sig.cols / noBatch;

        // training given layer
        SparseAE::trainSparseAutoencoder(dict[layer], sig, dictSize[layer], 3e-3, 0.1, 3, 2e-2, 80000);

        // update features
        feature[layer] = dict[layer].W1 * sig + repeat(dict[layer].b1, 1, sig.cols);
        feature[layer] = SparseAE::sigmoid(feature[layer]);
    }

    /*
     * Function: train a single layer based on input to such layer
     *
     * INPUT
     * sig     : input for such layer
     * layer   : layer indicator
     */
    void trainSingleLayer(Mat sig, int layer)
    {
        trainSingleLayer(sig, layer, 4);
    }

    /*
     * Function: this function is for caucluating feature from an image
     *           for a trained network.
     *
     * INPUT
     * image : image in patches form
     *
     * OUTPUT
     * f : output feature array, this variable should be initialized
     *     before feeding.
     */
    void observe(cv::Mat image, vector<cv::Mat> & f)
    {
        cv::Mat inF;
        for (int i=nLayer-1; i>=0; i--)
        {
            if (i!=0)
            {
                organizeRepresentation(f[i-1], inF);
                f[i] = dict[i].W1 * inF + repeat(dict[i].b1, 1, inF.cols);
                f[i] = SparseAE::sigmoid(f[i]);
            }
            else
            {
                f[i] = dict[i].W1 * image + repeat(dict[i].b1, 1, image.cols);
                f[i] = SparseAE::sigmoid(f[i]);
            }
        }

        inF.release();
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


    /** DRAWING FUNCTIONS **/

    /*
     * Function: get a centroid image from specific layer.
     *
     * INPUT
     * layer      : the specific layer.
     * centroidID : this is the ID of a specific centroid, the ID
     *              is identified according to dictionary's index.
     *
     * OUTPUT
     * centroid : selected centroid's image.
     */
    void getCentroid(int layer, int centroidID, cv::Mat & centroid)
    {
        // get centroid information

        // gradually reconstruct information

        // reorganize centroid
    }

    /*
     * Function: get a layer's centroids image.
     *
     * INPUT
     * layer   : the specific layer.
     * noCols  : how many centroids in a row.
     * spacing : black spacing between each centroid's image.
     *
     * OUTPUT
     * layerCentroid : selected layer's centroids image.
     */
    void getLayerCentroid(int layer, int noCols, int spacing, Mat &layerCentroid)
    {
        int noRows=dictSize[layer]/noCols;

        // reorganize centroid images
    }

    /*
     * Function: get a layer's centroids image with initial settings
     *
     * INPUT
     * layer : the specific layer.
     *
     * OUTPUT
     * layerCentroid : selected layer's centroids image.
     */
    void getLayerCentroid(int layer, cv::Mat & layerCentroid)
    {
        int noCols=(int)sqrt(dictSize[layer]);
        getLayerCentroid(layer, noCols, 2, layerCentroid);
    }

    /*
     * Function: get image reconstruction from different layer
     *
     * INPUT
     * layer : the specific layer
     *
     * OUTPUT
     * reconstruction : the reconstruction image
     *
     */
    void getRecontruction(int layer, cv::Mat & reconstruction)
    {
        // get feature from certain layer

        cv::Mat f=feature[layer];

        // gradually reconstruct the image to bottom layer

        cv::Mat temp;
        for (int i=layer; i>=0; i--)
        {
            temp = dict[i].W2 * f + repeat(dict[i].b2, 1, f.cols);
            temp = SparseAE::sigmoid(temp);

            if (i!=0) reorganizeRepresentation(temp, f);
            else f=temp;
        }

        ProcTool::reorganizePatchesToImage(f, reconstruction);
        f.release();
    }

    /** Load and Save Function **/

    /*
     * Function: this function load network from file
     *
     * INPUT
     * filename : input file name
     */
    void load(string filename)
    {

    }

    /*
     * Function: this function save network to file
     *
     * INPUT
     * filename : export file name
     *
     */
    void save(string filename)
    {

    }

};

#endif // DESTIN_NETWORK_H
