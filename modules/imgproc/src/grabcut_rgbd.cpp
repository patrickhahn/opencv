/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of Intel Corporation may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "precomp.hpp"
#include "gcgraph.hpp"
#include <limits>

using namespace cv;

/*
This is implementation of image segmentation algorithm GrabCut described in
"GrabCut — Interactive Foreground Extraction using Iterated Graph Cuts".
Carsten Rother, Vladimir Kolmogorov, Andrew Blake.
 */

/*
 GMM - Gaussian Mixture Model
*/
class GMM
{
public:
    static const int componentsCount = 5;

    GMM( Mat& _model );
    double operator()( const Vec4d color ) const;
    double operator()( int ci, const Vec4d color ) const;
    int whichComponent( const Vec4d color ) const;

    void initLearning();
    void addSample( int ci, const Vec4d color );
    void endLearning();

private:
    void calcInverseCovAndDeterm( int ci );
    Mat model;
    double* coefs;
    double* mean;
    double* cov;

    double inverseCovs[componentsCount][4][4];
    double covDeterms[componentsCount];

    double sums[componentsCount][4];
    double prods[componentsCount][4][4];
    int sampleCounts[componentsCount];
    int totalSampleCount;
};

GMM::GMM( Mat& _model )
{
    const int modelSize = 4/*mean*/ + 16/*covariance*/ + 1/*component weight*/;
    if( _model.empty() )
    {
        _model.create( 1, modelSize*componentsCount, CV_64FC1 );
        _model.setTo(Scalar(0));
    }
    else if( (_model.type() != CV_64FC1) || (_model.rows != 1) || (_model.cols != modelSize*componentsCount) )
        CV_Error( CV_StsBadArg, "_model must have CV_64FC1 type, rows == 1 and cols == 13*componentsCount" );

    model = _model;

    coefs = model.ptr<double>(0);
    mean = coefs + componentsCount;
    cov = mean + 4*componentsCount;

    for( int ci = 0; ci < componentsCount; ci++ )
        if( coefs[ci] > 0 )
             calcInverseCovAndDeterm( ci );
}

double GMM::operator()( const Vec4d color ) const
{
    double res = 0;
    for( int ci = 0; ci < componentsCount; ci++ )
        res += coefs[ci] * (*this)(ci, color );
    return res;
}

double GMM::operator()( int ci, const Vec4d color ) const
{
    double res = 0;
    if( coefs[ci] > 0 )
    {
        CV_Assert( covDeterms[ci] > std::numeric_limits<double>::epsilon() );
        Vec4d diff = color;
        double* m = mean + 4*ci;
        diff[0] -= m[0]; diff[1] -= m[1]; diff[2] -= m[2]; diff[3] -= m[3];
        double mult = diff[0]*(diff[0]*inverseCovs[ci][0][0] + diff[1]*inverseCovs[ci][1][0] + diff[2]*inverseCovs[ci][2][0] + diff[3]*inverseCovs[ci][3][0])
                   + diff[1]*(diff[0]*inverseCovs[ci][0][1] + diff[1]*inverseCovs[ci][1][1] + diff[2]*inverseCovs[ci][2][1] + diff[3]*inverseCovs[ci][3][1])
                   + diff[2]*(diff[0]*inverseCovs[ci][0][2] + diff[1]*inverseCovs[ci][1][2] + diff[2]*inverseCovs[ci][2][2] + diff[3]*inverseCovs[ci][3][2])
                   + diff[3]*(diff[0]*inverseCovs[ci][0][3] + diff[1]*inverseCovs[ci][1][3] + diff[2]*inverseCovs[ci][2][3] + diff[3]*inverseCovs[ci][3][3]);
        res = 1.0f/sqrt(covDeterms[ci]) * exp(-0.5f*mult);
    }
    return res;
}

int GMM::whichComponent( const Vec4d color ) const
{
    int k = 0;
    double max = 0;

    for( int ci = 0; ci < componentsCount; ci++ )
    {
        double p = (*this)( ci, color );
        if( p > max )
        {
            k = ci;
            max = p;
        }
    }
    return k;
}

void GMM::initLearning()
{
    for( int ci = 0; ci < componentsCount; ci++)
    {
        sums[ci][0] = sums[ci][1] = sums[ci][2] = sums[ci][3] = 0;
        prods[ci][0][0] = prods[ci][0][1] = prods[ci][0][2] = prods[ci][0][3] = 0;
        prods[ci][1][0] = prods[ci][1][1] = prods[ci][1][2] = prods[ci][1][3] = 0;
        prods[ci][2][0] = prods[ci][2][1] = prods[ci][2][2] = prods[ci][2][3] = 0;
        prods[ci][3][0] = prods[ci][3][1] = prods[ci][3][2] = prods[ci][3][3] = 0;
        sampleCounts[ci] = 0;
    }
    totalSampleCount = 0;
}

void GMM::addSample( int ci, const Vec4d color )
{
    sums[ci][0] += color[0]; sums[ci][1] += color[1]; sums[ci][2] += color[2]; sums[ci][3] += color[3];
    prods[ci][0][0] += color[0]*color[0]; prods[ci][0][1] += color[0]*color[1];
    prods[ci][0][2] += color[0]*color[2]; prods[ci][0][3] += color[0]*color[3];
    prods[ci][1][0] += color[1]*color[0]; prods[ci][1][1] += color[1]*color[1];
    prods[ci][1][2] += color[1]*color[2]; prods[ci][1][3] += color[1]*color[3];
    prods[ci][2][0] += color[2]*color[0]; prods[ci][2][1] += color[2]*color[1];
    prods[ci][2][2] += color[2]*color[2]; prods[ci][2][3] += color[2]*color[3];
    prods[ci][3][0] += color[3]*color[0]; prods[ci][3][1] += color[3]*color[1];
    prods[ci][3][2] += color[3]*color[2]; prods[ci][3][3] += color[3]*color[3];
    sampleCounts[ci]++;
    totalSampleCount++;
}

void GMM::endLearning()
{
    const double variance = 0.01;
    for( int ci = 0; ci < componentsCount; ci++ )
    {
        int n = sampleCounts[ci];
        if( n == 0 )
            coefs[ci] = 0;
        else
        {
            coefs[ci] = (double)n/totalSampleCount;

            double* m = mean + 4*ci;
            m[0] = sums[ci][0]/n; m[1] = sums[ci][1]/n; m[2] = sums[ci][2]/n; m[3] = sums[ci][3]/n;

            double* c = cov + 16*ci;
            c[0] = prods[ci][0][0]/n - m[0]*m[0]; c[1] = prods[ci][0][1]/n - m[0]*m[1];
            c[2] = prods[ci][0][2]/n - m[0]*m[2]; c[3] = prods[ci][0][3]/n - m[0]*m[3];
            c[4] = prods[ci][1][0]/n - m[1]*m[0]; c[5] = prods[ci][1][1]/n - m[1]*m[1];
            c[6] = prods[ci][1][2]/n - m[1]*m[2]; c[7] = prods[ci][1][3]/n - m[1]*m[3];
            c[8] = prods[ci][2][0]/n - m[2]*m[0]; c[9] = prods[ci][2][1]/n - m[2]*m[1];
            c[10] = prods[ci][2][2]/n - m[2]*m[2]; c[11] = prods[ci][2][3]/n - m[2]*m[3];
            c[12] = prods[ci][3][0]/n - m[3]*m[0]; c[13] = prods[ci][3][1]/n - m[3]*m[1];
            c[14] = prods[ci][3][2]/n - m[3]*m[2]; c[15] = prods[ci][3][3]/n - m[3]*m[3];

            double dtrm = c[0] * (c[5]*(c[10]*c[15]-c[11]*c[14]) - c[6]*(c[9]*c[15]-c[11]*c[13]) + c[7]*(c[9]*c[14]-c[10]*c[13]))
                          - c[1] * (c[4]*(c[10]*c[15]-c[11]*c[14]) - c[6]*(c[8]*c[15]-c[11]*c[12]) + c[7]*(c[8]*c[14]-c[10]*c[12]))
                          + c[2] * (c[4]*(c[9]*c[15]-c[11]*c[13]) - c[5]*(c[8]*c[15]-c[11]*c[12]) + c[7]*(c[8]*c[13]-c[9]*c[12]))
                          - c[3] * (c[4]*(c[9]*c[14]-c[10]*c[13]) - c[5]*(c[8]*c[14]-c[10]*c[12]) + c[6]*(c[8]*c[13]-c[9]*c[12]));
            if( dtrm <= std::numeric_limits<double>::epsilon() )
            {
                // Adds the white noise to avoid singular covariance matrix.
                c[0] += variance;
                c[5] += variance;
                c[10] += variance;
                c[15] += variance;
            }

            calcInverseCovAndDeterm(ci);
        }
    }
}

void GMM::calcInverseCovAndDeterm( int ci )
{
    if( coefs[ci] > 0 )
    {
        double *c = cov + 16*ci;
        double dtrm =
              covDeterms[ci] = c[0] * (c[5]*(c[10]*c[15]-c[11]*c[14]) - c[6]*(c[9]*c[15]-c[11]*c[13]) + c[7]*(c[9]*c[14]-c[10]*c[13]))
                            - c[1] * (c[4]*(c[10]*c[15]-c[11]*c[14]) - c[6]*(c[8]*c[15]-c[11]*c[12]) + c[7]*(c[8]*c[14]-c[10]*c[12]))
                            + c[2] * (c[4]*(c[9]*c[15]-c[11]*c[13]) - c[5]*(c[8]*c[15]-c[11]*c[12]) + c[7]*(c[8]*c[13]-c[9]*c[12]))
                            - c[3] * (c[4]*(c[9]*c[14]-c[10]*c[13]) - c[5]*(c[8]*c[14]-c[10]*c[12]) + c[6]*(c[8]*c[13]-c[9]*c[12]));

        CV_Assert( dtrm > std::numeric_limits<double>::epsilon() );

        inverseCovs[ci][0][0] = c[5]*(c[10]*c[15]-c[11]*c[14]) - c[6]*(c[9]*c[15]-c[11]*c[13]) + c[7]*(c[9]*c[14]-c[10]*c[13]) / dtrm;
        inverseCovs[ci][1][0] = -(c[4]*(c[10]*c[15]-c[11]*c[14]) - c[6]*(c[8]*c[15]-c[11]*c[12]) + c[7]*(c[8]*c[14]-c[10]*c[12])) / dtrm;
        inverseCovs[ci][2][0] = c[4]*(c[9]*c[15]-c[11]*c[13]) - c[5]*(c[8]*c[15]-c[11]*c[12]) + c[7]*(c[8]*c[13]-c[9]*c[12]) / dtrm;
        inverseCovs[ci][3][0] = -(c[4]*(c[9]*c[14]-c[10]*c[13]) - c[5]*(c[8]*c[14]-c[10]*c[12]) + c[6]*(c[8]*c[13]-c[9]*c[12])) / dtrm;

        inverseCovs[ci][0][1] = -(c[1]*(c[10]*c[15]-c[11]*c[14]) - c[2]*(c[9]*c[15]-c[11]*c[13]) + c[3]*(c[9]*c[14]-c[10]*c[13])) / dtrm;
        inverseCovs[ci][1][1] = c[0]*(c[10]*c[15]-c[11]*c[14]) - c[2]*(c[8]*c[15]-c[11]*c[12]) + c[3]*(c[8]*c[14]-c[10]*c[12]) / dtrm;
        inverseCovs[ci][2][1] = -(c[0]*(c[9]*c[15]-c[11]*c[13]) - c[1]*(c[8]*c[15]-c[11]*c[12]) + c[3]*(c[8]*c[13]-c[9]*c[12])) / dtrm;
        inverseCovs[ci][3][1] = c[0]*(c[9]*c[14]-c[10]*c[13]) - c[1]*(c[8]*c[14]-c[10]*c[12]) + c[2]*(c[8]*c[13]-c[9]*c[12]) / dtrm;

        inverseCovs[ci][0][2] = c[1]*(c[6]*c[15]-c[7]*c[14]) - c[2]*(c[5]*c[15]-c[7]*c[13]) + c[3]*(c[5]*c[14]-c[6]*c[13]) / dtrm;
        inverseCovs[ci][1][2] = -(c[0]*(c[6]*c[15]-c[7]*c[14]) - c[2]*(c[4]*c[15]-c[7]*c[12]) + c[3]*(c[4]*c[14]-c[6]*c[12])) / dtrm;
        inverseCovs[ci][2][2] = c[0]*(c[5]*c[15]-c[7]*c[13]) - c[1]*(c[4]*c[15]-c[7]*c[12]) + c[3]*(c[4]*c[13]-c[5]*c[12]) / dtrm;
        inverseCovs[ci][3][2] = -(c[0]*(c[5]*c[14]-c[6]*c[13]) - c[1]*(c[4]*c[14]-c[6]*c[12]) + c[2]*(c[4]*c[13]-c[5]*c[12])) / dtrm;

        inverseCovs[ci][0][3] = -(c[1]*(c[6]*c[11]-c[7]*c[10]) - c[2]*(c[5]*c[11]-c[7]*c[9]) + c[3]*(c[5]*c[10]-c[6]*c[9])) / dtrm;
        inverseCovs[ci][1][3] = c[0]*(c[6]*c[11]-c[7]*c[10]) - c[2]*(c[4]*c[11]-c[7]*c[8]) + c[3]*(c[4]*c[10]-c[6]*c[8]) / dtrm;
        inverseCovs[ci][2][3] = -(c[0]*(c[5]*c[11]-c[7]*c[9]) - c[1]*(c[4]*c[11]-c[7]*c[8]) + c[3]*(c[4]*c[9]-c[5]*c[8])) / dtrm;
        inverseCovs[ci][3][3] = c[0]*(c[5]*c[10]-c[6]*c[9]) - c[1]*(c[4]*c[10]-c[6]*c[8]) + c[2]*(c[4]*c[9]-c[5]*c[8]) / dtrm;
    }
}

/*
  Calculate beta - parameter of GrabCut algorithm.
  beta = 1/(2*avg(sqr(||color[i] - color[j]||)))
*/
static double calcBeta( const Mat& img )
{
    double beta = 0;
    for( int y = 0; y < img.rows; y++ )
    {
        for( int x = 0; x < img.cols; x++ )
        {
            Vec4d color = img.at<Vec4b>(y,x);
            if( x>0 ) // left
            {
                Vec4d diff = color - (Vec4d)img.at<Vec4b>(y,x-1);
                beta += diff.dot(diff);
            }
            if( y>0 && x>0 ) // upleft
            {
                Vec4d diff = color - (Vec4d)img.at<Vec4b>(y-1,x-1);
                beta += diff.dot(diff);
            }
            if( y>0 ) // up
            {
                Vec4d diff = color - (Vec4d)img.at<Vec4b>(y-1,x);
                beta += diff.dot(diff);
            }
            if( y>0 && x<img.cols-1) // upright
            {
                Vec4d diff = color - (Vec4d)img.at<Vec4b>(y-1,x+1);
                beta += diff.dot(diff);
            }
        }
    }
    if( beta <= std::numeric_limits<double>::epsilon() )
        beta = 0;
    else
        beta = 1.f / (2 * beta/(4*img.cols*img.rows - 3*img.cols - 3*img.rows + 2) );

    return beta;
}

/*
  Calculate weights of noterminal vertices of graph.
  beta and gamma - parameters of GrabCut algorithm.
 */
static void calcNWeights( const Mat& img, Mat& leftW, Mat& upleftW, Mat& upW, Mat& uprightW, double beta, double gamma )
{
    const double gammaDivSqrt2 = gamma / std::sqrt(2.0f);
    leftW.create( img.rows, img.cols, CV_64FC1 );
    upleftW.create( img.rows, img.cols, CV_64FC1 );
    upW.create( img.rows, img.cols, CV_64FC1 );
    uprightW.create( img.rows, img.cols, CV_64FC1 );
    for( int y = 0; y < img.rows; y++ )
    {
        for( int x = 0; x < img.cols; x++ )
        {
            Vec4d color = img.at<Vec4b>(y,x);
            if( x-1>=0 ) // left
            {
                Vec4d diff = color - (Vec4d)img.at<Vec4b>(y,x-1);
                leftW.at<double>(y,x) = gamma * exp(-beta*diff.dot(diff));
            }
            else
                leftW.at<double>(y,x) = 0;
            if( x-1>=0 && y-1>=0 ) // upleft
            {
                Vec4d diff = color - (Vec4d)img.at<Vec4b>(y-1,x-1);
                upleftW.at<double>(y,x) = gammaDivSqrt2 * exp(-beta*diff.dot(diff));
            }
            else
                upleftW.at<double>(y,x) = 0;
            if( y-1>=0 ) // up
            {
                Vec4d diff = color - (Vec4d)img.at<Vec4b>(y-1,x);
                upW.at<double>(y,x) = gamma * exp(-beta*diff.dot(diff));
            }
            else
                upW.at<double>(y,x) = 0;
            if( x+1<img.cols && y-1>=0 ) // upright
            {
                Vec4d diff = color - (Vec4d)img.at<Vec4b>(y-1,x+1);
                uprightW.at<double>(y,x) = gammaDivSqrt2 * exp(-beta*diff.dot(diff));
            }
            else
                uprightW.at<double>(y,x) = 0;
        }
    }
}

/*
  Check size, type and element values of mask matrix.
 */
static void checkMask( const Mat& img, const Mat& mask )
{
    if( mask.empty() )
        CV_Error( CV_StsBadArg, "mask is empty" );
    if( mask.type() != CV_8UC1 )
        CV_Error( CV_StsBadArg, "mask must have CV_8UC1 type" );
    if( mask.cols != img.cols || mask.rows != img.rows )
        CV_Error( CV_StsBadArg, "mask must have as many rows and cols as img" );
    for( int y = 0; y < mask.rows; y++ )
    {
        for( int x = 0; x < mask.cols; x++ )
        {
            uchar val = mask.at<uchar>(y,x);
            if( val!=GC_BGD && val!=GC_FGD && val!=GC_PR_BGD && val!=GC_PR_FGD )
                CV_Error( CV_StsBadArg, "mask element value must be equel"
                    "GC_BGD or GC_FGD or GC_PR_BGD or GC_PR_FGD" );
        }
    }
}

/*
  Initialize mask using rectangular.
*/
static void initMaskWithRect( Mat& mask, Size imgSize, Rect rect )
{
    mask.create( imgSize, CV_8UC1 );
    mask.setTo( GC_BGD );

    rect.x = std::max(0, rect.x);
    rect.y = std::max(0, rect.y);
    rect.width = std::min(rect.width, imgSize.width-rect.x);
    rect.height = std::min(rect.height, imgSize.height-rect.y);

    (mask(rect)).setTo( Scalar(GC_PR_FGD) );
}

/*
  Initialize GMM background and foreground models using kmeans algorithm.
*/
static void initGMMs( const Mat& img, const Mat& mask, GMM& bgdGMM, GMM& fgdGMM )
{
    const int kMeansItCount = 10;
    const int kMeansType = KMEANS_PP_CENTERS;

    Mat bgdLabels, fgdLabels;
    std::vector<Vec4f> bgdSamples, fgdSamples;
    Point p;
    for( p.y = 0; p.y < img.rows; p.y++ )
    {
        for( p.x = 0; p.x < img.cols; p.x++ )
        {
            if( mask.at<uchar>(p) == GC_BGD || mask.at<uchar>(p) == GC_PR_BGD )
                bgdSamples.push_back( (Vec4f)img.at<Vec4b>(p) );
            else // GC_FGD | GC_PR_FGD
                fgdSamples.push_back( (Vec4f)img.at<Vec4b>(p) );
        }
    }
    CV_Assert( !bgdSamples.empty() && !fgdSamples.empty() );
    Mat _bgdSamples( (int)bgdSamples.size(), 4, CV_32FC1, &bgdSamples[0][0] );
    kmeans( _bgdSamples, GMM::componentsCount, bgdLabels,
            TermCriteria( CV_TERMCRIT_ITER, kMeansItCount, 0.0), 0, kMeansType );
    Mat _fgdSamples( (int)fgdSamples.size(), 4, CV_32FC1, &fgdSamples[0][0] );
    kmeans( _fgdSamples, GMM::componentsCount, fgdLabels,
            TermCriteria( CV_TERMCRIT_ITER, kMeansItCount, 0.0), 0, kMeansType );

    bgdGMM.initLearning();
    for( int i = 0; i < (int)bgdSamples.size(); i++ )
        bgdGMM.addSample( bgdLabels.at<int>(i,0), bgdSamples[i] );
    bgdGMM.endLearning();

    fgdGMM.initLearning();
    for( int i = 0; i < (int)fgdSamples.size(); i++ )
        fgdGMM.addSample( fgdLabels.at<int>(i,0), fgdSamples[i] );
    fgdGMM.endLearning();
}

/*
  Assign GMMs components for each pixel.
*/
static void assignGMMsComponents( const Mat& img, const Mat& mask, const GMM& bgdGMM, const GMM& fgdGMM, Mat& compIdxs )
{
    Point p;
    for( p.y = 0; p.y < img.rows; p.y++ )
    {
        for( p.x = 0; p.x < img.cols; p.x++ )
        {
            Vec4d color = img.at<Vec4b>(p);
            compIdxs.at<int>(p) = mask.at<uchar>(p) == GC_BGD || mask.at<uchar>(p) == GC_PR_BGD ?
                bgdGMM.whichComponent(color) : fgdGMM.whichComponent(color);
        }
    }
}

/*
  Learn GMMs parameters.
*/
static void learnGMMs( const Mat& img, const Mat& mask, const Mat& compIdxs, GMM& bgdGMM, GMM& fgdGMM )
{
    bgdGMM.initLearning();
    fgdGMM.initLearning();
    Point p;
    for( int ci = 0; ci < GMM::componentsCount; ci++ )
    {
        for( p.y = 0; p.y < img.rows; p.y++ )
        {
            for( p.x = 0; p.x < img.cols; p.x++ )
            {
                if( compIdxs.at<int>(p) == ci )
                {
                    if( mask.at<uchar>(p) == GC_BGD || mask.at<uchar>(p) == GC_PR_BGD )
                        bgdGMM.addSample( ci, img.at<Vec4b>(p) );
                    else
                        fgdGMM.addSample( ci, img.at<Vec4b>(p) );
                }
            }
        }
    }
    bgdGMM.endLearning();
    fgdGMM.endLearning();
}

/*
  Construct GCGraph
*/
static void constructGCGraph( const Mat& img, const Mat& mask, const GMM& bgdGMM, const GMM& fgdGMM, double lambda,
                       const Mat& leftW, const Mat& upleftW, const Mat& upW, const Mat& uprightW,
                       GCGraph<double>& graph )
{
    int vtxCount = img.cols*img.rows,
        edgeCount = 2*(4*img.cols*img.rows - 3*(img.cols + img.rows) + 2);
    graph.create(vtxCount, edgeCount);
    Point p;
    for( p.y = 0; p.y < img.rows; p.y++ )
    {
        for( p.x = 0; p.x < img.cols; p.x++)
        {
            // add node
            int vtxIdx = graph.addVtx();
            Vec4b color = img.at<Vec4b>(p);

            // set t-weights
            double fromSource, toSink;
            if( mask.at<uchar>(p) == GC_PR_BGD || mask.at<uchar>(p) == GC_PR_FGD )
            {
                fromSource = -log( bgdGMM(color) );
                toSink = -log( fgdGMM(color) );
            }
            else if( mask.at<uchar>(p) == GC_BGD )
            {
                fromSource = 0;
                toSink = lambda;
            }
            else // GC_FGD
            {
                fromSource = lambda;
                toSink = 0;
            }
            graph.addTermWeights( vtxIdx, fromSource, toSink );

            // set n-weights
            if( p.x>0 )
            {
                double w = leftW.at<double>(p);
                graph.addEdges( vtxIdx, vtxIdx-1, w, w );
            }
            if( p.x>0 && p.y>0 )
            {
                double w = upleftW.at<double>(p);
                graph.addEdges( vtxIdx, vtxIdx-img.cols-1, w, w );
            }
            if( p.y>0 )
            {
                double w = upW.at<double>(p);
                graph.addEdges( vtxIdx, vtxIdx-img.cols, w, w );
            }
            if( p.x<img.cols-1 && p.y>0 )
            {
                double w = uprightW.at<double>(p);
                graph.addEdges( vtxIdx, vtxIdx-img.cols+1, w, w );
            }
        }
    }
}

/*
  Estimate segmentation using MaxFlow algorithm
*/
static void estimateSegmentation( GCGraph<double>& graph, Mat& mask )
{
    graph.maxFlow();
    Point p;
    for( p.y = 0; p.y < mask.rows; p.y++ )
    {
        for( p.x = 0; p.x < mask.cols; p.x++ )
        {
            if( mask.at<uchar>(p) == GC_PR_BGD || mask.at<uchar>(p) == GC_PR_FGD )
            {
                if( graph.inSourceSegment( p.y*mask.cols+p.x /*vertex index*/ ) )
                    mask.at<uchar>(p) = GC_PR_FGD;
                else
                    mask.at<uchar>(p) = GC_PR_BGD;
            }
        }
    }
}

void cv::grabCut_rgbd( InputArray _img, InputOutputArray _mask, Rect rect,
                  InputOutputArray _bgdModel, InputOutputArray _fgdModel,
                  int iterCount, int mode )
{
    Mat img = _img.getMat();
    Mat& mask = _mask.getMatRef();
    Mat& bgdModel = _bgdModel.getMatRef();
    Mat& fgdModel = _fgdModel.getMatRef();

    if( img.empty() )
        CV_Error( CV_StsBadArg, "image is empty" );
    if( img.type() != CV_8UC4 )
        CV_Error( CV_StsBadArg, "image must have CV_8UC4 type" );

    GMM bgdGMM( bgdModel ), fgdGMM( fgdModel );
    Mat compIdxs( img.size(), CV_32SC1 );

    if( mode == GC_INIT_WITH_RECT || mode == GC_INIT_WITH_MASK )
    {
        if( mode == GC_INIT_WITH_RECT )
            initMaskWithRect( mask, img.size(), rect );
        else // flag == GC_INIT_WITH_MASK
            checkMask( img, mask );
        initGMMs( img, mask, bgdGMM, fgdGMM );
    }

    if( iterCount <= 0)
        return;

    if( mode == GC_EVAL )
        checkMask( img, mask );

    const double gamma = 50;
    const double lambda = 9*gamma;
    const double beta = calcBeta( img );

    Mat leftW, upleftW, upW, uprightW;
    calcNWeights( img, leftW, upleftW, upW, uprightW, beta, gamma );

    for( int i = 0; i < iterCount; i++ )
    {
        GCGraph<double> graph;
        assignGMMsComponents( img, mask, bgdGMM, fgdGMM, compIdxs );
        learnGMMs( img, mask, compIdxs, bgdGMM, fgdGMM );
        constructGCGraph(img, mask, bgdGMM, fgdGMM, lambda, leftW, upleftW, upW, uprightW, graph );
        estimateSegmentation( graph, mask );
    }
}
