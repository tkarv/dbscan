//Copyright (c) 2018 Tuukka Karvonen
//
//Permission is hereby granted, free of charge, to any person obtaining a copy
//of this software and associated documentation files (the "Software"), to deal
//in the Software without restriction, including without limitation the rights
//to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//copies of the Software, and to permit persons to whom the Software is
//furnished to do so, subject to the following conditions:
//
//The above copyright notice and this permission notice shall be included in all
//copies or substantial portions of the Software.
//
//THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//SOFTWARE.

#ifndef DBSCAN_H
#define DBSCAN_H

#include <math.h>
#include <vector>

class DBSCAN
{
public:
    DBSCAN(const float epsilon, const int min_samples);
    std::vector<int> cluster(const std::vector<std::vector<float>> &features);

    int numClusters() const { return num_clusters; }

private:
    std::vector<int> findNeighbors(const std::vector<float> &dataPoint, const int &idx, const std::vector<std::vector<float>> &features);

    float m_eps;
    float m_msamples;

    int num_clusters;
};

#endif // DBSCAN_H
