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

#include "dbscan.hpp"

DBSCAN::DBSCAN(const float epsilon, const int min_samples) : m_eps(epsilon), m_msamples(min_samples), num_clusters(0)
{

}

std::vector<int> DBSCAN::findNeighbors(const std::vector<float> &dataPoint, const int &idx, const std::vector<std::vector<float>> &features) {
    std::vector<int> ret;
    auto dist = [](const std::vector<float> &dp1, const std::vector<float> &dp2) {
        float accum = 0.0;
        for(int j=0;j<dp1.size();j++) {
            accum += (dp1[j] - dp2[j])*(dp1[j] - dp2[j]);
        }
        return sqrt(accum);
    };
    for(int j=0;j<features.size();j++) {
        if(j==idx) continue;
        if(dist(features[j], dataPoint) < m_eps) {
            ret.push_back(j);
        }
    }
    return ret;
}

std::vector<int> DBSCAN::cluster(const std::vector<std::vector<float> > &features) {
    int C = 0; // cluster counter
    std::vector<int> labels = std::vector<int>(features.size(), 0);
    for(int j=0;j<features.size();j++) {
        const auto &dataPoint = features[j];
        if(labels[j] != 0) { // unclustered
            continue;
        }
        // find neighbors
        std::vector<int> neighbors = findNeighbors(dataPoint, j, features);
        if(neighbors.size() < m_msamples-1 /* current point is not included */) {
            labels[j] = -1; // noise
        }
        C++;
        labels[j] = C;
        for(int i=0;i<neighbors.size();i++) {
            if(labels[neighbors[i]] == -1) { // noise
                labels[neighbors[i]] = C; // change from noise to border point
            } else if(labels[neighbors[i]] != 0) { // not undefined nor noise
                continue;
            }
            labels[neighbors[i]] = C;
            std::vector<int> nbrs = findNeighbors(features[neighbors[i]], neighbors[i], features);
            if(nbrs.size() >= m_msamples-1) {
                // nbrs contains indexes to features; neighbors contains indexes to features
                // for each nbrs not in neighhbors, append to neighbors
                for(int h=0;h<nbrs.size();h++) {
                    if(std::find(neighbors.begin(), neighbors.end(), nbrs[h]) == neighbors.end() && (nbrs[h] != j)) { // not found
                        neighbors.push_back(nbrs[h]);
                    }
                }
            }
        }
    }

    num_clusters = C; // 0 is not a cluster
    return labels;
}
