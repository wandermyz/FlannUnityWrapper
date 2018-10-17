#define _CRT_SECURE_NO_DEPRECATE
#define _SCL_SECURE_NO_WARNINGS

#include <flann/flann.h>

#define DllExport extern "C" __declspec( dllexport )   

class FlannPointCloud
{
private:
    flann::Matrix<double> dataset;
    flann::Index<flann::L2<double> > index;

public:
    FlannPointCloud(double* points, size_t numInput, size_t dimension) :
        dataset(points, numInput, dimension),
        index(dataset, flann::KDTreeIndexParams(8))
    {
        index.buildIndex();
    }

    int QueryFlannPointCloud(double x, double y, double z, double radius, int limit, int* outIndices)
    {
        double queryData[] = { x, y, z };
        flann::Matrix<double> query(queryData, 1, 3);

        std::vector< std::vector<int> > indices;
        std::vector< std::vector<double> > dists;
        flann::SearchParams params;
        params.checks = -1;
        params.sorted = true;
        params.max_neighbors = limit;

        int resultCount = index.radiusSearch(query, indices, dists, radius, params);
        int resCount = std::min(limit, resultCount);
        std::copy_n(indices[0].begin(), std::min(limit, resCount), outIndices);

        return resCount;
    }
};

DllExport FlannPointCloud* CreateFlannPointCloud(double* rawData, size_t length)
{
    FlannPointCloud* pc = new FlannPointCloud(rawData, (size_t)(length / 3), 3);
    return pc;
}

DllExport void DeleteFlannPointCloud(FlannPointCloud* pc)
{
    delete pc;
}

DllExport int QueryFlannPointCloud(FlannPointCloud* pc, double x, double y, double z, double radius, int limit, int* outIndices)
{
    return pc->QueryFlannPointCloud(x, y, z, radius, limit, outIndices);
}

