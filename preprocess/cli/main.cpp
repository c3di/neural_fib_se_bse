#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <iterator>
#include <vector>

#include "HeightFieldExtractor.h"
#include "sphere.h"
#include "cylinder.h"
// #include "cuda_matrix.h"

int main( int argc, char* argv[] )
{ 
	std::cout << "reading input" << std::endl;
    std::vector<Sphere> spheres;
    std::vector<Cylinder> cylinders;

    bool do_read = false;
    if (do_read) 
    {
        std::fstream ifs("../data/Sphere_Vv03_r10-15_Num1_ITWMconfig.txt", std::ifstream::in);
        std::string line;

        std::getline(ifs, line); // check number of spheres
        int n_spheres = stoi(line);
        spheres.reserve(n_spheres);

        std::getline(ifs, line); // discard first entry
        int n_cylinders = stoi(line);

        int id;
        Sphere sphere;
        for (size_t i = 0; i < n_spheres; i++)
        {
            std::getline(ifs, line);
            std::istringstream line_stream(line);
            line_stream >> id >> sphere.x >> sphere.y >> sphere.z >> sphere.r;
            spheres.push_back(sphere);
        }

        Cylinder cylinder;
        for (size_t i = 0; i < n_spheres; i++)
        {
            std::getline(ifs, line);
            std::istringstream line_stream(line);
            line_stream >> id >> cylinder.x >> cylinder.y >> cylinder.z >> cylinder.orientation.x >> cylinder.orientation.y >> cylinder.orientation.z >> cylinder.orientation.w >> cylinder.r >> cylinder.l;
            cylinders.push_back(cylinder);
        }
        ifs.close();
    }
    else 
    {
        Cylinder cylinder;
        cylinder.x = 425.0f;
        cylinder.y = 425.0f;
        cylinder.z = 400.0f;
        cylinder.orientation = make_float4(0.0, 1.0, 0.0, 0.78539816339f / 2.0f);
        cylinder.r = 50.0f;
        cylinder.l = 150.0f;
        cylinders.push_back(cylinder);
        cylinder.orientation = make_float4(1.0, 0.0, 0.0, 0.78539816339f / 2.0f);
        cylinders.push_back(cylinder);
    }

    std::cout << "performing preprocessing" << std::endl;
    auto preprocessor = new HeightFieldExtractor( std::tuple<int, int>(850, 850), 2, 64 );

    if ( spheres.size() > 0 )
        preprocessor->add_spheres(spheres);
    if ( cylinders.size() > 0 )
        preprocessor->add_cylinders(cylinders);

    auto extended_heightfield = preprocessor->extract_data_representation( 0.0f );
    std::cout << "done" << std::endl;

    delete preprocessor;
}
