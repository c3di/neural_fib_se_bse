#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <iterator>
#include <vector>

#include "sphere.h"
#include "HeightFieldExtractor.h"
#include "cuda_matrix.h"

int main( int argc, char* argv[] )
{ 
	std::cout << "reading input" << std::endl;
    std::vector<Sphere> spheres;
    std::vector<Cylinder> cylinders;
    std::fstream ifs("../data/Sphere_Vv03_r10-15_Num1_ITWMconfig.txt", std::ifstream::in);
    std::string line;

    std::getline(ifs, line); // check number of spheres
    int n_spheres = stoi(line);
    spheres.reserve(n_spheres);

    std::getline(ifs, line); // discard first entry
    int n_cylinders  = stoi(line);

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
        line_stream >> id >> cylinder.x >> cylinder.y >> cylinder.z >> cylinder.orientation.x >> cylinder.orientation.y >> cylinder.orientation.z >> cylinder.r >> cylinder.l;
        cylinders.push_back(cylinder);
    }
    ifs.close();

    std::cout << "performing preprocessing" << std::endl;
    auto preprocessor = new HeightFieldExtractor( std::pair<int, int>(850, 850), 2, 64 );
    preprocessor->add_spheres(spheres);
    preprocessor->add_cylinders(cylinders);

    auto extended_heightfield = preprocessor->extract_data_representation( 0.0f );
    std::cout << "done" << std::endl;

    delete preprocessor;
}
