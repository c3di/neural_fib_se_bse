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
    std::fstream ifs("../data/Sphere_Vv03_r10-15_Num1_ITWMconfig.txt", std::ifstream::in);
    std::string line;

    std::getline(ifs, line); // check number of spheres
    int n_spheres = stoi(line);
    spheres.reserve(n_spheres);

    std::getline(ifs, line); // discard first entry
    while ( std::getline(ifs, line )  )
    {
        std::istringstream line_stream(line);
        int id;
        Sphere sphere;

        line_stream >> id >> sphere.x >> sphere.y >> sphere.z >> sphere.r;
        spheres.push_back(sphere);
    }
    ifs.close();

    std::cout << "performing preprocessing" << std::endl;
    auto preprocessor = new HeightFieldExtractor( std::pair<int, int>(850, 850), 2, 64 );
    preprocessor->add_spheres( spheres );

    auto extended_heightfield = preprocessor->extract_data_representation( 0.0f );
    std::cout << "done" << std::endl;

    delete preprocessor;
}
