#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <iterator>
#include <vector>

#include "HeightFieldExtractor.h"
#include "sphere.h"
#include "cylinder.h"
#include "cuboid.h"

int main( int argc, char* argv[] )
{ 
	std::cout << "reading input" << std::endl;
    std::vector<Sphere>   spheres;
    std::vector<Cylinder> cylinders;
    std::vector<Cuboid>   cuboids;

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
            line_stream >> id >> sphere.position.x >> sphere.position.y >> sphere.position.z >> sphere.r;
            spheres.push_back(sphere);
        }

        Cylinder cylinder;
        for (size_t i = 0; i < n_spheres; i++)
        {
            std::getline(ifs, line);
            std::istringstream line_stream(line);
            line_stream >> id >> cylinder.position.x >> cylinder.position.y >> cylinder.position.z >> cylinder.orientation.x >> cylinder.orientation.y >> cylinder.orientation.z >> cylinder.orientation.w >> cylinder.r >> cylinder.l;
            cylinders.push_back(cylinder);
        }

        ifs.close();
    }
    else 
    {
        Cylinder cylinder;
        cylinder.position.x = 425.0f;
        cylinder.position.y = 425.0f;
        cylinder.position.z = 400.0f;
        cylinder.orientation = make_float4(0.0, 1.0, 0.0, 0.78539816339f / 2.0f);
        cylinder.r = 50.0f;
        cylinder.l = 150.0f;
        cylinders.push_back(cylinder);
        cylinder.orientation = make_float4(1.0, 0.0, 0.0, 0.78539816339f / 2.0f);
        cylinders.push_back(cylinder); 

        Cuboid cuboid;
        cuboid.position.x = 425.0f;
        cuboid.position.y = 425.0f;
        cuboid.position.z = 400.0f;
        cuboid.orientation = make_float4(0.0, 1.0, 0.0, 0.78539816339f / 2.0f);
        cuboid.size.x = 10.0f;
        cuboid.size.y = 20.0f;
        cuboid.size.z = 40.0f;
        cuboids.push_back(cuboid);
    }

    std::cout << "performing preprocessing" << std::endl;

    int x_res = 850;
    int y_res = 850;

    auto preprocessor = new HeightFieldExtractor( std::tuple<int, int>( x_res, y_res ), 2, 64 );

    if ( spheres.size() > 0 )
        preprocessor->add_spheres(spheres);
    if (cylinders.size() > 0)
        preprocessor->add_cylinders(cylinders);
    if (cuboids.size() > 0)
        preprocessor->add_cuboids(cuboids);

    auto data_representation = preprocessor->extract_data_representation( 0.0f );

    auto extended_heightfield = std::get<0>(data_representation);

    for (size_t i = 0; i < x_res * y_res; ++i)
    {
        auto entry_exit = extended_heightfield[i];
        {
            if ((entry_exit.x < 65535.0f) && (entry_exit.x > 0.0f))
                std::cout << entry_exit.x << "/" << entry_exit.y << std::endl;
        }
    }
    std::cout << "done" << std::endl;

    delete preprocessor;
}