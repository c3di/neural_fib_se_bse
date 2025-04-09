#include <iostream>
#include <iomanip>
#include <cuda.h>
#include <cuda_runtime.h>
#include <random>

#include "cuda_matrix.h"

std::ostream& operator<<(std::ostream& os, const Matrix3x3& m)
{
    std::cout << std::setprecision(2);
    for ( int i = 0; i<3; i++)
        os << m.m_row[i].x << " " << m.m_row[i].y << " " << m.m_row[i].z << std::endl;
    os << std::endl;
    return os;
}

void perform_matrix_test()
{
	Matrix3x3 m;
    setIdentity(m);
    std::cout << m;

    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_real_distribution<> distribution(-10.0f, 10.0f );

    Matrix3x3 a,b;
    for (int i = 0; i < 3; i++) {
        a.m_row[i].x = distribution(rng);
        a.m_row[i].y = distribution(rng);
        a.m_row[i].z = distribution(rng);
    }

    std::cout << a;
    b = getInverse(a);
    std::cout << b;

    Matrix3x3 c = MatrixMul(a,b);
    std::cout << c;
}