#pragma once

/*****************************************
                Vector
/*****************************************/

__host__ __device__
inline float3 operator+(const float3& a, const float3& b) {

    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__host__ __device__
inline float3 operator*(const float& a, const float3& b) {

    return make_float3(a * b.x, a * b.y, a * b.z);
}

__host__ __device__
inline float3 getCrossProduct(float3 a, float3 b)
{
    return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

__host__ __device__
inline float getDotProduct(float3 a, float3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__
inline float3 getNormalizedVec(const float3 v)
{
    float invLen = 1.0f / sqrtf(getDotProduct(v, v));
    return make_float3(v.x * invLen, v.y * invLen, v.z * invLen);
}

/*****************************************
                Matrix3x3
/*****************************************/
struct Matrix3x3
{
    float3 m_row[3];
    __host__ __device__ inline const float& m00() const { return m_row[0].x; };
    __host__ __device__ inline const float& m01() const { return m_row[0].y; };
    __host__ __device__ inline const float& m02() const { return m_row[0].z; };

    __host__ __device__ inline const float& m10() const { return m_row[1].x; };
    __host__ __device__ inline const float& m11() const { return m_row[1].y; };
    __host__ __device__ inline const float& m12() const { return m_row[1].z; };

    __host__ __device__ inline const float& m20() const { return m_row[2].x; };
    __host__ __device__ inline const float& m21() const { return m_row[2].y; };
    __host__ __device__ inline const float& m22() const { return m_row[2].z; };

    __host__ __device__ inline float& m00() { return m_row[0].x; };
    __host__ __device__ inline float& m01() { return m_row[0].y; };
    __host__ __device__ inline float& m02() { return m_row[0].z; };

    __host__ __device__ inline float& m10() { return m_row[1].x; };
    __host__ __device__ inline float& m11() { return m_row[1].y; };
    __host__ __device__ inline float& m12() { return m_row[1].z; };

    __host__ __device__ inline float& m20() { return m_row[2].x; };
    __host__ __device__ inline float& m21() { return m_row[2].y; };
    __host__ __device__ inline float& m22() { return m_row[2].z; };
};

__host__ __device__
inline void setZero(Matrix3x3& m)
{
    m.m_row[0] = make_float3(0.0f, 0.0f, 0.0f);
    m.m_row[1] = make_float3(0.0f, 0.0f, 0.0f);
    m.m_row[2] = make_float3(0.0f, 0.0f, 0.0f);
}

__host__ __device__
inline void setIdentity(Matrix3x3& m)
{
    m.m_row[0] = make_float3(1.0f, 0.0f, 0.0f);
    m.m_row[1] = make_float3(0.0f, 1.0f, 0.0f);
    m.m_row[2] = make_float3(0.0f, 0.0f, 1.0f);
}

__host__ __device__
inline Matrix3x3 getTranspose(const Matrix3x3 m)
{
    Matrix3x3 out;
    out.m_row[0] = make_float3(m.m_row[0].x, m.m_row[1].x, m.m_row[2].x);
    out.m_row[1] = make_float3(m.m_row[0].y, m.m_row[1].y, m.m_row[2].y);
    out.m_row[2] = make_float3(m.m_row[0].z, m.m_row[1].z, m.m_row[2].z);
    return out;
}

__host__ __device__
inline float getDeterminant(const Matrix3x3& m)
{
    return m.m00() * (m.m11() * m.m22() - m.m21() * m.m12() ) -
           m.m01() * (m.m10() * m.m22() - m.m12() * m.m20() ) +
           m.m02() * (m.m10() * m.m21() - m.m11() * m.m20() );
}

__host__ __device__
inline Matrix3x3 getInverse(const Matrix3x3 m)
{
    float invdet = 1.0 / getDeterminant(m);

    Matrix3x3 minv;
    minv.m00() = (m.m11() * m.m22() - m.m21() * m.m12()) * invdet;
    minv.m01() = (m.m02() * m.m21() - m.m01() * m.m22()) * invdet;
    minv.m02() = (m.m01() * m.m12() - m.m02() * m.m11()) * invdet;
    minv.m10() = (m.m12() * m.m20() - m.m10() * m.m22()) * invdet;
    minv.m11() = (m.m00() * m.m22() - m.m02() * m.m20()) * invdet;
    minv.m12() = (m.m10() * m.m02() - m.m00() * m.m12()) * invdet;
    minv.m20() = (m.m10() * m.m21() - m.m20() * m.m11()) * invdet;
    minv.m21() = (m.m20() * m.m01() - m.m00() * m.m21()) * invdet;
    minv.m22() = (m.m00() * m.m11() - m.m10() * m.m01()) * invdet;
    return minv;
}

__host__ __device__
inline Matrix3x3 MatrixMul( const Matrix3x3& a, const Matrix3x3& b)
{
    Matrix3x3 transB = getTranspose(b);
    Matrix3x3 ans;

    for (int i = 0; i < 3; i++)
    {
        ans.m_row[i].x = getDotProduct(a.m_row[i], transB.m_row[0]);
        ans.m_row[i].y = getDotProduct(a.m_row[i], transB.m_row[1]);
        ans.m_row[i].z = getDotProduct(a.m_row[i], transB.m_row[2]);
    }
    return ans;
}

__host__ __device__
inline float3 MatrixMul( const Matrix3x3& a, float3 b)
{
    return make_float3( getDotProduct( a.m_row[0], b ), getDotProduct( a.m_row[1], b ), getDotProduct( a.m_row[2], b ) );
}

__host__ __device__
inline Matrix3x3 getXRotationMatrix(float theta)
{
    Matrix3x3 out;
    const float cos_theta = cosf(theta);
    const float sin_theta = sinf(theta);
    out.m_row[0] = make_float3(1.0f, 0.0f,       0.0f);
    out.m_row[1] = make_float3(0.0f, cos_theta, -sin_theta);
    out.m_row[2] = make_float3(0.0f, sin_theta,  cos_theta);
    return out;
}

__host__ __device__
inline Matrix3x3 getYRotationMatrix(float theta)
{
    Matrix3x3 out;
    const float cos_theta = cosf(theta);
    const float sin_theta = sinf(theta);
    out.m_row[0] = make_float3(cos_theta, 0.0f, sin_theta);
    out.m_row[1] = make_float3(0.0f, 1.0f, 0.0f);
    out.m_row[2] = make_float3(-sin_theta, 0.0f, cos_theta);
    return out;
}

__host__ __device__
inline Matrix3x3 getZRotationMatrix(float theta)
{
    Matrix3x3 out;
    const float cos_theta = cosf(theta);
    const float sin_theta = sinf(theta);
    out.m_row[0] = make_float3(cos_theta, -sin_theta, 0.0f);
    out.m_row[1] = make_float3(sin_theta,  cos_theta, 0.0f);
    out.m_row[2] = make_float3(0.0f,       0.0f,      1.0f);
    return out;
}

__host__ __device__
inline Matrix3x3 getFromEulerAngles( float3 angles )
{
    Matrix3x3 rotX = getXRotationMatrix(angles.x);
    Matrix3x3 rotY = getYRotationMatrix(angles.y);
    Matrix3x3 rotZ = getZRotationMatrix(angles.z);
    Matrix3x3 rotXY = MatrixMul(rotX, rotY);
    return MatrixMul( rotZ, rotXY );
}

/*****************************************
                Test Functions 
/*****************************************/
void perform_matrix_test();