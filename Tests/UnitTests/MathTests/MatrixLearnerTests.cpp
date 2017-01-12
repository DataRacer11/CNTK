//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#include "stdafx.h"
#include <math.h>
#ifdef _WIN32
#include <crtdefs.h>
#endif 
#include "../../../Source/Math/Matrix.h"
#include "../../../Source/Math/CPUMatrix.h"

using namespace Microsoft::MSR::CNTK;

static const size_t dim1 = 256;
static const size_t dim2 = 128;
static const size_t dim3 = 2048;

static void PrepareMatrices(
    SingleMatrix& matSG,
    SingleMatrix& matSGsparse,
    SingleMatrix& matM,
    SingleMatrix& matMsparse,
    SingleMatrix& matG,
    SingleMatrix& matGsparseBSC,
    RandomSeedFixture& r)
{
    // smoothed gradient
    matSG = SingleMatrix::RandomGaussian(dim1, dim2, c_deviceIdZero, -1.0f, 1.0f, r.IncrementCounter());
    matSGsparse = SingleMatrix(matSG.DeepClone());

    // model
    matM = SingleMatrix::RandomGaussian(dim1, dim2, c_deviceIdZero, -1.0f, 1.0f, r.IncrementCounter());
    matMsparse = SingleMatrix(matM.DeepClone());

    // generates gradient
    SingleMatrix matG1(c_deviceIdZero);
    matG1.AssignTruncateBottomOf(Matrix<float>::RandomUniform(dim2, dim3, c_deviceIdZero, -300.0f, 0.1f, r.IncrementCounter()), 0);

    SingleMatrix matG1sparseCSC(matG1.DeepClone());
    matG1sparseCSC.SwitchToMatrixType(MatrixType::SPARSE, matrixFormatSparseCSC, true);

    SingleMatrix matG2 = SingleMatrix::RandomGaussian(dim1, dim3, c_deviceIdZero, -1.0f, 1.0f, r.IncrementCounter());

    SingleMatrix::MultiplyAndWeightedAdd(1, matG2, false, matG1, true, 0, matG);

    matGsparseBSC.SwitchToMatrixType(MatrixType::SPARSE, matrixFormatSparseBlockCol, false);
    SingleMatrix::MultiplyAndAdd(matG2, false, matG1sparseCSC, true, matGsparseBSC);
}

namespace Microsoft { namespace MSR { namespace CNTK { namespace Test {

BOOST_AUTO_TEST_SUITE(MatrixLearnerSuite)

// tests FSAdagrad sparse vs. dense
BOOST_FIXTURE_TEST_CASE(FSAdagradSparse, RandomSeedFixture)
{
    SingleMatrix matSG(c_deviceIdZero);
    SingleMatrix matSGsparse(c_deviceIdZero);
    SingleMatrix matM(c_deviceIdZero);
    SingleMatrix matMsparse(c_deviceIdZero);
    SingleMatrix matG(c_deviceIdZero);
    SingleMatrix matGsparseBSC(c_deviceIdZero);

    PrepareMatrices(matSG, matSGsparse, matM, matMsparse, matG, matGsparseBSC, *this);

    // run learner
    double smoothedCount = 1000;
    matSG.FSAdagradUpdate(dim2, matG, matM, smoothedCount, 0.0001, 1.0, 0.9, 0.9);

    smoothedCount = 1000;
    matSGsparse.FSAdagradUpdate(dim2, matGsparseBSC, matMsparse, smoothedCount, 0.0001, 1.0, 0.9, 0.9);

    BOOST_CHECK(matSG.IsEqualTo(matSGsparse, c_epsilonFloatE5));
    BOOST_CHECK(matM.IsEqualTo(matMsparse, c_epsilonFloatE5));
}

// tests RmsProp sparse vs. dense
BOOST_FIXTURE_TEST_CASE(RmsPropSparse, RandomSeedFixture)
{
    SingleMatrix matSG(c_deviceIdZero);
    SingleMatrix matSGsparse(c_deviceIdZero);
    SingleMatrix matM(c_deviceIdZero);
    SingleMatrix matMsparse(c_deviceIdZero);
    SingleMatrix matG(c_deviceIdZero);
    SingleMatrix matGsparseBSC(c_deviceIdZero);

    PrepareMatrices(matSG, matSGsparse, matM, matMsparse, matG, matGsparseBSC, *this);

    // run learner
    matSG.RmsProp(matG, 0.99f, 1.2f, 10.0f, 0.75f, 0.1f, false);
    matSGsparse.RmsProp(matGsparseBSC, 0.99f, 1.2f, 10.0f, 0.75f, 0.1f, false);

    BOOST_CHECK(matSG.IsEqualTo(matSGsparse, c_epsilonFloatE4));
}

BOOST_AUTO_TEST_SUITE_END()
}}}}
