/*=========================================================================
 *
 *  Copyright UMC Utrecht and contributors
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/

// First include the header file to be tested:
#include "itkFrugalBSplineInterpolateImageFunction.h"

#include <itkImage.h>
#include <itkImageBufferRange.h>
#include <itkImageRegionConstIteratorWithIndex.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <random>


namespace
{
using ImageType = itk::Image<float, 3>;

ImageType::Pointer
RandomImage(const unsigned int seed)
{
  const auto image = ImageType::New();
  image->SetRegions(itk::Size<3>{ { 40, 12, 20 } }); // lines longer and shorter than the horizon of a pole
  image->Allocate();
  // A spacing and a direction the derivatives go through.
  image->SetSpacing(itk::MakeVector(0.7, 1.3, 2.5));
  ImageType::DirectionType direction;
  direction.SetIdentity();
  direction(0, 0) = direction(1, 1) = 0.6;
  direction(0, 1) = -0.8;
  direction(1, 0) = 0.8;
  image->SetDirection(direction);
  std::mt19937                          generator(seed);
  std::uniform_real_distribution<float> distribution(-1000.0f, 1000.0f);
  for (float & pixel : itk::ImageBufferRange<ImageType>(*image))
  {
    pixel = distribution(generator);
  }
  return image;
}
} // namespace


// The values and the derivatives are those of the ITK function at each order and around each pixel, asked for
// apart and together, with the coefficients in the pixel type of the image, which are the pixels themselves at
// order 0 and 1, and in double precision.
GTEST_TEST(FrugalBSplineInterpolateImageFunction, HasTheValuesOfTheItkFunction)
{
  const auto image = RandomImage(1);
  const auto check = [&](const auto & expected, const auto & actual) {
    for (const unsigned int order : { 0, 1, 2, 3, 4, 5 })
    {
      expected->SetSplineOrder(order);
      expected->SetInputImage(image);
      actual->SetSplineOrder(order);
      actual->SetInputImage(image);
      unsigned int numberOfDifferences{};
      for (itk::ImageRegionConstIteratorWithIndex<ImageType> iterator(image, image->GetBufferedRegion());
           !iterator.IsAtEnd();
           ++iterator)
      {
        itk::ContinuousIndex<double, 3> index(iterator.GetIndex());
        index[0] = std::min(index[0] + 0.37, 39.0);
        index[1] = std::min(index[1] + 0.61, 11.0);
        index[2] -= 0.45; // before the first pixel too
        double                           expectedValue, actualValue;
        itk::CovariantVector<double, 3> expectedDerivative, actualDerivative;
        expected->EvaluateValueAndDerivativeAtContinuousIndex(index, expectedValue, expectedDerivative);
        actual->EvaluateValueAndDerivativeAtContinuousIndex(index, actualValue, actualDerivative);
        numberOfDifferences +=
          actual->EvaluateAtContinuousIndex(index) != expected->EvaluateAtContinuousIndex(index) ||
          actual->EvaluateDerivativeAtContinuousIndex(index) != expected->EvaluateDerivativeAtContinuousIndex(index) ||
          actualValue != expectedValue || actualDerivative != expectedDerivative;
      }
      EXPECT_EQ(numberOfDifferences, 0u) << "order " << order;
    }
  };
  check(itk::BSplineInterpolateImageFunction<ImageType, double, float>::New(),
        itk::FrugalBSplineInterpolateImageFunction<ImageType, double, float>::New());
  check(itk::BSplineInterpolateImageFunction<ImageType, double, double>::New(),
        itk::FrugalBSplineInterpolateImageFunction<ImageType, double, double>::New());
}


// An image that shares its pixels with the image before keeps its coefficients, until the pixels are modified.
GTEST_TEST(FrugalBSplineInterpolateImageFunction, KeepsTheCoefficientsOfTheSamePixels)
{
  const auto                            image = RandomImage(1);
  const auto                            other = RandomImage(2);
  const itk::ContinuousIndex<double, 3> index(itk::MakePoint(3.25, 6.5, 8.75).GetDataPointer());

  const auto expected = itk::BSplineInterpolateImageFunction<ImageType>::New();
  const auto actual = itk::FrugalBSplineInterpolateImageFunction<ImageType>::New();
  expected->SetInputImage(image);
  actual->SetInputImage(image);

  // Another image of the same pixels, as a pyramid level grafted from the level before.
  const auto level = ImageType::New();
  level->Graft(image);
  actual->SetInputImage(level);
  EXPECT_EQ(actual->EvaluateAtContinuousIndex(index), expected->EvaluateAtContinuousIndex(index));

  // The pixels change, and their container says so: the coefficients are computed again.
  std::copy_n(other->GetBufferPointer(), 40 * 12 * 20, image->GetBufferPointer());
  image->GetPixelContainer()->Modified();
  expected->SetInputImage(other);
  actual->SetInputImage(level);
  EXPECT_EQ(actual->EvaluateAtContinuousIndex(index), expected->EvaluateAtContinuousIndex(index));
}
