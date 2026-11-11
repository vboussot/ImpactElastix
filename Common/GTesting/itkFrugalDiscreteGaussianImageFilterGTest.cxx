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
#include "itkFrugalDiscreteGaussianImageFilter.h"

#include <itkDiscreteGaussianImageFilter.h>
#include <itkImage.h>
#include <itkImageBufferRange.h>
#include <itkStreamingImageFilter.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <random>


namespace
{
// The pixels are those of the ITK filter to the bit, computed at once or in several pieces, for kernels smaller and
// larger than the image, and for an axis that is not smoothed.
template <typename TImage>
void
ExpectThePixelsOfTheItkFilter(const typename TImage::SizeType & size, const double * const variance)
{
  using RangeType = itk::ImageBufferRange<const TImage>;

  const auto image = TImage::New();
  image->SetRegions(size);
  image->Allocate();
  std::mt19937                          generator(1);
  std::uniform_real_distribution<float> distribution(-1000.0f, 1000.0f);
  for (auto & pixel : itk::ImageBufferRange<TImage>(*image))
  {
    pixel = static_cast<typename TImage::PixelType>(distribution(generator));
  }

  const auto expected = itk::DiscreteGaussianImageFilter<TImage, TImage>::New();
  expected->SetUseImageSpacing(false);
  expected->SetMaximumError(0.1);
  expected->SetVariance(variance);
  expected->SetInput(image);
  expected->UpdateLargestPossibleRegion();

  const auto actual = itk::FrugalDiscreteGaussianImageFilter<TImage>::New();
  actual->SetMaximumError(0.1);
  actual->SetVariance(variance);
  actual->SetInput(image);
  for (const unsigned int numberOfPieces : { 1, 5 })
  {
    const auto streamer = itk::StreamingImageFilter<TImage, TImage>::New();
    streamer->SetInput(actual->GetOutput());
    streamer->SetNumberOfStreamDivisions(numberOfPieces);
    streamer->UpdateLargestPossibleRegion();
    const RangeType range1(*expected->GetOutput());
    const RangeType range2(*streamer->GetOutput());
    ASSERT_EQ(range2.size(), range1.size());
    EXPECT_TRUE(std::equal(range1.begin(), range1.end(), range2.begin())) << numberOfPieces << " pieces";
  }
}
} // namespace


GTEST_TEST(FrugalDiscreteGaussianImageFilter, HasThePixelsOfTheItkFilter)
{
  const double pyramid[] = { 1.0, 1.0, 0.0 };
  const double wide[] = { 4.0, 0.25, 16.0 };
  ExpectThePixelsOfTheItkFilter<itk::Image<float, 3>>({ { 37, 23, 19 } }, pyramid);
  ExpectThePixelsOfTheItkFilter<itk::Image<float, 3>>({ { 37, 3, 19 } }, wide);
  ExpectThePixelsOfTheItkFilter<itk::Image<short, 3>>({ { 37, 23, 19 } }, wide);
  ExpectThePixelsOfTheItkFilter<itk::Image<float, 2>>({ { 37, 23 } }, wide);
  // More than a slab of slices.
  ExpectThePixelsOfTheItkFilter<itk::Image<float, 3>>({ { 300, 280, 12 } }, pyramid);
}
