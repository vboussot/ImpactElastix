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
#include "itkStreamedRecursivePyramidImageFilter.h"

#include <itkImage.h>
#include <itkImageBufferRange.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <random>


// The levels are those of the ITK filter to the bit, the smaller ones computed at once or in several pieces, and the
// level of factor one shares the pixels of the image.
GTEST_TEST(StreamedRecursivePyramidImageFilter, HasTheLevelsOfTheItkFilter)
{
  using ImageType = itk::Image<float, 3>;
  using RangeType = itk::ImageBufferRange<const ImageType>;

  const auto image = ImageType::New();
  image->SetRegions(itk::Size<3>{ { 300, 280, 260 } }); // 83 MiB: two pieces
  image->Allocate();
  image->SetSpacing(itk::MakeVector(0.5, 0.75, 1.25));
  image->SetOrigin(itk::MakePoint(-10.0, 4.0, 30.0));
  std::mt19937                          generator(1);
  std::uniform_real_distribution<float> distribution(-1000.0f, 1000.0f);
  for (float & pixel : itk::ImageBufferRange<ImageType>(*image))
  {
    pixel = distribution(generator);
  }
  ASSERT_GT(image->GetLargestPossibleRegion().GetNumberOfPixels() * sizeof(float),
            (itk::StreamedRecursivePyramidImageFilter<ImageType, ImageType>::PieceSize));

  const auto expected = itk::RecursiveMultiResolutionPyramidImageFilter<ImageType, ImageType>::New();
  expected->SetInput(image);
  expected->SetNumberOfLevels(3);
  expected->UpdateLargestPossibleRegion();
  for (const bool inPieces : { false, true })
  {
    itk::StreamedRecursivePyramidOptions::InPieces() = inPieces;
    const auto actual = itk::StreamedRecursivePyramidImageFilter<ImageType, ImageType>::New();
    actual->SetInput(image);
    actual->SetNumberOfLevels(3);
    actual->UpdateLargestPossibleRegion();
    for (unsigned int level = 0; level < 3; ++level)
    {
      const ImageType & level1 = *expected->GetOutput(level);
      const ImageType & level2 = *actual->GetOutput(level);
      ASSERT_EQ(level2.GetBufferedRegion(), level1.GetBufferedRegion());
      EXPECT_EQ(level2.GetLargestPossibleRegion(), level1.GetLargestPossibleRegion());
      EXPECT_EQ(level2.GetSpacing(), level1.GetSpacing());
      EXPECT_EQ(level2.GetOrigin(), level1.GetOrigin());
      const RangeType range1(level1);
      const RangeType range2(level2);
      EXPECT_TRUE(std::equal(range1.begin(), range1.end(), range2.begin()));
    }
    EXPECT_EQ(actual->GetOutput(2)->GetBufferPointer(), image->GetBufferPointer());
    EXPECT_EQ(image->GetRequestedRegion(), image->GetLargestPossibleRegion());
  }
  itk::StreamedRecursivePyramidOptions::InPieces() = false;
}


// Asked to keep them, a pyramid takes the smaller levels of the pyramid of the image before, until the image is
// modified.
GTEST_TEST(StreamedRecursivePyramidImageFilter, KeepsTheLevelsOfAnImage)
{
  using ImageType = itk::Image<float, 3>;

  const auto image = ImageType::New();
  image->SetRegions(itk::Size<3>::Filled(32));
  image->AllocateInitialized();
  const auto coarsestLevel = [image] {
    const auto pyramid = itk::StreamedRecursivePyramidImageFilter<ImageType, ImageType>::New();
    pyramid->SetInput(image);
    pyramid->SetNumberOfLevels(2);
    pyramid->UpdateLargestPossibleRegion();
    return ImageType::Pointer(pyramid->GetOutput(0));
  };

  const auto inMemory = coarsestLevel();
  EXPECT_NE(coarsestLevel()->GetBufferPointer(), inMemory->GetBufferPointer());

  itk::StreamedRecursivePyramidOptions::KeepsLevels() = true;
  const auto kept = coarsestLevel();
  EXPECT_EQ(coarsestLevel()->GetBufferPointer(), kept->GetBufferPointer());
  image->Modified();
  EXPECT_NE(coarsestLevel()->GetBufferPointer(), kept->GetBufferPointer());
  itk::StreamedRecursivePyramidOptions::KeepsLevels() = false;
}
