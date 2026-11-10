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
#include "itkFileMappedImageContainer.h"

#include <itkImage.h>
#include <itkImageFileWriter.h>
#include <itkImageRegionRange.h>
#include <itksys/SystemTools.hxx>

#include <gtest/gtest.h>

#include <algorithm>
#include <numeric>

#ifndef _WIN32

// An image as large as the threshold keeps its buffer in a file of the directory, zero-filled, and reads back what
// is written to it; a smaller one is allocated as usual; once the factory is gone, so is every image.
GTEST_TEST(FileMappedImageContainer, MapsTheBufferOfALargeImage)
{
  using ImageType = itk::Image<float, 3>;
  using MappedContainerType = itk::FileMappedImageContainer<float>;

  const std::string directory = itksys::SystemTools::GetCurrentWorkingDirectory();
  itk::FileMappedImageContainerFactory::MapImagesFrom(directory);
  const itk::ObjectFactoryBase::Pointer factory = itk::ObjectFactoryBase::GetRegisteredFactories().back();

  {
    const auto large = ImageType::New();
    large->SetRegions(itk::Size<3>::Filled(162)); // 162^3 floats: just past 16 MiB
    large->AllocateInitialized();
    ASSERT_NE(dynamic_cast<MappedContainerType *>(large->GetPixelContainer()), nullptr);

    const itk::ImageRegionRange<ImageType> range(*large, large->GetBufferedRegion());
    EXPECT_TRUE(std::all_of(range.begin(), range.end(), [](const float value) { return value == 0.0f; }));
    large->FillBuffer(3.0f);
    EXPECT_EQ(large->GetPixel(itk::Index<3>::Filled(161)), 3.0f);
    // Its file left the directory when it was mapped, and its mapping goes with the image.
    EXPECT_FALSE(itk::FileMappedScratch::Release(nullptr));
    float * const buffer = large->GetBufferPointer();
    large->Initialize();
    EXPECT_FALSE(itk::FileMappedScratch::Release(buffer));

    const auto small = ImageType::New();
    small->SetRegions(itk::Size<3>::Filled(8));
    small->AllocateInitialized();
    small->FillBuffer(1.0f);
    EXPECT_EQ(small->GetPixel(itk::Index<3>::Filled(7)), 1.0f);
  }
  itk::ObjectFactoryBase::UnRegisterFactory(factory);
  EXPECT_EQ(dynamic_cast<MappedContainerType *>(ImageType::New()->GetPixelContainer()), nullptr);
}


// The pixels of an uncompressed MetaImage are mapped from its file; a compressed one, or one of another pixel type,
// is left to its reader, and so is every file without the factory.
GTEST_TEST(FileMappedImageContainer, MapsThePixelsOfAMetaImage)
{
  using ImageType = itk::Image<float, 3>;
  using RangeType = itk::ImageRegionRange<ImageType>;

  const std::string directory = itksys::SystemTools::GetCurrentWorkingDirectory();
  const auto        image = ImageType::New();
  image->SetRegions(itk::Size<3>::Filled(162));
  image->Allocate();
  image->SetSpacing(itk::MakeVector(0.5, 0.75, 1.25));
  image->SetOrigin(itk::MakePoint(-10.0, 4.0, 30.0));
  const RangeType pixels(*image, image->GetBufferedRegion());
  std::iota(pixels.begin(), pixels.end(), 0.0f);
  itk::WriteImage(image, directory + "/FileMappedPixels.mhd");
  itk::WriteImage(image, directory + "/FileMappedPixelsCompressed.mhd", true);
  EXPECT_EQ(itk::MapImageFile<ImageType>(directory + "/FileMappedPixels.mhd"), nullptr);

  itk::FileMappedImageContainerFactory::MapImagesFrom(directory);
  const itk::ObjectFactoryBase::Pointer factory = itk::ObjectFactoryBase::GetRegisteredFactories().back();
  {
    const auto mapped = itk::MapImageFile<ImageType>(directory + "/FileMappedPixels.mhd");
    ASSERT_NE(mapped, nullptr);
    ASSERT_EQ(mapped->GetBufferedRegion(), image->GetBufferedRegion());
    EXPECT_EQ(mapped->GetSpacing(), image->GetSpacing());
    EXPECT_EQ(mapped->GetOrigin(), image->GetOrigin());
    const RangeType read(*mapped, mapped->GetBufferedRegion());
    EXPECT_TRUE(std::equal(pixels.begin(), pixels.end(), read.begin()));
    // A write stays in memory: the file keeps its pixels.
    mapped->FillBuffer(-1.0f);
    const auto again = itk::MapImageFile<ImageType>(directory + "/FileMappedPixels.mhd");
    EXPECT_EQ(again->GetPixel(itk::Index<3>::Filled(161)), image->GetPixel(itk::Index<3>::Filled(161)));

    EXPECT_EQ(itk::MapImageFile<ImageType>(directory + "/FileMappedPixelsCompressed.mhd"), nullptr);
    EXPECT_EQ((itk::MapImageFile<itk::Image<short, 3>>(directory + "/FileMappedPixels.mhd")), nullptr);
  }
  itk::ObjectFactoryBase::UnRegisterFactory(factory);
}

#endif
