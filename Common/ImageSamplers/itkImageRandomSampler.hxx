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
#ifndef itkImageRandomSampler_hxx
#define itkImageRandomSampler_hxx

#include "itkImageRandomSampler.h"

#include "itkImageRandomConstIteratorWithIndex.h"
#include <itkDeref.h>
#include <algorithm>
#include <cassert>
#include <vector>

namespace itk
{

/**
 * ******************* GenerateData *******************
 */

template <typename TInputImage>
void
ImageRandomSampler<TInputImage>::GenerateData()
{
  /** Get handles to the input image, output sample container. */
  const InputImageType & inputImage = Deref(this->GetInput());
  auto &                 samples = Deref(this->GetOutput()).CastToSTLContainer();

  /** Get a handle to the mask. If there was no mask supplied we exercise a multi-threaded version. */
  const MaskType * const mask = this->Superclass::GetMask();
  if (mask == nullptr && Superclass::m_UseMultiThread)
  {
    Superclass::GenerateRandomNumberList();
    const auto & randomNumberList = Superclass::m_RandomNumberList;
    samples.resize(randomNumberList.size());

    const auto & croppedInputImageRegion = this->GetCroppedInputImageRegion();
    UserData     userData{
      randomNumberList, inputImage, croppedInputImageRegion.GetIndex(), croppedInputImageRegion.GetSize(), samples
    };

    Deref(this->ProcessObject::GetMultiThreader()).SetSingleMethodAndExecute(&Self::ThreaderCallback, &userData);
    return;
  }

  /** Reserve memory for the output. */
  samples.resize(this->GetNumberOfSamples());

  /** Setup a random iterator over the input image. */
  ImageRandomConstIteratorWithIndex<InputImageType> randIter(&inputImage, this->GetCroppedInputImageRegion());

  /** Seeded as the multi-threaded version: left to itself, the iterator takes its seed from the clock. */
  randIter.ReinitializeSeed(this->NextSeed());
  randIter.GoToBegin();

  if (mask == nullptr)
  {
    /** number of samples + 1, because of the initial ++randIter. */
    randIter.SetNumberOfSamples(this->GetNumberOfSamples() + 1);
    /** Advance one, in order to generate the same sequence as when using a mask */
    ++randIter;
    for (auto & sample : samples)
    {
      /** Get the index, transform it to the physical coordinates and put it in the sample. */
      InputImageIndexType index = randIter.GetIndex();
      inputImage.TransformIndexToPhysicalPoint(index, sample.m_ImageCoordinates);
      /** Get the value and put it in the sample. */
      sample.m_ImageValue = randIter.Get();
      /** Jump to a random position. */
      ++randIter;

    } // end for loop
  } // end if no mask
  else
  {
    /** Update the mask. */
    mask->UpdateSource();

    /** Make sure we are not eternally trying to find samples: */
    randIter.SetNumberOfSamples(10 * this->GetNumberOfSamples());

    /** The samples are the first positions of the random sequence inside the mask. The positions
     * are drawn a batch at a time, and those of a batch are looked up in the mask and in the image
     * in parallel: they are places apart, each one a page to wait for when the image is not in memory.
     */
    std::vector<InputImageIndexType> indices;
    std::vector<ImageSampleType>     candidates;
    std::vector<char>                inside;
    std::size_t                      numberOfSamples = 0;
    std::size_t                      numberOfCandidates = 0;
    std::size_t                      numberInside = 0;
    while (numberOfSamples < samples.size())
    {
      /** As many positions as the samples left take at the rate of the positions inside so far. */
      const std::size_t left = samples.size() - numberOfSamples;
      const std::size_t batch =
        16 + (numberInside == 0 ? left : static_cast<std::size_t>(1.1 * left * numberOfCandidates / numberInside));
      indices.clear();
      while (indices.size() < batch && !(++randIter).IsAtEnd())
      {
        indices.push_back(randIter.GetIndex());
      }
      candidates.resize(indices.size());
      inside.resize(indices.size());
      const auto lookUp = [&](const SizeValueType i) {
        inputImage.TransformIndexToPhysicalPoint(indices[i], candidates[i].m_ImageCoordinates);
        inside[i] = mask->IsInsideInWorldSpace(candidates[i].m_ImageCoordinates);
        if (inside[i])
        {
          candidates[i].m_ImageValue = inputImage.GetPixel(indices[i]);
        }
      };
      if (Superclass::m_UseMultiThread)
      {
        Deref(this->ProcessObject::GetMultiThreader()).ParallelizeArray(0, indices.size(), lookUp, nullptr);
      }
      else
      {
        for (SizeValueType i = 0; i < indices.size(); ++i)
        {
          lookUp(i);
        }
      }
      for (std::size_t i = 0; i < indices.size() && numberOfSamples < samples.size(); ++i)
      {
        if (inside[i])
        {
          samples[numberOfSamples++] = candidates[i];
        }
      }
      numberOfCandidates += indices.size();
      numberInside += std::count(inside.begin(), inside.end(), 1);

      /** Check if we are not trying eternally to find a valid point. */
      if (randIter.IsAtEnd() && numberOfSamples < samples.size())
      {
        /** Squeeze the sample container to the size that is still valid. */
        samples.resize(numberOfSamples);
        itkExceptionMacro("Could not find enough image samples within reasonable time. Probably the mask is too small");
      }
    }

    /** Extra random sample to make sure the same sequence is generated
     * with and without mask.
     */
    ++randIter;
  }

} // end GenerateData()


template <typename TInputImage>
ITK_THREAD_RETURN_FUNCTION_CALL_CONVENTION
ImageRandomSampler<TInputImage>::ThreaderCallback(void * const arg)
{
  assert(arg);
  const auto & info = *static_cast<const MultiThreaderBase::WorkUnitInfo *>(arg);

  assert(info.UserData);
  auto & userData = *static_cast<UserData *>(info.UserData);

  const auto & randomNumberList = userData.m_RandomNumberList;
  auto &       samples = userData.m_Samples;

  const auto totalNumberOfSamples = samples.size();
  assert(totalNumberOfSamples == randomNumberList.size());

  const auto numberOfSamplesPerWorkUnit = totalNumberOfSamples / info.NumberOfWorkUnits;
  const auto remainderNumberOfSamples = totalNumberOfSamples % info.NumberOfWorkUnits;

  const auto offset =
    info.WorkUnitID * numberOfSamplesPerWorkUnit + std::min<size_t>(info.WorkUnitID, remainderNumberOfSamples);
  const auto   beginOfRandomNumbers = randomNumberList.data() + offset;
  const auto   beginOfSamples = samples.data() + offset;
  const auto & inputImage = userData.m_InputImage;

  const InputImageSizeType  regionSize = userData.m_RegionSize;
  const InputImageIndexType regionIndex = userData.m_RegionIndex;

  const size_t n{ numberOfSamplesPerWorkUnit + (info.WorkUnitID < remainderNumberOfSamples ? 1 : 0) };

  for (size_t i = 0; i < n; ++i)
  {
    auto   randomPosition = static_cast<size_t>(beginOfRandomNumbers[i]);
    auto & sample = beginOfSamples[i];

    /** Translate randomPosition to an index, copied from ImageRandomConstIteratorWithIndex. */
    InputImageIndexType positionIndex;

    for (unsigned int dim = 0; dim < InputImageDimension; ++dim)
    {
      const auto sizeInThisDimension = regionSize[dim];
      const auto residual = randomPosition % sizeInThisDimension;
      positionIndex[dim] = static_cast<IndexValueType>(residual) + regionIndex[dim];
      randomPosition -= residual;
      randomPosition /= sizeInThisDimension;
    }

    /** Transform index to the physical coordinates and put it in the sample. */
    inputImage.TransformIndexToPhysicalPoint(positionIndex, sample.m_ImageCoordinates);

    /** Get the value and put it in the sample. */
    sample.m_ImageValue = static_cast<ImageSampleValueType>(inputImage.GetPixel(positionIndex));
  }
  return ITK_THREAD_RETURN_DEFAULT_VALUE;
}


} // end namespace itk

#endif // end #ifndef itkImageRandomSampler_hxx
