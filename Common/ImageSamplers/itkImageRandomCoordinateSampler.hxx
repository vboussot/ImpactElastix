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
#ifndef itkImageRandomCoordinateSampler_hxx
#define itkImageRandomCoordinateSampler_hxx

#include "itkImageRandomCoordinateSampler.h"
#include <itkDeref.h>
#include <vnl/vnl_math.h>
#include <cassert>

namespace itk
{

/**
 * ******************* GenerateData *******************
 */

template <typename TInputImage>
void
ImageRandomCoordinateSampler<TInputImage>::GenerateData()
{
  /** Get handles to the input image, output sample container, and interpolator. */
  const InputImageType &             inputImage = Deref(this->GetInput());
  auto &                             samples = Deref(this->GetOutput()).CastToSTLContainer();
  typename InterpolatorType::Pointer interpolator = this->GetModifiableInterpolator();

  /** Set up the interpolator. */
  interpolator->SetInputImage(&inputImage); // only once?

  const auto croppedInputImageRegion = this->GetCroppedInputImageRegion();

  /** Convert inputImageRegion to bounding box in physical space. */
  const auto                          unitSize = InputImageSizeType::Filled(1);
  const InputImageIndexType           smallestIndex = croppedInputImageRegion.GetIndex();
  const InputImageIndexType           largestIndex = smallestIndex + croppedInputImageRegion.GetSize() - unitSize;
  const InputImageContinuousIndexType smallestImageContIndex(smallestIndex);
  const InputImageContinuousIndexType largestImageContIndex(largestIndex);
  InputImageContinuousIndexType       smallestContIndex;
  InputImageContinuousIndexType       largestContIndex;
  this->GenerateSampleRegion(smallestImageContIndex, largestImageContIndex, smallestContIndex, largestContIndex);

  samples.resize(this->Superclass::m_NumberOfSamples);

  /** Get a handle to the mask. */
  const MaskType * const mask = this->Superclass::GetMask();

  /** The coordinates are drawn first, those inside the mask when there is one; the image is then
   * interpolated at them, in parallel when multi-threaded, each one a place to wait for when the
   * image is not in memory.
   */
  m_RandomCoordinates.clear();
  m_RandomCoordinates.reserve(this->m_NumberOfSamples);

  if (mask != nullptr)
  {
    mask->UpdateSource();
  }
  const unsigned long maximumNumberOfSamplesToTry = 10 * this->GetNumberOfSamples();
  unsigned long       numberOfSamplesTried = 0;
  InputImagePointType point;
  while (m_RandomCoordinates.size() < this->m_NumberOfSamples)
  {
    /** Check if we are not trying eternally to find a valid point. */
    if (mask != nullptr && ++numberOfSamplesTried > maximumNumberOfSamplesToTry)
    {
      samples.resize(m_RandomCoordinates.size());
      itkExceptionMacro("Could not find enough image samples within reasonable time. Probably the mask is too small");
    }
    InputImageContinuousIndexType randomCIndex;
    this->GenerateRandomCoordinate(smallestContIndex, largestContIndex, randomCIndex);
    if (mask != nullptr)
    {
      inputImage.TransformContinuousIndexToPhysicalPoint(randomCIndex, point);
      if (!interpolator->IsInsideBuffer(randomCIndex) || !mask->IsInsideInWorldSpace(point))
      {
        continue;
      }
    }
    m_RandomCoordinates.push_back(randomCIndex);
  }

  if (Superclass::m_UseMultiThread)
  {
    UserData userData{ m_RandomCoordinates, inputImage, *interpolator, samples };

    Deref(this->ProcessObject::GetMultiThreader()).SetSingleMethodAndExecute(&Self::ThreaderCallback, &userData);
    return;
  }

  for (size_t i = 0; i < samples.size(); ++i)
  {
    const InputImageContinuousIndexType & sampleCIndex = m_RandomCoordinates[i];
    inputImage.TransformContinuousIndexToPhysicalPoint(sampleCIndex, samples[i].m_ImageCoordinates);
    samples[i].m_ImageValue = static_cast<ImageSampleValueType>(interpolator->EvaluateAtContinuousIndex(sampleCIndex));
  }

} // end GenerateData()


/**
 * ******************* ThreaderCallback *******************
 */

template <typename TInputImage>
ITK_THREAD_RETURN_FUNCTION_CALL_CONVENTION
ImageRandomCoordinateSampler<TInputImage>::ThreaderCallback(void * const arg)
{
  assert(arg);
  const auto & info = *static_cast<const MultiThreaderBase::WorkUnitInfo *>(arg);

  assert(info.UserData);
  auto & userData = *static_cast<UserData *>(info.UserData);

  const auto & randomCoordinates = userData.m_RandomCoordinates;
  auto &       samples = userData.m_Samples;
  const auto & interpolator = userData.m_Interpolator;

  const auto totalNumberOfSamples = samples.size();
  assert(totalNumberOfSamples == randomCoordinates.size());

  const auto numberOfSamplesPerWorkUnit = totalNumberOfSamples / info.NumberOfWorkUnits;
  const auto remainderNumberOfSamples = totalNumberOfSamples % info.NumberOfWorkUnits;

  const auto offset =
    info.WorkUnitID * numberOfSamplesPerWorkUnit + std::min<size_t>(info.WorkUnitID, remainderNumberOfSamples);
  const auto beginOfRandomCoordinates = randomCoordinates.data() + offset;
  const auto beginOfSamples = samples.data() + offset;

  const auto & inputImage = userData.m_InputImage;

  const size_t n{ numberOfSamplesPerWorkUnit + (info.WorkUnitID < remainderNumberOfSamples ? 1 : 0) };

  for (size_t i = 0; i < n; ++i)
  {
    auto &                              sample = beginOfSamples[i];
    const InputImageContinuousIndexType sampleCIndex = beginOfRandomCoordinates[i];

    /** Convert to point */
    inputImage.TransformContinuousIndexToPhysicalPoint(sampleCIndex, sample.m_ImageCoordinates);

    /** Compute the value at the continuous index. */
    sample.m_ImageValue = static_cast<ImageSampleValueType>(interpolator.EvaluateAtContinuousIndex(sampleCIndex));

  } // end for loop

  return ITK_THREAD_RETURN_DEFAULT_VALUE;
}


/**
 * ******************* GenerateRandomCoordinate *******************
 */

template <typename TInputImage>
void
ImageRandomCoordinateSampler<TInputImage>::GenerateRandomCoordinate(
  const InputImageContinuousIndexType & smallestContIndex,
  const InputImageContinuousIndexType & largestContIndex,
  InputImageContinuousIndexType &       randomContIndex)
{
  Statistics::MersenneTwisterRandomVariateGenerator & randomVariateGenerator = Superclass::GetRandomVariateGenerator();

  for (unsigned int i = 0; i < InputImageDimension; ++i)
  {
    randomContIndex[i] = static_cast<InputImagePointValueType>(
      randomVariateGenerator.GetUniformVariate(smallestContIndex[i], largestContIndex[i]));
  }
} // end GenerateRandomCoordinate()


/**
 * ******************* GenerateSampleRegion *******************
 */

template <typename TInputImage>
void
ImageRandomCoordinateSampler<TInputImage>::GenerateSampleRegion(
  const InputImageContinuousIndexType & smallestImageContIndex,
  const InputImageContinuousIndexType & largestImageContIndex,
  InputImageContinuousIndexType &       smallestContIndex,
  InputImageContinuousIndexType &       largestContIndex)
{
  if (!this->GetUseRandomSampleRegion())
  {
    smallestContIndex = smallestImageContIndex;
    largestContIndex = largestImageContIndex;
    return;
  }

  /** Convert sampleRegionSize to continuous index space and
   * compute the maximum allowed value for the smallestContIndex,
   * such that a sample region of size SampleRegionSize still fits.
   */
  using CIndexVectorType = typename InputImageContinuousIndexType::VectorType;
  CIndexVectorType              sampleRegionSize;
  InputImageContinuousIndexType maxSmallestContIndex;
  for (unsigned int i = 0; i < InputImageDimension; ++i)
  {
    sampleRegionSize[i] = this->GetSampleRegionSize()[i] / this->GetInput()->GetSpacing()[i];
    maxSmallestContIndex[i] = largestImageContIndex[i] - sampleRegionSize[i];

    /** Make sure it is larger than the lower bound. */
    maxSmallestContIndex[i] = std::max(maxSmallestContIndex[i], smallestImageContIndex[i]);
  }

  this->GenerateRandomCoordinate(smallestImageContIndex, maxSmallestContIndex, smallestContIndex);
  largestContIndex = smallestContIndex;
  largestContIndex += sampleRegionSize;

} // end GenerateSampleRegion()


/**
 * ******************* PrintSelf *******************
 */

template <typename TInputImage>
void
ImageRandomCoordinateSampler<TInputImage>::PrintSelf(std::ostream & os, Indent indent) const
{
  Superclass::PrintSelf(os, indent);

  os << indent << "Interpolator: " << this->m_Interpolator.GetPointer() << std::endl;

} // end PrintSelf()


} // end namespace itk

#endif // end #ifndef itkImageRandomCoordinateSampler_hxx
