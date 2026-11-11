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
#ifndef itkStreamedRecursivePyramidImageFilter_h
#define itkStreamedRecursivePyramidImageFilter_h

#include "itkCastImageFilter.h"
#include "itkFrugalDiscreteGaussianImageFilter.h"
#include "itkIdentityTransform.h"
#include "itkLinearInterpolateImageFunction.h"
#include "itkRecursiveMultiResolutionPyramidImageFilter.h"
#include "itkResampleImageFilter.h"
#include "itkShrinkImageFilter.h"
#include "itkStreamingImageFilter.h"

#include <cstddef>
#include <deque>
#include <mutex>
#include <type_traits>
#include <vector>

namespace itk
{

/** \class StreamedRecursivePyramidOptions
 * \brief What a process asks of the recursive pyramids it computes.
 */
struct StreamedRecursivePyramidOptions
{
  /** Whether a pyramid keeps its smaller levels for the next pyramid of the same image, schedule
   * and settings. For a process whose images are those of files, as elastix run from the command
   * line: each parameter file computes the same pyramids.
   */
  static bool &
  KeepsLevels()
  {
    static bool keeps = false;
    return keeps;
  }

  /** Whether a level is smoothed and shrunk a piece of its source at a time. For images that are
   * not in memory: the intermediate images are of the size of a piece, and the source is read
   * once, in order. In memory one piece is faster.
   */
  static bool &
  InPieces()
  {
    static bool pieces = false;
    return pieces;
  }
};


/** \class StreamedRecursivePyramidImageFilter
 * \brief The pyramid of RecursiveMultiResolutionPyramidImageFilter, without its copies.
 *
 * The levels are those of the superclass, to the bit. A level of the size of the one it comes
 * from shares its pixels, where the superclass copies them. StreamedRecursivePyramidOptions
 * asks for more: the smaller levels computed PieceSize bytes of their source at a time, and kept
 * for the next pyramid of the same image.
 */
template <typename TInputImage, typename TOutputImage>
class ITK_TEMPLATE_EXPORT StreamedRecursivePyramidImageFilter
  : public RecursiveMultiResolutionPyramidImageFilter<TInputImage, TOutputImage>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(StreamedRecursivePyramidImageFilter);

  using Self = StreamedRecursivePyramidImageFilter;
  using Superclass = RecursiveMultiResolutionPyramidImageFilter<TInputImage, TOutputImage>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  itkNewMacro(Self);
  itkOverrideGetNameOfClassMacro(StreamedRecursivePyramidImageFilter);

  itkStaticConstMacro(ImageDimension, unsigned int, TInputImage::ImageDimension);

  /** The size in bytes of the piece of a level smoothed and shrunk at a time. */
  static constexpr std::size_t PieceSize = std::size_t{ 64 } << 20;

protected:
  StreamedRecursivePyramidImageFilter() = default;
  ~StreamedRecursivePyramidImageFilter() override = default;

  void
  GenerateData() override
  {
    const auto & schedule = this->GetSchedule();
    const int    numberOfLevels = static_cast<int>(this->GetNumberOfLevels());
    bool         whole = this->IsScheduleDownwardDivisible(schedule);
    for (int level = 0; level < numberOfLevels; ++level)
    {
      whole = whole && this->GetOutput(level)->GetRequestedRegion() == this->GetOutput(level)->GetLargestPossibleRegion();
    }
    if (!whole)
    {
      Superclass::GenerateData();
      return;
    }

    // The levels kept from the pyramid of this image, if it had this schedule.
    const TInputImage * const         input = this->GetInput();
    const bool                   keeps = StreamedRecursivePyramidOptions::KeepsLevels();
    std::unique_lock<std::mutex> lock(Mutex(), std::defer_lock);
    const Levels *               kept = nullptr;
    if (keeps)
    {
      lock.lock();
      for (const Levels & levels : Kept())
      {
        if (levels.input == input && levels.time == input->GetMTime() && levels.buffer == input->GetBufferPointer() &&
            levels.schedule == schedule && levels.shrinks == this->GetUseShrinkImageFilter() &&
            levels.maximumError == this->GetMaximumError())
        {
          kept = &levels;
        }
      }
    }
    Levels computed{ input,
                     input->GetMTime(),
                     input->GetBufferPointer(),
                     schedule,
                     this->GetUseShrinkImageFilter(),
                     this->GetMaximumError(),
                     std::vector<typename TOutputImage::Pointer>(numberOfLevels) };

    // The image a level comes from: the input, then the level before.
    typename TOutputImage::Pointer source;
    if constexpr (std::is_same_v<TInputImage, TOutputImage>)
    {
      source = const_cast<TInputImage *>(this->GetInput());
    }
    else
    {
      const auto caster = CastImageFilter<TInputImage, TOutputImage>::New();
      caster->SetInput(this->GetInput());
      caster->Update();
      source = caster->GetOutput();
      source->DisconnectPipeline();
    }

    for (int level = numberOfLevels - 1; level >= 0; --level)
    {
      unsigned int factors[ImageDimension];
      double       variance[ImageDimension];
      bool         allOnes = true;
      for (unsigned int dim = 0; dim < ImageDimension; ++dim)
      {
        factors[dim] =
          level == numberOfLevels - 1 ? schedule[level][dim] : schedule[level][dim] / schedule[level + 1][dim];
        variance[dim] = factors[dim] == 1 ? 0.0 : Math::sqr(0.5 * static_cast<float>(factors[dim]));
        allOnes = allOnes && factors[dim] == 1;
      }
      if (kept != nullptr && kept->images[level] != nullptr)
      {
        source = kept->images[level];
      }
      else if (!allOnes)
      {
        const auto smoother = FrugalDiscreteGaussianImageFilter<TOutputImage>::New();
        smoother->SetMaximumError(this->GetMaximumError());
        smoother->SetVariance(variance);
        smoother->SetInput(source);

        typename ImageToImageFilter<TOutputImage, TOutputImage>::Pointer shrinker;
        if (this->GetUseShrinkImageFilter())
        {
          const auto filter = ShrinkImageFilter<TOutputImage, TOutputImage>::New();
          filter->SetShrinkFactors(factors);
          shrinker = filter;
        }
        else
        {
          const auto filter = ResampleImageFilter<TOutputImage, TOutputImage>::New();
          filter->SetInterpolator(LinearInterpolateImageFunction<TOutputImage, double>::New());
          filter->SetDefaultPixelValue(0);
          filter->SetTransform(IdentityTransform<double, ImageDimension>::New());
          filter->SetOutputParametersFromImage(this->GetOutput(level));
          shrinker = filter;
        }
        shrinker->SetInput(smoother->GetOutput());

        typename ImageSource<TOutputImage>::Pointer last = shrinker;
        if (StreamedRecursivePyramidOptions::InPieces())
        {
          const auto streamer = StreamingImageFilter<TOutputImage, TOutputImage>::New();
          streamer->SetInput(shrinker->GetOutput());
          streamer->SetNumberOfStreamDivisions(
            1 + source->GetLargestPossibleRegion().GetNumberOfPixels() * sizeof(typename TOutputImage::PixelType) /
                  PieceSize);
          last = streamer;
        }
        // The update leaves the region it asked for requested from the source, which may be the input.
        const auto requested = source->GetRequestedRegion();
        last->UpdateLargestPossibleRegion();
        source->SetRequestedRegion(requested);

        source = last->GetOutput();
        source->DisconnectPipeline();
        source->SetLargestPossibleRegion(this->GetOutput(level)->GetLargestPossibleRegion());
      }
      this->GraftNthOutput(level, source);
      // A level that shares the pixels of the input is the input: it is not kept.
      if (source->GetBufferPointer() != static_cast<const void *>(input->GetBufferPointer()))
      {
        computed.images[level] = source;
      }
    }
    if (keeps && kept == nullptr)
    {
      // The pyramids of a registration: of its fixed image and of its moving one.
      Kept().push_front(std::move(computed));
      if (Kept().size() > 2)
      {
        Kept().pop_back();
      }
    }
  }

private:
  /** The smaller levels of an image, with what they were computed from. */
  struct Levels
  {
    const TInputImage *                         input;
    ModifiedTimeType                            time;
    const void *                                buffer;
    typename Superclass::ScheduleType           schedule;
    bool                                        shrinks;
    double                                      maximumError;
    std::vector<typename TOutputImage::Pointer> images;
  };

  static std::deque<Levels> &
  Kept()
  {
    static std::deque<Levels> kept;
    return kept;
  }
  static std::mutex &
  Mutex()
  {
    static std::mutex mutex;
    return mutex;
  }
};

} // end namespace itk

#endif
