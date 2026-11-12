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
#ifndef itkFrugalDiscreteGaussianImageFilter_h
#define itkFrugalDiscreteGaussianImageFilter_h

#include "itkGaussianOperator.h"
#include "itkImageToImageFilter.h"
#include "itkIndexRange.h"
#include "itkNumericTraits.h"

#include <algorithm>
#include <vector>

namespace itk
{

/** \class FrugalDiscreteGaussianImageFilter
 * \brief The smoothing of DiscreteGaussianImageFilter with a variance in pixels, a line at a time.
 *
 * DiscreteGaussianImageFilter smooths along the last axis, then the one before, down to the first,
 * each with a NeighborhoodOperatorImageFilter, which visits the pixels through a neighborhood
 * iterator and keeps an image between two axes. Here the same sums are taken a line of the first
 * axis at a time, term after term for the whole line, and what an axis leaves for the next is
 * kept for a few slices only.
 *
 * The pixels are those of DiscreteGaussianImageFilter without UseImageSpacing: the kernels are
 * its GaussianOperator's, a pixel is the sum of its terms in their order, in the same types, cast
 * to the pixel type after each axis, and a neighbor outside the image is the nearest pixel inside.
 */
template <typename TImage>
class ITK_TEMPLATE_EXPORT FrugalDiscreteGaussianImageFilter : public ImageToImageFilter<TImage, TImage>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(FrugalDiscreteGaussianImageFilter);

  using Self = FrugalDiscreteGaussianImageFilter;
  using Superclass = ImageToImageFilter<TImage, TImage>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  itkNewMacro(Self);
  itkOverrideGetNameOfClassMacro(FrugalDiscreteGaussianImageFilter);

  static constexpr unsigned int ImageDimension = TImage::ImageDimension;

  using ArrayType = FixedArray<double, ImageDimension>;

  /** The variance of the Gaussian along each axis, in pixels. */
  itkSetMacro(Variance, ArrayType);
  itkGetConstMacro(Variance, ArrayType);

  /** The error a truncated kernel may have, as DiscreteGaussianImageFilter has it. */
  itkSetMacro(MaximumError, double);
  itkGetConstMacro(MaximumError, double);

protected:
  FrugalDiscreteGaussianImageFilter() { this->DynamicMultiThreadingOn(); }
  ~FrugalDiscreteGaussianImageFilter() override = default;

  using PixelType = typename TImage::PixelType;
  using RegionType = typename TImage::RegionType;
  using IndexType = typename TImage::IndexType;
  using RealType = typename NumericTraits<PixelType>::RealType;
  using AccumulateType = typename NumericTraits<RealType>::AccumulateType;
  using KernelValueType = typename NumericTraits<RealType>::ValueType;
  using KernelType = std::vector<KernelValueType>;

  /** The input needs the output region and the radius of each kernel around it. */
  void
  GenerateInputRequestedRegion() override
  {
    Superclass::GenerateInputRequestedRegion();
    if (const auto input = const_cast<TImage *>(this->GetInput()))
    {
      RegionType region = input->GetRequestedRegion();
      for (unsigned int axis = 0; axis < ImageDimension; ++axis)
      {
        Pad(region, axis, this->Kernel(axis).size() / 2);
      }
      region.Crop(input->GetLargestPossibleRegion());
      input->SetRequestedRegion(region);
    }
  }

  void
  BeforeThreadedGenerateData() override
  {
    for (unsigned int axis = 0; axis < ImageDimension; ++axis)
    {
      m_Kernels[axis] = this->Kernel(axis);
    }
  }

  void
  DynamicThreadedGenerateData(const RegionType & outputRegionForThread) override
  {
    const TImage &         input = *this->GetInput();
    TImage &               output = *this->GetOutput();
    const RegionType       largest = input.GetLargestPossibleRegion();
    constexpr auto         last = ImageDimension - 1;
    std::vector<PixelType> scratch[2];

    // A few slices at a time, for what an axis leaves for the next to stay in the cache.
    const SizeValueType pixelsPerSlice =
      outputRegionForThread.GetNumberOfPixels() / outputRegionForThread.GetSize(last);
    const SizeValueType slicesPerSlab = std::max<SizeValueType>((SizeValueType{ 1 } << 18) / pixelsPerSlice, 1);
    for (SizeValueType first = 0; first < outputRegionForThread.GetSize(last); first += slicesPerSlab)
    {
      RegionType slab = outputRegionForThread;
      slab.SetIndex(last, outputRegionForThread.GetIndex(last) + first);
      slab.SetSize(last, std::min(slicesPerSlab, outputRegionForThread.GetSize(last) - first));

      // From the last axis to the first: an axis writes the slab and what the axes after it read around it.
      Buffer source{ input.GetBufferPointer(), input.GetBufferedRegion() };
      for (unsigned int axis = ImageDimension; axis-- > 0;)
      {
        RegionType written = slab;
        for (unsigned int before = 0; before < axis; ++before)
        {
          Pad(written, before, m_Kernels[before].size() / 2);
        }
        written.Crop(largest);
        Buffer destination{ output.GetBufferPointer(), output.GetBufferedRegion() };
        if (axis > 0)
        {
          scratch[axis % 2].resize(written.GetNumberOfPixels());
          destination = { scratch[axis % 2].data(), written };
        }
        Convolve(source, destination, written, axis, m_Kernels[axis], largest);
        source = destination;
      }
    }
  }

private:
  /** Pixels in memory and the region they are the pixels of. */
  struct Buffer
  {
    const PixelType * Pixels;
    RegionType        Region;

    const PixelType *
    At(const IndexType & index) const
    {
      OffsetValueType offset = 0;
      OffsetValueType stride = 1;
      for (unsigned int axis = 0; axis < ImageDimension; ++axis)
      {
        offset += (index[axis] - Region.GetIndex(axis)) * stride;
        stride *= Region.GetSize(axis);
      }
      return Pixels + offset;
    }
  };

  static void
  Pad(RegionType & region, const unsigned int axis, const SizeValueType radius)
  {
    region.SetIndex(axis, region.GetIndex(axis) - static_cast<IndexValueType>(radius));
    region.SetSize(axis, region.GetSize(axis) + 2 * radius);
  }

  /** The kernel of an axis: the coefficients of the operator DiscreteGaussianImageFilter makes. */
  KernelType
  Kernel(const unsigned int axis) const
  {
    GaussianOperator<KernelValueType, ImageDimension> oper;
    oper.SetDirection(axis);
    oper.SetMaximumError(m_MaximumError);
    oper.SetMaximumKernelWidth(32);
    oper.SetVariance(m_Variance[axis]);
    oper.CreateDirectional();
    return KernelType(oper.Begin(), oper.End());
  }

  /** Writes the region `written` of `destination`: `source` convolved with a kernel along an axis,
   * a neighbor outside `largest` being the nearest pixel inside. The sum of a pixel is that of
   * NeighborhoodInnerProduct; the lines of the first axis take a term of it at a time.
   */
  static void
  Convolve(const Buffer &     source,
           const Buffer &     destination,
           const RegionType & written,
           const unsigned int axis,
           const KernelType & kernel,
           const RegionType & largest)
  {
    const IndexValueType radius = kernel.size() / 2;
    const IndexValueType lowest = largest.GetIndex(axis);
    const IndexValueType highest = lowest + static_cast<IndexValueType>(largest.GetSize(axis)) - 1;
    const SizeValueType  length = written.GetSize(0);

    std::vector<AccumulateType> sums(length);
    std::vector<RealType>       line(axis == 0 ? length + 2 * radius : 0);
    // The lines of the region: its indices where the first axis starts.
    RegionType lines = written;
    lines.SetSize(0, 1);
    for (const IndexType index : ImageRegionIndexRange<ImageDimension>(lines))
    {
      std::fill(sums.begin(), sums.end(), AccumulateType{});
      if (axis == 0)
      {
        // The line and its neighbors at both ends.
        IndexType at = index;
        for (SizeValueType j = 0; j < line.size(); ++j)
        {
          at[0] = std::clamp(index[0] - radius + static_cast<IndexValueType>(j), lowest, highest);
          line[j] = static_cast<RealType>(*source.At(at));
        }
        for (SizeValueType k = 0; k < kernel.size(); ++k)
        {
          const KernelValueType  coefficient = kernel[k];
          const RealType * const neighbors = line.data() + k;
          for (SizeValueType x = 0; x < length; ++x)
          {
            sums[x] += static_cast<AccumulateType>(coefficient * neighbors[x]);
          }
        }
      }
      else
      {
        IndexType at = index;
        for (SizeValueType k = 0; k < kernel.size(); ++k)
        {
          at[axis] = std::clamp(index[axis] - radius + static_cast<IndexValueType>(k), lowest, highest);
          const KernelValueType   coefficient = kernel[k];
          const PixelType * const neighbors = source.At(at);
          for (SizeValueType x = 0; x < length; ++x)
          {
            sums[x] += static_cast<AccumulateType>(coefficient * static_cast<RealType>(neighbors[x]));
          }
        }
      }
      PixelType * const pixels = const_cast<PixelType *>(destination.At(index));
      for (SizeValueType x = 0; x < length; ++x)
      {
        pixels[x] = static_cast<PixelType>(static_cast<KernelValueType>(sums[x]));
      }
    }
  }

  ArrayType  m_Variance{};
  double     m_MaximumError{ 0.01 };
  KernelType m_Kernels[ImageDimension];
};

} // end namespace itk

#endif
