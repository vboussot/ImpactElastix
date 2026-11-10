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
#ifndef itkFrugalBSplineInterpolateImageFunction_h
#define itkFrugalBSplineInterpolateImageFunction_h

#include "itkBSplineInterpolateImageFunction.h"
#include "itkImageAlgorithm.h"
#include "itkImageLinearIteratorWithIndex.h"
#include "itkMultiThreaderBase.h"

#include <cmath>
#include <type_traits>
#include <vector>

namespace itk
{

/** \class FrugalBSplineInterpolateImageFunction
 * \brief A BSplineInterpolateImageFunction that computes its coefficients only when it has to,
 * on all the threads.
 *
 * The superclass computes a coefficient image each time an image is set, a line at a time on one
 * thread. Here:
 * \li At order 0 or 1 a B-spline has no poles and its coefficients are the pixels: the image is
 *     its own coefficient image when both have the same pixel type, and is only cast otherwise.
 * \li An image that shares its pixels with the image set before, as two levels of a pyramid may,
 *     has the coefficients computed for that one.
 * \li The lines of an axis do not depend on each other: they are filtered in parallel, each one
 *     as BSplineDecompositionImageFilter filters it.
 *
 * The values are those of the superclass.
 */
template <typename TImageType, typename TCoordinate = double, typename TCoefficientType = double>
class ITK_TEMPLATE_EXPORT FrugalBSplineInterpolateImageFunction
  : public BSplineInterpolateImageFunction<TImageType, TCoordinate, TCoefficientType>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(FrugalBSplineInterpolateImageFunction);

  using Self = FrugalBSplineInterpolateImageFunction;
  using Superclass = BSplineInterpolateImageFunction<TImageType, TCoordinate, TCoefficientType>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  itkNewMacro(Self);
  itkOverrideGetNameOfClassMacro(FrugalBSplineInterpolateImageFunction);

  using typename Superclass::CoefficientImageType;

  void
  SetInputImage(const TImageType * inputData) override
  {
    const Object * const pixels = inputData == nullptr ? nullptr : inputData->GetPixelContainer();
    const bool           pixelsAreCoefficients = std::is_same_v<TImageType, CoefficientImageType> && pixels != nullptr &&
                                       this->GetSplineOrder() <= 1;
    if (pixelsAreCoefficients)
    {
      if constexpr (std::is_same_v<TImageType, CoefficientImageType>)
      {
        this->m_Coefficients = inputData;
      }
    }
    else if (pixels == nullptr)
    {
      Superclass::SetInputImage(inputData);
      m_DecomposedPixels = nullptr;
      return;
    }
    else if (pixels != m_DecomposedPixels || pixels->GetMTime() != m_DecomposedTime ||
             this->GetSplineOrder() != m_DecomposedOrder)
    {
      this->Decompose(*inputData);
      m_DecomposedPixels = pixels;
      m_DecomposedTime = pixels->GetMTime();
      m_DecomposedOrder = this->GetSplineOrder();
    }
    // The coefficients are in place: the image is set as the superclass of the superclass sets it.
    if (pixelsAreCoefficients)
    {
      m_DecomposedPixels = nullptr;
    }
    this->m_DataLength = inputData->GetBufferedRegion().GetSize();
    Superclass::Superclass::SetInputImage(inputData);
  }

protected:
  FrugalBSplineInterpolateImageFunction() = default;
  ~FrugalBSplineInterpolateImageFunction() override = default;

private:
  /** The poles of the B-spline of an order, as BSplineDecompositionImageFilter has them. */
  static std::vector<double>
  Poles(const int order)
  {
    switch (order)
    {
      case 2:
        return { std::sqrt(8.0) - 3.0 };
      case 3:
        return { std::sqrt(3.0) - 2.0 };
      case 4:
        return { std::sqrt(664.0 - std::sqrt(438976.0)) + std::sqrt(304.0) - 19.0,
                 std::sqrt(664.0 + std::sqrt(438976.0)) - std::sqrt(304.0) - 19.0 };
      case 5:
        return { std::sqrt(135.0 / 2.0 - std::sqrt(17745.0 / 4.0)) + std::sqrt(105.0 / 4.0) - 13.0 / 2.0,
                 std::sqrt(135.0 / 2.0 + std::sqrt(17745.0 / 4.0)) - std::sqrt(105.0 / 4.0) - 13.0 / 2.0 };
      default:
        return {};
    }
  }

  /** The values of a line turned into its coefficients: BSplineDecompositionImageFilter's
   * DataToCoefficients1D, operation for operation, with its mirror boundaries and its tolerance.
   */
  static void
  DataToCoefficients(std::vector<double> & line, const std::vector<double> & poles)
  {
    const SizeValueType length = line.size();
    double              gain = 1.0;
    for (const double pole : poles)
    {
      gain = gain * (1.0 - pole) * (1.0 - 1.0 / pole);
    }
    for (double & value : line)
    {
      value *= gain;
    }
    for (const double z : poles)
    {
      // Causal initialization
      double     zn = z;
      const auto horizon = static_cast<SizeValueType>(std::ceil(std::log(1e-10) / std::log(itk::Math::abs(z))));
      if (horizon < length)
      {
        double sum = line[0];
        for (unsigned int n = 1; n < horizon; ++n)
        {
          sum += zn * line[n];
          zn *= z;
        }
        line[0] = sum;
      }
      else
      {
        const double iz = 1.0 / z;
        double       z2n = std::pow(z, static_cast<double>(length - 1L));
        double       sum = line[0] + z2n * line[length - 1L];
        z2n *= z2n * iz;
        for (unsigned int n = 1; n <= (length - 2); ++n)
        {
          sum += (zn + z2n) * line[n];
          zn *= z;
          z2n *= iz;
        }
        line[0] = sum / (1.0 - zn * zn);
      }
      // Causal recursion
      for (unsigned int n = 1; n < length; ++n)
      {
        line[n] += z * line[n - 1];
      }
      // Anticausal initialization and recursion
      line[length - 1] = (z / (z * z - 1.0)) * (z * line[length - 2] + line[length - 1]);
      for (int n = static_cast<int>(length) - 2; 0 <= n; n--)
      {
        line[n] = z * (line[n + 1] - line[n]);
      }
    }
  }

  /** Computes the coefficients of `image`: its pixels cast, then the lines of each axis filtered. */
  void
  Decompose(const TImageType & image)
  {
    using RegionType = typename TImageType::RegionType;
    using CoefficientType = typename CoefficientImageType::PixelType;

    const RegionType region = image.GetBufferedRegion();
    const auto       coefficients = CoefficientImageType::New();
    coefficients->CopyInformation(&image);
    coefficients->SetLargestPossibleRegion(image.GetLargestPossibleRegion());
    coefficients->SetBufferedRegion(region);
    coefficients->SetRequestedRegion(region);
    coefficients->Allocate();

    const auto threader = MultiThreaderBase::New();
    threader->template ParallelizeImageRegion<Superclass::ImageDimension>(
      region,
      [&](const RegionType & piece) { ImageAlgorithm::Copy(&image, coefficients.GetPointer(), piece, piece); },
      nullptr);

    const std::vector<double> poles = Poles(this->GetSplineOrder());
    for (unsigned int axis = 0; axis < Superclass::ImageDimension && !poles.empty(); ++axis)
    {
      const SizeValueType length = region.GetSize(axis);
      if (length == 1)
      {
        continue; // Required by mirror boundaries
      }
      threader->template ParallelizeImageRegionRestrictDirection<Superclass::ImageDimension>(
        axis,
        region,
        [&](const RegionType & piece) {
          ImageLinearIteratorWithIndex<CoefficientImageType> iterator(coefficients, piece);
          iterator.SetDirection(axis);
          std::vector<double> line(length);
          for (; !iterator.IsAtEnd(); iterator.NextLine())
          {
            for (double & value : line)
            {
              value = static_cast<double>(iterator.Get());
              ++iterator;
            }
            DataToCoefficients(line, poles);
            iterator.GoToBeginOfLine();
            for (const double value : line)
            {
              iterator.Set(static_cast<CoefficientType>(value));
              ++iterator;
            }
          }
        },
        nullptr);
    }
    this->m_Coefficients = coefficients;
  }

  /** The pixels the coefficients were computed from, when, and at which order. */
  const Object *   m_DecomposedPixels{};
  ModifiedTimeType m_DecomposedTime{};
  int              m_DecomposedOrder{};
};

} // end namespace itk

#endif
