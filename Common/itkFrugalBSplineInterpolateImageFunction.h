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

#include <algorithm>
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
 * \li At order 1 or 3 a value and its derivative read the coefficients around a point once,
 *     straight from their buffer.
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
    const bool           pixelsAreCoefficients =
      std::is_same_v<TImageType, CoefficientImageType> && pixels != nullptr && this->GetSplineOrder() <= 1;
    if (pixelsAreCoefficients)
    {
      if constexpr (std::is_same_v<TImageType, CoefficientImageType>)
      {
        this->m_Coefficients = inputData;
      }
      m_DecomposedPixels = nullptr;
    }
    else if (pixels == nullptr)
    {
      Superclass::SetInputImage(inputData);
      m_DecomposedPixels = nullptr;
      return;
    }
    else if (pixels != m_DecomposedPixels || pixels->GetMTime() != m_DecomposedTime ||
             this->GetSplineOrder() != m_DecomposedOrder ||
             (inputData == this->GetInputImage() && inputData->GetMTime() != m_InputTime))
    {
      this->Decompose(*inputData);
      m_DecomposedPixels = pixels;
      m_DecomposedTime = pixels->GetMTime();
      m_DecomposedOrder = this->GetSplineOrder();
    }
    // The coefficients are in place: the image is set as the superclass of the superclass sets it.
    m_InputTime = inputData->GetMTime();
    this->m_DataLength = inputData->GetBufferedRegion().GetSize();
    Superclass::Superclass::SetInputImage(inputData);
  }

  /** Asks the processor for the coefficients an evaluation at a continuous index reads, ahead of
   * the evaluation: the points of a metric fall all over the image, and their coefficients come
   * from memory while the points before them are evaluated. A hint, which changes no value.
   */
  void
  Prefetch(const typename Superclass::ContinuousIndexType & x) const
  {
#if defined(__GNUC__)
    constexpr unsigned int        Dimension = Superclass::ImageDimension;
    const CoefficientImageType &  image = *this->m_Coefficients;
    const auto &                  region = image.GetBufferedRegion();
    const OffsetValueType * const strides = image.GetOffsetTable();
    const auto * const            buffer = image.GetBufferPointer();
    const unsigned int            order = this->m_SplineOrder;
    // The offset along an axis of a position of the support, kept inside the buffer wherever the point is.
    const auto offset = [&](const unsigned int n, const unsigned int k) {
      const double position = std::floor(x[n] + (order & 1 ? 0.0 : 0.5)) - order / 2 - region.GetIndex(n) + k;
      const double last = static_cast<double>(region.GetSize(n) - 1);
      return static_cast<OffsetValueType>(position > 0.0 ? std::min(position, last) : 0.0) * strides[n];
    };
    // Both ends of each line of the support along the first axis.
    unsigned int position[Dimension]{};
    while (true)
    {
      OffsetValueType line = 0;
      for (unsigned int n = 1; n < Dimension; ++n)
      {
        line += offset(n, position[n]);
      }
      __builtin_prefetch(buffer + line + offset(0, 0));
      __builtin_prefetch(buffer + line + offset(0, order));
      unsigned int n = 1;
      while (n < Dimension && ++position[n] > order)
      {
        position[n++] = 0;
      }
      if (n >= Dimension)
      {
        break;
      }
    }
#endif
  }

protected:
  FrugalBSplineInterpolateImageFunction() = default;
  ~FrugalBSplineInterpolateImageFunction() override = default;

  using typename Superclass::ContinuousIndexType;
  using typename Superclass::CovariantVectorType;
  using typename Superclass::OutputType;

  OutputType
  EvaluateAtContinuousIndexInternal(const ContinuousIndexType & x,
                                    vnl_matrix<long> &          evaluateIndex,
                                    vnl_matrix<double> &        weights) const override
  {
    switch (this->m_SplineOrder)
    {
      case 1:
        return this->template Value<1>(x);
      case 3:
        return this->template Value<3>(x);
      default:
        return Superclass::EvaluateAtContinuousIndexInternal(x, evaluateIndex, weights);
    }
  }

  void
  EvaluateValueAndDerivativeAtContinuousIndexInternal(const ContinuousIndexType & x,
                                                      OutputType &                value,
                                                      CovariantVectorType &       derivativeValue,
                                                      vnl_matrix<long> &          evaluateIndex,
                                                      vnl_matrix<double> &        weights,
                                                      vnl_matrix<double> &        weightsDerivative) const override
  {
    switch (this->m_SplineOrder)
    {
      case 1:
        this->template ValueAndDerivative<1>(x, value, derivativeValue);
        break;
      case 3:
        this->template ValueAndDerivative<3>(x, value, derivativeValue);
        break;
      default:
        Superclass::EvaluateValueAndDerivativeAtContinuousIndexInternal(
          x, value, derivativeValue, evaluateIndex, weights, weightsDerivative);
    }
  }

private:
  /** What the superclass computes around a continuous index at order 1 or 3: the weights of each
   * axis, those of its derivative, and the coefficients of the points of the support, the first
   * axis fastest, with its mirror boundaries. The coefficients are read once, from the buffer of
   * their image, where the superclass looks each of them up for the value and for each axis of
   * the derivative.
   */
  template <unsigned int VOrder>
  struct Support
  {
    static constexpr unsigned int Dimension = Superclass::ImageDimension;
    static constexpr unsigned int Length = VOrder + 1;
    static constexpr unsigned int BitsPerAxis = VOrder == 1 ? 1 : 2;
    static constexpr unsigned int NumberOfPoints = 1u << (Dimension * BitsPerAxis);

    /** The position along an axis of a point of the support. */
    static constexpr unsigned int
    Position(const unsigned int point, const unsigned int axis)
    {
      return (point >> (axis * BitsPerAxis)) & VOrder;
    }

    double Weights[Dimension][Length];
    double DerivativeWeights[Dimension][Length];
    double Coefficients[NumberOfPoints];

    Support(const Self & function, const ContinuousIndexType & x)
    {
      static_assert(VOrder == 1 || VOrder == 3);
      const CoefficientImageType &  image = *function.m_Coefficients;
      const auto &                  startIndex = function.GetStartIndex();
      const auto &                  endIndex = function.GetEndIndex();
      const auto &                  bufferIndex = image.GetBufferedRegion().GetIndex();
      const OffsetValueType * const strides = image.GetOffsetTable();
      OffsetValueType               offsets[Dimension][Length];
      for (unsigned int n = 0; n < Dimension; ++n)
      {
        const long first = static_cast<long>(std::floor(static_cast<float>(x[n]))) - VOrder / 2;
        if constexpr (VOrder == 1)
        {
          const double w = x[n] - static_cast<double>(first);
          Weights[n][1] = w;
          Weights[n][0] = 1.0 - w;
          DerivativeWeights[n][0] = -1.0;
          DerivativeWeights[n][1] = 1.0;
        }
        else
        {
          double w = x[n] - static_cast<double>(first + 1);
          Weights[n][3] = (1.0 / 6.0) * w * w * w;
          Weights[n][0] = (1.0 / 6.0) + 0.5 * w * (w - 1.0) - Weights[n][3];
          Weights[n][2] = w + Weights[n][0] - 2.0 * Weights[n][3];
          Weights[n][1] = 1.0 - Weights[n][0] - Weights[n][2] - Weights[n][3];
          w = x[n] + .5 - static_cast<double>(first + 2);
          const double w2 = 0.75 - w * w;
          const double w3 = 0.5 * (w - w2 + 1.0);
          const double w1 = 1.0 - w2 - w3;
          DerivativeWeights[n][0] = 0.0 - w1;
          DerivativeWeights[n][1] = w1 - w2;
          DerivativeWeights[n][2] = w2 - w3;
          DerivativeWeights[n][3] = w3;
        }
        for (unsigned int k = 0; k < Length; ++k)
        {
          long index = first + k;
          if (function.m_DataLength[n] == 1)
          {
            index = 0;
          }
          else
          {
            if (index < startIndex[n])
            {
              index = startIndex[n] + (startIndex[n] - index);
            }
            if (index >= endIndex[n])
            {
              index = endIndex[n] - (index - endIndex[n]);
            }
          }
          offsets[n][k] = (index - bufferIndex[n]) * strides[n];
        }
      }
      const auto * const buffer = image.GetBufferPointer();
      for (unsigned int p = 0; p < NumberOfPoints; ++p)
      {
        OffsetValueType offset = 0;
        for (unsigned int n = 0; n < Dimension; ++n)
        {
          offset += offsets[n][Position(p, n)];
        }
        Coefficients[p] = buffer[offset];
      }
    }
  };

  /** The value at a continuous index, summed as the superclass sums it. */
  template <unsigned int VOrder>
  OutputType
  Value(const ContinuousIndexType & x) const
  {
    using SupportType = Support<VOrder>;
    const SupportType support(*this, x);
    double            interpolated = 0.0;
    for (unsigned int p = 0; p < SupportType::NumberOfPoints; ++p)
    {
      double w = 1.0;
      for (unsigned int n = 0; n < SupportType::Dimension; ++n)
      {
        w *= support.Weights[n][SupportType::Position(p, n)];
      }
      interpolated += w * support.Coefficients[p];
    }
    return interpolated;
  }

  /** The value and the derivative at a continuous index. The superclass sums the value and each axis
   * of the derivative over the support one after the other; here one pass over the support adds a
   * term to each sum, in the same order, so that the sums do not wait for each other.
   */
  template <unsigned int VOrder>
  void
  ValueAndDerivative(const ContinuousIndexType & x, OutputType & value, CovariantVectorType & derivativeValue) const
  {
    using SupportType = Support<VOrder>;
    const SupportType support(*this, x);
    double            interpolated = 0.0;
    double            derivative[SupportType::Dimension]{};
    for (unsigned int p = 0; p < SupportType::NumberOfPoints; ++p)
    {
      const double coefficient = support.Coefficients[p];
      double       w = 1.0;
      for (unsigned int n = 0; n < SupportType::Dimension; ++n)
      {
        w *= support.Weights[n][SupportType::Position(p, n)];
      }
      interpolated += w * coefficient;
      for (unsigned int n = 0; n < SupportType::Dimension; ++n)
      {
        double w1 = 1.0;
        for (unsigned int n1 = 0; n1 < SupportType::Dimension; ++n1)
        {
          const unsigned int k = SupportType::Position(p, n1);
          w1 *= n1 == n ? support.DerivativeWeights[n1][k] : support.Weights[n1][k];
        }
        derivative[n] += w1 * coefficient;
      }
    }
    const TImageType & image = *this->GetInputImage();
    value = interpolated;
    for (unsigned int n = 0; n < SupportType::Dimension; ++n)
    {
      derivativeValue[n] = derivative[n] / image.GetSpacing()[n];
    }
    if (this->GetUseImageDirection())
    {
      derivativeValue = image.TransformLocalVectorToPhysicalVector(derivativeValue);
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
    coefficients->SetBufferedRegion(region);
    coefficients->SetRequestedRegion(region);
    coefficients->Allocate();

    const auto threader = MultiThreaderBase::New();
    threader->template ParallelizeImageRegion<Superclass::ImageDimension>(
      region,
      [&](const RegionType & piece) { ImageAlgorithm::Copy(&image, coefficients.GetPointer(), piece, piece); },
      nullptr);

    // The poles of BSplineDecompositionImageFilter, which has none at order 0 or 1 and keeps those it had.
    const auto decomposition = Superclass::CoefficientFilter::New();
    decomposition->SetSplineOrder(this->GetSplineOrder());
    std::vector<double> poles = decomposition->GetSplinePoles();
    poles.resize(decomposition->GetNumberOfPoles());
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

  /** The modification time of the image set: the same image, modified since, has other pixels. */
  ModifiedTimeType m_InputTime{};
};

} // end namespace itk

#endif
