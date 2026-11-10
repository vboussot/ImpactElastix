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

#include <type_traits>

namespace itk
{

/** \class FrugalBSplineInterpolateImageFunction
 * \brief A BSplineInterpolateImageFunction that computes its coefficients only when it has to.
 *
 * The superclass computes a coefficient image each time an image is set. Two cases need none:
 * \li At order 0 or 1 a B-spline has no poles and its coefficients are the pixels: the image is
 *     its own coefficient image when both have the same pixel type.
 * \li An image that shares its pixels with the image set before, as two levels of a pyramid may,
 *     has the coefficients computed for that one.
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
    else if (pixels == nullptr || pixels != m_DecomposedPixels || pixels->GetMTime() != m_DecomposedTime ||
             this->GetSplineOrder() != m_DecomposedOrder)
    {
      Superclass::SetInputImage(inputData);
      m_DecomposedPixels = pixels;
      m_DecomposedTime = pixels == nullptr ? 0 : pixels->GetMTime();
      m_DecomposedOrder = this->GetSplineOrder();
      return;
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
  /** The pixels the coefficients were computed from, when, and at which order. */
  const Object *   m_DecomposedPixels{};
  ModifiedTimeType m_DecomposedTime{};
  int              m_DecomposedOrder{};
};

} // end namespace itk

#endif
