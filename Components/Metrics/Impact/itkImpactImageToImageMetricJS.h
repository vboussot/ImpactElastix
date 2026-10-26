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
#ifndef itkImpactImageToImageMetricJS_h
#define itkImpactImageToImageMetricJS_h

#include "itkAdvancedImageToImageMetric.h"
#include "itkBSplineInterpolateImageFunction.h"
#include "itkImpactModelConfiguration.h"
#include "itkImpactLossNormalization.h"

#include <random>
#include <string>
#include <vector>

namespace itk
{

/**
 * \class ImpactImageToImageMetric
 * \brief The IMPACT metric of the WebAssembly build: itkImpactImageToImageMetric without LibTorch.
 *
 * Everything elastix does stays here as the native metric does it: the samples, SampleCheck, the images resampled
 * on each model's grid (ImageToTensorFilter's grid), the channels drawn, the layers' weights and their normalization,
 * and the chain rule through the transform's Jacobian. What the native metric ran on LibTorch (the networks, the
 * feature maps and their cubic B-spline interpolation, the distances) is asked of the page's konfai-impact through
 * the JavaScript calls of itkImpactJS.h. Static mode only, and one model list for both images.
 *
 * \ingroup RegistrationMetrics
 * \ingroup Metrics
 */
template <typename TFixedImage, typename TMovingImage>
class ITK_TEMPLATE_EXPORT ImpactImageToImageMetric : public AdvancedImageToImageMetric<TFixedImage, TMovingImage>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(ImpactImageToImageMetric);

  using Self = ImpactImageToImageMetric;
  using Superclass = AdvancedImageToImageMetric<TFixedImage, TMovingImage>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  itkNewMacro(Self);
  itkTypeMacro(ImpactImageToImageMetric, AdvancedImageToImageMetric);

  using typename Superclass::CoordinateRepresentationType;
  using typename Superclass::MovingImageType;
  using typename Superclass::FixedImageType;
  using typename Superclass::TransformJacobianType;
  using typename Superclass::InterpolatorType;
  using typename Superclass::InterpolatorPointer;
  using typename Superclass::MeasureType;
  using typename Superclass::DerivativeType;
  using typename Superclass::DerivativeValueType;
  using typename Superclass::ParametersType;
  using typename Superclass::ImageSampleContainerType;
  using typename Superclass::FixedImagePointType;
  using typename Superclass::MovingImagePointType;
  using typename Superclass::NonZeroJacobianIndicesType;

  itkStaticConstMacro(FixedImageDimension, unsigned int, FixedImageType::ImageDimension);
  itkStaticConstMacro(MovingImageDimension, unsigned int, MovingImageType::ImageDimension);

  MeasureType
  GetValue(const ParametersType & parameters) const override;

  void
  GetDerivative(const ParametersType & parameters, DerivativeType & derivative) const override;

  void
  GetValueAndDerivative(const ParametersType & parameters,
                        MeasureType &          value,
                        DerivativeType &       derivative) const override;

  /** The level's feature maps (Static), extracted by the page's konfai-impact. */
  void
  Initialize() override;

  /** The moving image's maps extracted again (ImpactFeaturesMapUpdateInterval). */
  void
  UpdateMovingFeaturesMaps();

  itkSetMacro(FixedModelsConfiguration, std::vector<ImpactModelConfiguration>);
  itkGetConstReferenceMacro(FixedModelsConfiguration, std::vector<ImpactModelConfiguration>);
  itkSetMacro(MovingModelsConfiguration, std::vector<ImpactModelConfiguration>);
  itkGetConstReferenceMacro(MovingModelsConfiguration, std::vector<ImpactModelConfiguration>);
  itkSetMacro(SubsetFeatures, std::vector<unsigned int>);
  itkGetConstMacro(SubsetFeatures, std::vector<unsigned int>);
  itkSetMacro(LayersWeight, std::vector<float>);
  itkGetConstMacro(LayersWeight, std::vector<float>);
  itkSetMacro(Distance, std::vector<std::string>);
  itkGetConstMacro(Distance, std::vector<std::string>);
  itkSetMacro(PCA, std::vector<unsigned int>);
  itkGetConstMacro(PCA, std::vector<unsigned int>);
  itkSetMacro(UseMixedPrecision, bool);
  itkGetConstMacro(UseMixedPrecision, bool);
  itkSetMacro(WriteFeatureMaps, bool);
  itkGetConstMacro(WriteFeatureMaps, bool);
  itkSetMacro(FeatureMapsPath, std::string);
  itkGetConstMacro(FeatureMapsPath, std::string);
  itkSetMacro(Mode, std::string);
  itkGetConstMacro(Mode, std::string);
  itkSetMacro(CurrentLevel, unsigned int);
  itkSetMacro(Seed, unsigned int);
  itkSetMacro(FeaturesMapUpdateInterval, int);
  itkGetConstMacro(FeaturesMapUpdateInterval, int);
  itkSetMacro(NormalizeLosses, bool);
  itkSetConstObjectMacro(ImpactFixedImage, FixedImageType);
  itkSetConstObjectMacro(ImpactMovingImage, MovingImageType);

  /** Where the models run, for the log. */
  std::string
  GetDevice() const
  {
    return "WebGPU (konfai-impact)";
  }

protected:
  ImpactImageToImageMetric();
  ~ImpactImageToImageMetric() override;

private:
  /** The grid of a feature map: the original image's origin and direction, its extent over the map's size. */
  using MapGeometry = ImageBase<FixedImageDimension>;

  /** Each kept layer's channels and the grids of its fixed and moving maps. */
  struct Layer
  {
    unsigned int                  channels;
    typename MapGeometry::Pointer fixed;
    typename MapGeometry::Pointer moving;
  };

  /** The value, and the derivative when `derivative` is not null, at `parameters`. */
  void
  Evaluate(const ParametersType & parameters, MeasureType & value, DerivativeType * derivative) const;

  /** The maps of every model on one image (0 fixed, 1 moving): their grids, and their channels for the fixed. */
  void
  ExtractFeaturesMaps(int side);

  /** The level's JSON for impact_js_create. */
  std::string
  LevelConfiguration() const;

  std::vector<unsigned int>
  GetSubsetOfFeatures(const std::vector<unsigned int> & featuresIndex, int n) const;

  bool
  SampleCheck(const FixedImagePointType & fixedImageCenterCoordinate) const;

  void
  ReleaseLevel();

  std::vector<ImpactModelConfiguration> m_FixedModelsConfiguration;
  std::vector<ImpactModelConfiguration> m_MovingModelsConfiguration;
  std::vector<unsigned int>             m_SubsetFeatures;
  std::vector<unsigned int>             m_PCA;
  std::vector<float>                    m_LayersWeight;
  std::vector<std::string>              m_Distance;
  int                                   m_FeaturesMapUpdateInterval{ -1 };
  std::string                           m_Mode{ "Static" };
  bool                                  m_NormalizeLosses{ true };
  bool                                  m_WriteFeatureMaps{ false };
  std::string                           m_FeatureMapsPath;
  bool                                  m_UseMixedPrecision{ false };
  unsigned int                          m_CurrentLevel{ 0 };
  unsigned int                          m_Seed{ 0 };

  typename FixedImageType::ConstPointer  m_ImpactFixedImage;
  typename MovingImageType::ConstPointer m_ImpactMovingImage;
  const FixedImageType *
  ImpactFixedImage() const
  {
    return m_ImpactFixedImage ? m_ImpactFixedImage.GetPointer() : Superclass::m_FixedImage.GetPointer();
  }
  const MovingImageType *
  ImpactMovingImage() const
  {
    return m_ImpactMovingImage ? m_ImpactMovingImage.GetPointer() : Superclass::m_MovingImage.GetPointer();
  }

  /** The fixed image read with a cubic B-spline onto the models' grids, as the native metric reads it. */
  InterpolatorPointer m_FixedInterpolator = [] {
    const auto interpolator =
      BSplineInterpolateImageFunction<FixedImageType, CoordinateRepresentationType, double>::New();
    interpolator->SetSplineOrder(3);
    return interpolator;
  }();

  int                                    m_Level{ -1 };
  std::vector<Layer>                     m_Layers;
  std::vector<std::vector<unsigned int>> m_FeaturesIndexes;
  mutable std::mt19937                   m_RandomGenerator;
  mutable Impact::LossNormalization      m_LossNormalization;
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkImpactImageToImageMetricJS.hxx"
#endif

#endif // end #ifndef itkImpactImageToImageMetricJS_h
