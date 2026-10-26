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
#ifndef itkImpactImageToImageMetricJS_hxx
#define itkImpactImageToImageMetricJS_hxx

#include "itkImpactImageToImageMetricJS.h"
#include "itkImpactJS.h"
#include "itkStatisticsImageFilter.h"

#include <algorithm>
#include <cmath>
#include <ctime>
#include <numeric>
#include <sstream>

namespace itk
{

template <typename TFixedImage, typename TMovingImage>
ImpactImageToImageMetric<TFixedImage, TMovingImage>::ImpactImageToImageMetric()
{
  this->SetUseImageSampler(true);
  this->SetUseFixedImageLimiter(false);
  this->SetUseMovingImageLimiter(false);
}

template <typename TFixedImage, typename TMovingImage>
ImpactImageToImageMetric<TFixedImage, TMovingImage>::~ImpactImageToImageMetric()
{
  this->ReleaseLevel();
}

template <typename TFixedImage, typename TMovingImage>
void
ImpactImageToImageMetric<TFixedImage, TMovingImage>::ReleaseLevel()
{
  if (m_Level >= 0)
  {
    impact_js_free(m_Level);
    m_Level = -1;
  }
}

/** A JSON string. */
inline std::string
ImpactJSONString(const std::string & text)
{
  std::string quoted = "\"";
  for (const char c : text)
  {
    if (c == '"' || c == '\\')
    {
      quoted += '\\';
    }
    quoted += c;
  }
  return quoted + "\"";
}

template <typename TFixedImage, typename TMovingImage>
std::string
ImpactImageToImageMetric<TFixedImage, TMovingImage>::LevelConfiguration() const
{
  std::ostringstream json;
  json << "{\"mode\":" << ImpactJSONString(m_Mode) << ",\"seed\":" << m_Seed
       << ",\"mixed_precision\":" << (m_UseMixedPrecision ? "true" : "false") << ",\"models\":[";
  std::size_t layer = 0;
  for (std::size_t i = 0; i < m_FixedModelsConfiguration.size(); ++i)
  {
    const ImpactModelConfiguration & model = m_FixedModelsConfiguration[i];
    std::string                      mask;
    for (const bool kept : model.GetLayersMask())
    {
      mask += kept ? '1' : '0';
    }
    json << (i ? "," : "") << "{\"ref\":" << ImpactJSONString(model.GetModelPath())
         << ",\"layers_mask\":" << ImpactJSONString(mask)
         << ",\"feature_normalization\":" << ImpactJSONString(model.GetFeatureNormalization()) << ",\"pca\":[";
    for (unsigned int k = 0; k < NumberOfKeptLayers(model); ++k, ++layer)
    {
      json << (k ? "," : "") << (layer < m_PCA.size() ? m_PCA[layer] : 0u);
    }
    json << "]}";
  }
  json << "],\"distances\":[";
  for (std::size_t l = 0; l < m_Distance.size(); ++l)
  {
    json << (l ? "," : "") << ImpactJSONString(m_Distance[l]);
  }
  json << "]}";
  return json.str();
}

template <typename TFixedImage, typename TMovingImage>
void
ImpactImageToImageMetric<TFixedImage, TMovingImage>::ExtractFeaturesMaps(int side)
{
  using ImageType = FixedImageType; // fixed and moving images are of one type in elastix
  const ImageType *        image = side ? ImpactMovingImage() : ImpactFixedImage();
  const InterpolatorType * interpolator =
    side ? Superclass::m_Interpolator.GetPointer() : m_FixedInterpolator.GetPointer();
  const auto &           configs = side ? m_MovingModelsConfiguration : m_FixedModelsConfiguration;
  constexpr unsigned int dimension = FixedImageDimension;
  const auto &           oldSize = image->GetLargestPossibleRegion().GetSize();
  const auto &           oldSpacing = image->GetSpacing();

  // What a metadata-aware model takes (ComputeImageMetadata): the statistics of the image, its direction rounded.
  const auto statistics = StatisticsImageFilter<ImageType>::New();
  statistics->SetInput(image);
  statistics->Update();
  const float stats[4] = { static_cast<float>(statistics->GetMinimum()),
                           static_cast<float>(statistics->GetMaximum()),
                           static_cast<float>(statistics->GetMean()),
                           static_cast<float>(statistics->GetSigma()) };
  double      direction[9] = { 0, 0, 0, 0, 0, 0, 0, 0, 0 };
  for (unsigned int r = 0; r < dimension; ++r)
  {
    for (unsigned int c = 0; c < dimension; ++c)
    {
      direction[r * 3 + c] = static_cast<double>(std::llround(image->GetDirection()(r, c)));
    }
  }

  std::size_t layer = 0;
  for (std::size_t model = 0; model < configs.size(); ++model)
  {
    // The image on the model's grid, as ImageToTensorFilter samples it: the image's origin and axes, a size rounded
    // from its extent over the voxel size, and the step that size takes to span the extent.
    typename ImageType::SizeType    newSize;
    typename ImageType::SpacingType newSpacing;
    SizeValueType                   numberOfPixels = 1;
    for (unsigned int d = 0; d < dimension; ++d)
    {
      newSize[d] = static_cast<SizeValueType>(oldSize[d] * oldSpacing[d] / configs[model].GetVoxelSize()[d] + 0.5);
      newSpacing[d] = oldSize[d] * oldSpacing[d] / newSize[d];
      numberOfPixels *= newSize[d];
    }
    std::vector<float> buffer(numberOfPixels, 0.0f);
    for (SizeValueType linear = 0; linear < numberOfPixels; ++linear)
    {
      ContinuousIndex<typename ImageType::PointType::ValueType, dimension> index;
      SizeValueType                                                        remainder = linear;
      for (unsigned int d = 0; d < dimension; ++d)
      {
        index[d] = (remainder % newSize[d]) * newSpacing[d] / oldSpacing[d];
        remainder /= newSize[d];
      }
      typename ImageType::PointType point;
      image->TransformContinuousIndexToPhysicalPoint(index, point);
      if (interpolator->IsInsideBuffer(point))
      {
        buffer[linear] = static_cast<float>(interpolator->Evaluate(point));
      }
    }

    const unsigned int        kept = NumberOfKeptLayers(configs[model]);
    std::vector<unsigned int> shapes(4 * kept);
    const int                 extracted = impact_js_extract(m_Level,
                                            side,
                                            static_cast<int>(model),
                                            buffer.data(),
                                            static_cast<int>(newSize[0]),
                                            static_cast<int>(newSize[1]),
                                            static_cast<int>(newSize[2]),
                                            stats,
                                            direction,
                                            shapes.data(),
                                            static_cast<int>(kept));
    if (extracted != static_cast<int>(kept))
    {
      itkExceptionMacro("IMPACT: model " << model << " gave " << extracted << " layers, its LayersMask keeps " << kept);
    }

    for (unsigned int k = 0; k < kept; ++k, ++layer)
    {
      // The map's grid (AllocateFeatureImage): the image's origin and direction, the image's extent over its size.
      const auto                        geometry = MapGeometry::New();
      typename MapGeometry::SizeType    size;
      typename MapGeometry::SpacingType spacing;
      for (unsigned int d = 0; d < dimension; ++d)
      {
        size[d] = shapes[4 * k + 1 + d];
        spacing[d] = oldSize[d] * oldSpacing[d] / size[d];
      }
      geometry->SetRegions(typename MapGeometry::RegionType(size));
      geometry->SetSpacing(spacing);
      geometry->SetOrigin(image->GetOrigin());
      geometry->SetDirection(image->GetDirection());
      if (side == 0)
      {
        m_Layers.push_back({ shapes[4 * k], geometry, nullptr });
      }
      else if (layer >= m_Layers.size() || m_Layers[layer].channels != shapes[4 * k])
      {
        itkExceptionMacro("IMPACT: the moving image's layer " << layer << " has " << shapes[4 * k]
                                                              << " channels, not the fixed image's");
      }
      else
      {
        m_Layers[layer].moving = geometry;
      }
    }
  }
  if (side == 1 && layer != m_Layers.size())
  {
    itkExceptionMacro("IMPACT: " << m_Layers.size() << " fixed and " << layer << " moving feature maps");
  }
}

template <typename TFixedImage, typename TMovingImage>
void
ImpactImageToImageMetric<TFixedImage, TMovingImage>::Initialize()
{
  Superclass::Initialize();
  m_LossNormalization.Reset(); // each resolution starts at 1 again
  m_FixedInterpolator->SetInputImage(this->ImpactFixedImage());
  if (m_ImpactMovingImage)
  {
    Superclass::m_Interpolator->SetInputImage(m_ImpactMovingImage);
  }
  m_RandomGenerator = std::mt19937(m_Seed > 0 ? m_Seed : static_cast<unsigned int>(time(nullptr)));

  if (m_Mode != "Static")
  {
    itkExceptionMacro("IMPACT's " << m_Mode << " mode does not run in this build: Static only.");
  }
  if (!(m_FixedModelsConfiguration == m_MovingModelsConfiguration))
  {
    itkExceptionMacro("IMPACT in this build takes one model list for both images (ImpactModelsPath), not "
                      "ImpactFixedModelsPath and ImpactMovingModelsPath.");
  }

  this->ReleaseLevel();
  m_Level = impact_js_create(this->LevelConfiguration().c_str());
  m_Layers.clear();
  this->ExtractFeaturesMaps(0);
  this->ExtractFeaturesMaps(1);

  if (m_SubsetFeatures.size() != m_Layers.size() || m_LayersWeight.size() != m_Layers.size())
  {
    itkExceptionMacro("IMPACT: " << m_Layers.size() << " kept layers, " << m_SubsetFeatures.size()
                                 << " SubsetFeatures and " << m_LayersWeight.size() << " LayersWeight");
  }
  m_FeaturesIndexes.clear();
  for (std::size_t l = 0; l < m_Layers.size(); ++l)
  {
    m_SubsetFeatures[l] = std::clamp<unsigned int>(m_SubsetFeatures[l], 1, m_Layers[l].channels);
    m_FeaturesIndexes.emplace_back(m_Layers[l].channels);
    std::iota(m_FeaturesIndexes[l].begin(), m_FeaturesIndexes[l].end(), 0);
  }

  if (m_NormalizeLosses)
  {
    // Latched at the transform this resolution starts from, as the native metric does.
    this->GetValue(this->m_Transform->GetParameters());
  }
}

template <typename TFixedImage, typename TMovingImage>
void
ImpactImageToImageMetric<TFixedImage, TMovingImage>::UpdateMovingFeaturesMaps()
{
  this->ExtractFeaturesMaps(1);
}

template <typename TFixedImage, typename TMovingImage>
bool
ImpactImageToImageMetric<TFixedImage, TMovingImage>::SampleCheck(const FixedImagePointType & fixedPoint) const
{
  const MovingImagePointType mappedPoint = this->TransformPoint(fixedPoint);
  if (!Superclass::m_Interpolator->IsInsideBuffer(mappedPoint))
  {
    return false;
  }
  const auto * const mask = this->GetMovingImageMask();
  return mask == nullptr || mask->IsInsideInWorldSpace(mappedPoint);
}

template <typename TFixedImage, typename TMovingImage>
std::vector<unsigned int>
ImpactImageToImageMetric<TFixedImage, TMovingImage>::GetSubsetOfFeatures(
  const std::vector<unsigned int> & featuresIndex,
  int                               n) const
{
  if (featuresIndex.size() == static_cast<size_t>(n))
  {
    return featuresIndex;
  }
  std::vector<unsigned int> shuffled = featuresIndex;
  std::shuffle(shuffled.begin(), shuffled.end(), m_RandomGenerator);
  shuffled.resize(n);
  return shuffled;
}

template <typename TFixedImage, typename TMovingImage>
void
ImpactImageToImageMetric<TFixedImage, TMovingImage>::Evaluate(const ParametersType & parameters,
                                                              MeasureType &          value,
                                                              DerivativeType *       derivative) const
{
  this->BeforeThreadedGetValueAndDerivative(parameters);

  const ImageSampleContainerType & samples = *this->GetImageSampler()->GetOutput();
  std::vector<FixedImagePointType> points;
  for (std::size_t s = 0; s < samples.size(); ++s)
  {
    if (this->SampleCheck(samples[s].m_ImageCoordinates))
    {
      points.push_back(samples[s].m_ImageCoordinates);
    }
  }
  Superclass::m_NumberOfPixelsCounted = points.size();
  this->CheckNumberOfSamples();

  const std::size_t  layers = m_Layers.size(), n = points.size();
  std::vector<float> fixed(layers * n * 3), moving(layers * n * 3);
  for (std::size_t i = 0; i < n; ++i)
  {
    const MovingImagePointType mapped = this->TransformPoint(points[i]);
    for (std::size_t l = 0; l < layers; ++l)
    {
      const auto f = m_Layers[l].fixed->template TransformPhysicalPointToContinuousIndex<double>(points[i]);
      const auto m = m_Layers[l].moving->template TransformPhysicalPointToContinuousIndex<double>(mapped);
      for (unsigned int d = 0; d < 3; ++d)
      {
        fixed[(l * n + i) * 3 + d] = static_cast<float>(f[d]);
        moving[(l * n + i) * 3 + d] = static_cast<float>(m[d]);
      }
    }
  }

  // The channels each layer compares at this evaluation, drawn as the native metric draws them.
  std::vector<unsigned int> channels, counts;
  bool                      every = true;
  for (std::size_t l = 0; l < layers; ++l)
  {
    const std::vector<unsigned int> subset = this->GetSubsetOfFeatures(m_FeaturesIndexes[l], m_SubsetFeatures[l]);
    channels.insert(channels.end(), subset.begin(), subset.end());
    counts.push_back(subset.size());
    every = every && subset.size() == m_Layers[l].channels;
  }

  std::vector<double> sums(layers);
  std::vector<float>  gradients(derivative ? layers * n * 3 : 0);
  impact_js_evaluate(m_Level,
                     static_cast<int>(n),
                     static_cast<int>(layers),
                     fixed.data(),
                     moving.data(),
                     every ? nullptr : channels.data(),
                     every ? nullptr : counts.data(),
                     derivative != nullptr,
                     sums.data(),
                     gradients.data());

  // Each layer's mean over the points, weighed, and divided by its value at the level's start (LossPerThreadStruct).
  value = MeasureType{};
  std::vector<double> scale(layers);
  for (std::size_t l = 0; l < layers; ++l)
  {
    const double layerValue = sums[l] / static_cast<double>(n);
    const double factor = m_NormalizeLosses ? m_LossNormalization.Latch(l, layerValue) : 1.0;
    value += m_LayersWeight[l] * factor * layerValue;
    scale[l] = m_LayersWeight[l] * factor / static_cast<double>(n);
  }
  if (derivative == nullptr)
  {
    return;
  }

  // d(loss)/d(moving index) to physical space as BSplineInterpolateImageFunction::EvaluateDerivative turns it
  // (divided by the spacing, then through the direction), and through the transform's Jacobian.
  derivative->SetSize(this->GetNumberOfParameters());
  derivative->Fill(DerivativeValueType{});
  TransformJacobianType      jacobian;
  NonZeroJacobianIndicesType nonZero(this->m_AdvancedTransform->GetNumberOfNonZeroJacobianIndices());
  for (std::size_t i = 0; i < n; ++i)
  {
    double gradient[3] = { 0, 0, 0 };
    for (std::size_t l = 0; l < layers; ++l)
    {
      const auto & spacing = m_Layers[l].moving->GetSpacing();
      const auto & direction = m_Layers[l].moving->GetDirection();
      for (unsigned int r = 0; r < 3; ++r)
      {
        for (unsigned int c = 0; c < 3; ++c)
        {
          gradient[r] += scale[l] * direction(r, c) * gradients[(l * n + i) * 3 + c] / spacing[c];
        }
      }
    }
    this->m_AdvancedTransform->GetJacobian(points[i], jacobian, nonZero);
    for (std::size_t k = 0; k < nonZero.size(); ++k)
    {
      (*derivative)[nonZero[k]] +=
        gradient[0] * jacobian(0, k) + gradient[1] * jacobian(1, k) + gradient[2] * jacobian(2, k);
    }
  }
}

template <typename TFixedImage, typename TMovingImage>
auto
ImpactImageToImageMetric<TFixedImage, TMovingImage>::GetValue(const ParametersType & parameters) const -> MeasureType
{
  MeasureType value{};
  this->Evaluate(parameters, value, nullptr);
  return value;
}

template <typename TFixedImage, typename TMovingImage>
void
ImpactImageToImageMetric<TFixedImage, TMovingImage>::GetDerivative(const ParametersType & parameters,
                                                                   DerivativeType &       derivative) const
{
  MeasureType value{};
  this->Evaluate(parameters, value, &derivative);
}

template <typename TFixedImage, typename TMovingImage>
void
ImpactImageToImageMetric<TFixedImage, TMovingImage>::GetValueAndDerivative(const ParametersType & parameters,
                                                                           MeasureType &          value,
                                                                           DerivativeType &       derivative) const
{
  this->Evaluate(parameters, value, &derivative);
}

} // end namespace itk

#endif // end #ifndef itkImpactImageToImageMetricJS_hxx
