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

#ifndef _itkImpactImageToImageMetric_hxx
#define _itkImpactImageToImageMetric_hxx

#include "itkImpactImageToImageMetric.h"
#include <deque>
#include <functional>
#include <iterator>
#include <vector>

#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkLinearInterpolateImageFunction.h"
#include "itkResampleImageFilter.h"
#include "itkScaleTransform.h"
#include "itkObjectFactoryBase.h"
#include "itkMetaImageIOFactory.h"

namespace itk
{

/**
 * ******************* Constructor *******************
 */
template <typename TFixedImage, typename TMovingImage>
ImpactImageToImageMetric<TFixedImage, TMovingImage>::ImpactImageToImageMetric()
{
  this->SetUseImageSampler(true);
  this->SetUseFixedImageLimiter(false);
  this->SetUseMovingImageLimiter(false);
} // end Constructor

/**
 * ********************* UpdateFixedFeaturesMaps ****************************
 */
template <typename TFixedImage, typename TMovingImage>
void
ImpactImageToImageMetric<TFixedImage, TMovingImage>::UpdateFixedFeaturesMaps()
{
  m_FixedFeaturesMaps.clear();
  m_PrincipalComponents.clear();

  // Generate the feature maps with the backend itk::ImageToFeaturesMap filter (the
  // framework-neutral Static feature extraction, with its patch tiling), one filter per
  // model, one FeaturesMaps per kept layer; remember each model's PCA basis for the moving.
  const auto & fixedConfigs = GetFixedModelsConfiguration();
  for (unsigned int i = 0; i < fixedConfigs.size(); ++i)
  {
    auto featuresMap = itk::ImageToFeaturesMap<TFixedImage, InterpolatorType>::New();
    featuresMap->SetModelConfiguration(fixedConfigs[i]);
    featuresMap->SetInterpolator(m_FixedInterpolator);
    featuresMap->AddInput(this->ImpactFixedImage());
    featuresMap->SetPCA(GetPCA()[i]);
    featuresMap->SetDevice(GetDevice().str());
    featuresMap->Update();

    const std::vector<torch::Tensor> principalComponents = itk::GetPrincipalComponents(*featuresMap);
    m_PrincipalComponents.insert(m_PrincipalComponents.end(), principalComponents.begin(), principalComponents.end());

    const unsigned int numberOfLayers =
      std::accumulate(fixedConfigs[i].GetLayersMask().begin(), fixedConfigs[i].GetLayersMask().end(), 0);
    for (unsigned int layer = 0; layer < numberOfLayers; ++layer)
    {
      m_FixedFeaturesMaps.emplace_back(const_cast<FeaturesImageType *>(featuresMap->GetOutput(layer).GetPointer()));
    }
  }

  if (GetWriteFeatureMaps())
  {
    unsigned int level = this->GetCurrentLevel();
    using WriterType = itk::ImageFileWriter<FeaturesImageType>;
    for (int i = 0; i < m_FixedFeaturesMaps.size(); ++i)
    {
      typename WriterType::Pointer writer = WriterType::New();

      writer->SetFileName(GetFeatureMapsPath() + "/Fixed_" + std::to_string(level) + "_" + std::to_string(i) + ".mha");
      writer->SetInput(m_FixedFeaturesMaps[i].m_FeaturesMaps);
      try
      {
        writer->Update();
      }
      catch (itk::ExceptionObject & error)
      {
        itkGenericExceptionMacro("Error writing feature map: " << writer->GetFileName() << " ITK Exception: " << error);
      }
    }
  }
} // end UpdateFixedFeaturesMaps

/**
 * ********************* UpdateMovingFeaturesMaps ****************************
 */
template <typename TFixedImage, typename TMovingImage>
void
ImpactImageToImageMetric<TFixedImage, TMovingImage>::UpdateMovingFeaturesMaps()
{
  m_MovingFeaturesMaps.clear();

  // Same as the fixed image, reusing the PCA basis fitted on the fixed image. The map is the
  // moving image's own, on its own grid and untransformed: ComputeValue*Static look it up at
  // TransformPoint(x), so sampling it through the transform as well would apply the transform
  // twice whenever it is not the identity when the map is built -- with an initial transform,
  // and at every resolution after the first, which then undoes the one before it.
  const auto & movingConfigs = GetMovingModelsConfiguration();
  unsigned int pcaOffset = 0;
  for (unsigned int i = 0; i < movingConfigs.size(); ++i)
  {
    auto featuresMap = itk::ImageToFeaturesMap<TMovingImage, InterpolatorType>::New();
    featuresMap->SetModelConfiguration(movingConfigs[i]);
    featuresMap->SetInterpolator(Superclass::m_Interpolator);
    featuresMap->AddInput(this->ImpactMovingImage());
    featuresMap->SetPCA(GetPCA()[i]);
    featuresMap->SetDevice(GetDevice().str());

    const unsigned int numberOfLayers =
      std::accumulate(movingConfigs[i].GetLayersMask().begin(), movingConfigs[i].GetLayersMask().end(), 0);
    if (GetPCA()[i] > 0 && pcaOffset + numberOfLayers <= m_PrincipalComponents.size())
    {
      std::vector<torch::Tensor> principalComponents(m_PrincipalComponents.begin() + pcaOffset,
                                                     m_PrincipalComponents.begin() + pcaOffset + numberOfLayers);
      itk::SetPrincipalComponents(*featuresMap, principalComponents);
    }
    featuresMap->Update();

    for (unsigned int layer = 0; layer < numberOfLayers; ++layer)
    {
      m_MovingFeaturesMaps.emplace_back(const_cast<FeaturesImageType *>(featuresMap->GetOutput(layer).GetPointer()));
    }
    pcaOffset += numberOfLayers;
  }

  using WriterType = itk::ImageFileWriter<FeaturesImageType>;
  if (GetWriteFeatureMaps())
  {
    unsigned int level = GetCurrentLevel();

    for (int i = 0; i < m_MovingFeaturesMaps.size(); ++i)
    {
      typename WriterType::Pointer writer = WriterType::New();

      writer->SetFileName(GetFeatureMapsPath() + "/Moving_" + std::to_string(level) + "_" + std::to_string(i) + ".mha");
      writer->SetInput(m_MovingFeaturesMaps[i].m_FeaturesMaps);
      try
      {
        writer->Update();
      }
      catch (itk::ExceptionObject & error)
      {
        itkGenericExceptionMacro("Error writing feature map: " << writer->GetFileName() << "ITK Exception: " << error);
      }
    }
  }
} // end UpdateMovingFeaturesMaps

/**
 * ********************* Initialize ****************************
 */
template <typename TFixedImage, typename TMovingImage>
void
ImpactImageToImageMetric<TFixedImage, TMovingImage>::Initialize()
{
  /** Initialize transform, interpolator, etc. */
  Superclass::Initialize();
  m_LossNormalization.Reset(); // each resolution starts at 1 again
  m_FixedInterpolator->SetInputImage(this->ImpactFixedImage());
  if (m_ImpactMovingImage)
  {
    // This metric's own interpolator (each metric of a multi-metric registration has one): it samples IMPACT's
    // moving image, the other metrics keep theirs on the pyramid level.
    Superclass::m_Interpolator->SetInputImage(m_ImpactMovingImage);
  }
  m_FeaturesIndexes.clear();

  for (auto & config : m_FixedModelsConfiguration)
  {
    itk::SetupImageMetadata<TFixedImage>(config, this->ImpactFixedImage());
  }

  for (auto & config : m_MovingModelsConfiguration)
  {
    itk::SetupImageMetadata<TMovingImage>(config, this->ImpactMovingImage());
  }

  if (GetMode() == "Static")
  {
    // Explicitly register MetaImageIO to enable writing .mha files when IMPACT
    // is loaded as a dynamically linked component (IO factories may not be
    // auto-registered in this case, especially on macOS).
    itk::ObjectFactoryBase::RegisterFactory(itk::MetaImageIOFactory::New());
    UpdateFixedFeaturesMaps();
    UpdateMovingFeaturesMaps();

    if (m_FixedFeaturesMaps.size() != m_MovingFeaturesMaps.size())
    {
      itkExceptionMacro("Mismatch in number of feature maps: "
                        << "fixedFeaturesMaps.size() = " << m_FixedFeaturesMaps.size()
                        << ", movingFeaturesMaps.size() = " << m_MovingFeaturesMaps.size());
    }

    for (int i = 0; i < m_FixedFeaturesMaps.size(); ++i)
    {
      if (m_FixedFeaturesMaps[i].m_FeaturesMaps->GetNumberOfComponentsPerPixel() !=
          m_MovingFeaturesMaps[i].m_FeaturesMaps->GetNumberOfComponentsPerPixel())
      {
        itkExceptionMacro("Mismatch in number of components per feature map at layer "
                          << i << ": fixed = " << m_FixedFeaturesMaps[i].m_FeaturesMaps->GetNumberOfComponentsPerPixel()
                          << ", moving = " << m_MovingFeaturesMaps[i].m_FeaturesMaps->GetNumberOfComponentsPerPixel());
      }
    }

    for (int i = 0; i < m_FixedFeaturesMaps.size(); ++i)
    {
      int numComponents = m_FixedFeaturesMaps[i].m_FeaturesMaps->GetNumberOfComponentsPerPixel();
      m_SubsetFeatures[i] = std::clamp<unsigned int>(m_SubsetFeatures[i], 1, numComponents);
      m_FeaturesIndexes.push_back(std::vector<unsigned int>(numComponents));
      std::iota(m_FeaturesIndexes[i].begin(), m_FeaturesIndexes[i].end(), 0);
    }
  }
  else
  {
    std::vector<torch::Tensor> fixedOutputsTensor =
      itk::Impact::GetModelOutputsExample(m_FixedModelsConfiguration, "fixed", GetDevice());
    std::vector<torch::Tensor> movingOutputsTensor =
      itk::Impact::GetModelOutputsExample(m_MovingModelsConfiguration, "moving", GetDevice());

    if (fixedOutputsTensor.size() != movingOutputsTensor.size())
    {
      itkExceptionMacro("Mismatch in number of feature maps: " << "fixed = " << fixedOutputsTensor.size()
                                                               << ", moving = " << movingOutputsTensor.size());
    }

    for (int i = 0; i < fixedOutputsTensor.size(); ++i)
    {
      if (fixedOutputsTensor[i].size(0) != movingOutputsTensor[i].size(0))
      {
        itkExceptionMacro("Mismatch in number of components per feature map at layer "
                          << i << ": fixed = " << fixedOutputsTensor[i].size(0)
                          << ", moving = " << movingOutputsTensor[i].size(0));
      }
    }

    for (int i = 0; i < fixedOutputsTensor.size(); ++i)
    {
      int numComponents = fixedOutputsTensor[i].size(1);
      m_SubsetFeatures[i] = std::clamp<unsigned int>(m_SubsetFeatures[i], 1, numComponents);
      m_FeaturesIndexes.push_back(std::vector<unsigned int>(numComponents));
      std::iota(m_FeaturesIndexes[i].begin(), m_FeaturesIndexes[i].end(), 0);
    }
  }

  if (m_NormalizeLosses)
  {
    // Latch each layer's factor at the transform this resolution starts from (the registration sets it before
    // initializing the metric). The optimizer's own first evaluations are ASGD's step-size estimation, at randomly
    // perturbed transforms: latched there, every layer started below 1, the most sensitive ones the lowest.
    this->GetValue(this->m_Transform->GetParameters());
  }
} // end Initialize

/**
 * ******************* SampleCheck *******************
 */
template <typename TFixedImage, typename TMovingImage>
bool
ImpactImageToImageMetric<TFixedImage, TMovingImage>::SampleCheck(
  const FixedImagePointType & fixedImageCenterCoordinate) const
{
  FixedImagePointType  fixedImagePoint(fixedImageCenterCoordinate);
  MovingImagePointType mappedPoint;
  mappedPoint = this->TransformPoint(fixedImagePoint);
  if (Superclass::m_Interpolator->IsInsideBuffer(mappedPoint) == false)
  {
    return false;
  }
  else
  {
    if (const auto * const mask = this->GetMovingImageMask())
    {
      if (mask->IsInsideInWorldSpace(mappedPoint) == false)
      {
        return false;
      }
    }
  }
  return true;
} // end SampleCheck

/**
 * ******************* EvaluateOnline *******************
 */
template <typename TFixedImage, typename TMovingImage>
void
ImpactImageToImageMetric<TFixedImage, TMovingImage>::EvaluateOnline(MeasureType &    value,
                                                                    DerivativeType * derivative) const
{
  const auto units = static_cast<int64_t>(this->GetUseMultiThread() ? Self::GetNumberOfWorkUnits() : 1);
  /** The threads' shares run on one OpenMP thread each, and the reductions after them, over a derivative that a
   * B-spline transform makes tens of thousands long, are not worth a team either: one would compete with the threads.
   */
  const itk::Impact::OneOpenMPThread                                       oneThread(units > 1);
  const bool                                                               computeDerivative = derivative != nullptr;
  const std::function<void(int64_t, const std::function<void(int64_t)> &)> parallelize =
    [this](int64_t parts, const std::function<void(int64_t)> & body) {
      if (parts == 1)
      {
        body(0);
        return;
      }
      this->m_Threader->ParallelizeArray(
        0, parts, [&](SizeValueType part) { body(static_cast<int64_t>(part)); }, nullptr);
    };

  /** The samples whose patches lie inside the moving image (and its mask), checked over the threads: per model, the
   * patch at each -- the configuration's own for a model of the image's dimension, a plane drawn for the point for a
   * 2D model in a volume (itk::Impact::GetPatchIndex) -- and where the transform maps every patch voxel, where the
   * moving patch is then read. */
  using PatchType = std::vector<std::vector<float>>;
  struct Samples
  {
    std::vector<FixedImagePointType>               points;
    std::vector<std::vector<const PatchType *>>    patches; // per model, per point
    std::vector<std::vector<MovingImagePointType>> mapped;  // per model, the voxels of every point in turn
    std::deque<PatchType>                          planes;
  };
  const ImageSampleContainerPointer sampleContainer = this->GetImageSampler()->GetOutput();
  const auto                        count = static_cast<int64_t>(sampleContainer->Size());
  const auto &                      configs = GetFixedModelsConfiguration();
  const auto &                      direction = this->GetFixedImage()->GetDirection();
  const unsigned int                seed = this->GetMutableRandomVariateGenerator().GetSeed();
  const auto * const                mask = this->GetMovingImageMask();
  std::vector<Samples>              shares(units);
  parallelize(units, [&](int64_t part) {
    Samples & share = shares[part];
    share.patches.resize(configs.size());
    share.mapped.resize(configs.size());
    std::vector<const PatchType *>    patches(configs.size());
    std::vector<MovingImagePointType> mapped;
    for (int64_t s = count * part / units; s < count * (part + 1) / units; ++s)
    {
      const FixedImagePointType & point = (*sampleContainer)[s].m_ImageCoordinates;
      bool                        inside = true;
      mapped.clear();
      for (size_t i = 0; inside && i < configs.size(); ++i)
      {
        if (configs[i].GetPatchSize().size() == FixedImageDimension)
        {
          patches[i] = &configs[i].GetPatchIndex();
        }
        else
        {
          share.planes.push_back(itk::Impact::GetPatchIndex(
            configs[i], seed, i, this->ImpactFixedImage()->TransformPhysicalPointToIndex(point), FixedImageDimension));
          patches[i] = &share.planes.back();
        }
        for (const std::vector<float> & offset : *patches[i])
        {
          const MovingImagePointType at = this->TransformPoint(this->PatchPoint(point, direction, offset));
          if (!Superclass::m_Interpolator->IsInsideBuffer(at) || (mask && !mask->IsInsideInWorldSpace(at)))
          {
            inside = false;
            break;
          }
          mapped.push_back(at);
        }
      }
      if (!inside)
      {
        continue;
      }
      share.points.push_back(point);
      auto from = mapped.cbegin();
      for (size_t i = 0; i < configs.size(); ++i)
      {
        share.patches[i].push_back(patches[i]);
        share.mapped[i].insert(share.mapped[i].end(), from, from + patches[i]->size());
        from += patches[i]->size();
      }
    }
  });
  Samples kept; // the drawn planes stay in the shares
  kept.patches.resize(configs.size());
  kept.mapped.resize(configs.size());
  for (const Samples & share : shares)
  {
    kept.points.insert(kept.points.end(), share.points.begin(), share.points.end());
    for (size_t i = 0; i < configs.size(); ++i)
    {
      kept.patches[i].insert(kept.patches[i].end(), share.patches[i].begin(), share.patches[i].end());
      kept.mapped[i].insert(kept.mapped[i].end(), share.mapped[i].begin(), share.mapped[i].end());
    }
  }

  for (ThreadIdType i = 0; i < Self::GetNumberOfWorkUnits(); ++i)
  {
    m_LossThreadStruct[i].reset();
  }
  m_LossThreadStruct[0].m_NumberOfPixelsCounted = kept.points.size();

  /** The points as itk::Impact::EvaluateOnline asks for them: a row a patch per model, every voxel through the
   * metric's interpolators, and the transform's Jacobian at the point. */
  class Host : public itk::Impact::OnlineHost
  {
  public:
    Host(const Self &                                                               metric,
         const Samples &                                                            samples,
         const std::function<void(int64_t, const std::function<void(int64_t)> &)> & parallelize,
         int64_t                                                                    units)
      : m_Metric(metric)
      , m_Samples(samples)
      , m_Parallelize(parallelize)
      , m_Units(units)
      , m_Parameters(metric.m_AdvancedTransform->GetNumberOfNonZeroJacobianIndices())
    {}
    itk::Impact::OnlineBatch
    Sample(int64_t begin, int64_t end, bool parallel, bool derivative, bool fixed) const override
    {
      const int64_t            size = end - begin;
      itk::Impact::OnlineBatch batch;
      for (const auto & patches : m_Samples.patches)
      {
        const auto voxels = static_cast<int64_t>(patches.front()->size());
        batch.fixedPatches.push_back(fixed ? torch::empty({ size, voxels }) : torch::Tensor());
        batch.movingPatches.push_back(torch::empty({ size, voxels }));
        batch.movingGradients.push_back(derivative ? torch::empty({ size, voxels, MovingImageDimension })
                                                   : torch::Tensor());
      }
      if (derivative)
      {
        batch.transformJacobian = torch::empty({ size, MovingImageDimension, m_Parameters });
        batch.nonZeroJacobianIndices = torch::empty({ size, m_Parameters }, torch::kInt64);
      }
      const auto & direction = m_Metric.GetFixedImage()->GetDirection();
      auto         rows = [&](int64_t first, int64_t last) {
        TransformJacobianType      jacobian;
        NonZeroJacobianIndicesType indices(m_Parameters);
        for (int64_t row = first; row < last; ++row)
        {
          const FixedImagePointType & center = m_Samples.points[begin + row];
          for (size_t i = 0; i < m_Samples.patches.size(); ++i)
          {
            const PatchType &                  patch = *m_Samples.patches[i][begin + row];
            const auto                         voxels = static_cast<int64_t>(patch.size());
            const MovingImagePointType * const mapped = m_Samples.mapped[i].data() + (begin + row) * voxels;
            float * const fixedOut = fixed ? batch.fixedPatches[i].data_ptr<float>() + row * voxels : nullptr;
            float * const movingOut = batch.movingPatches[i].data_ptr<float>() + row * voxels;
            float * const gradientOut =
              derivative ? batch.movingGradients[i].data_ptr<float>() + row * voxels * MovingImageDimension : nullptr;
            for (int64_t v = 0; v < voxels; ++v)
            {
              if (fixedOut != nullptr)
              {
                fixedOut[v] = static_cast<float>(
                  m_Metric.m_FixedInterpolator->Evaluate(m_Metric.PatchPoint(center, direction, patch[v])));
              }
              RealType                  value;
              MovingImageDerivativeType gradient;
              m_Metric.EvaluateMovingImageValueAndDerivative(
                mapped[v], value, gradientOut != nullptr ? &gradient : nullptr);
              movingOut[v] = static_cast<float>(value);
              for (unsigned int d = 0; gradientOut != nullptr && d < MovingImageDimension; ++d)
              {
                gradientOut[v * MovingImageDimension + d] = static_cast<float>(gradient[d]);
              }
            }
          }
          if (derivative)
          {
            m_Metric.m_AdvancedTransform->GetJacobian(center, jacobian, indices);
            float * const jacobianOut =
              batch.transformJacobian.data_ptr<float>() + row * MovingImageDimension * m_Parameters;
            int64_t * const indicesOut = batch.nonZeroJacobianIndices.data_ptr<int64_t>() + row * m_Parameters;
            for (int64_t p = 0; p < m_Parameters; ++p)
            {
              for (unsigned int d = 0; d < MovingImageDimension; ++d)
              {
                jacobianOut[d * m_Parameters + p] = static_cast<float>(jacobian(d, p));
              }
              indicesOut[p] = static_cast<int64_t>(indices[p]);
            }
          }
        }
      };
      const int64_t parts = parallel ? std::min(size, m_Units) : 1;
      m_Parallelize(parts, [&](int64_t part) { rows(size * part / parts, size * (part + 1) / parts); });
      return batch;
    }
    void
    Parallelize(int64_t parts, const std::function<void(int64_t)> & body) const override
    {
      m_Parallelize(parts, body);
    }

  private:
    const Self &                                                               m_Metric;
    const Samples &                                                            m_Samples;
    const std::function<void(int64_t, const std::function<void(int64_t)> &)> & m_Parallelize;
    const int64_t                                                              m_Units;
    const int64_t                                                              m_Parameters;
  };

  const itk::Impact::OnlineSettings settings{
    GetFixedModelsConfiguration(), GetMovingModelsConfiguration(), m_SubsetsOfFeaturesOnDevice, GetDevice(), 0, units
  };
  itk::Impact::EvaluateOnline(settings,
                              Host(*this, kept, parallelize, units),
                              static_cast<int64_t>(kept.points.size()),
                              computeDerivative,
                              [this](int64_t part) -> std::vector<std::unique_ptr<itk::Impact::Loss>> & {
                                return m_LossThreadStruct[part].m_Losses;
                              });
  if (computeDerivative)
  {
    this->AfterThreadedGetValueAndDerivative(value, *derivative);
  }
  else
  {
    this->AfterThreadedGetValue(value);
  }
} // end EvaluateOnline

/**
 * ******************* BeforeThreadedGetValueAndDerivative *******************
 */
template <typename TFixedImage, typename TMovingImage>
void
ImpactImageToImageMetric<TFixedImage, TMovingImage>::BeforeThreadedGetValueAndDerivative(
  const ParametersType & parameters) const
{
  Superclass::BeforeThreadedGetValueAndDerivative(parameters);

  /** Draw where the superclass draws the samples: a multi-metric registration calls this once with
   * UseMetricSingleThreaded on, then again, from GetValueAndDerivative, with it off. */
  if (!this->GetUseMetricSingleThreaded())
  {
    return;
  }
  auto & randomVariateGenerator = this->GetMutableRandomVariateGenerator();
  m_SubsetsOfFeatures = m_FeaturesIndexes;
  m_SubsetsOfFeaturesOnDevice.clear();
  for (unsigned int i = 0; i < m_SubsetsOfFeatures.size(); ++i)
  {
    std::vector<unsigned int> & subset = m_SubsetsOfFeatures[i];
    const unsigned int          n = GetSubsetFeatures()[i];
    if (n < subset.size())
    {
      /** The first n steps of a Fisher-Yates shuffle. */
      for (unsigned int k = 0; k < n; ++k)
      {
        const auto last = static_cast<unsigned int>(subset.size() - 1 - k);
        std::swap(subset[k], subset[k + randomVariateGenerator.GetIntegerVariate(last)]);
      }
      subset.resize(n);
    }
    m_SubsetsOfFeaturesOnDevice.push_back(
      torch::tensor(std::vector<int64_t>(subset.begin(), subset.end())).to(GetDevice()));
  }
} // end BeforeThreadedGetValueAndDerivative


/**
 * ******************* ComputeValueStatic *******************
 */
template <typename TFixedImage, typename TMovingImage>
unsigned int
ImpactImageToImageMetric<TFixedImage, TMovingImage>::ComputeValueStatic(
  const std::vector<FixedImagePointType> & fixedPointsTmp,
  LossPerThreadStruct &                    loss) const
{
  std::vector<FixedImagePointType> fixedPoints;
  fixedPoints.reserve(fixedPointsTmp.size());
  for (FixedImagePointType fixedPoint : fixedPointsTmp)
  {
    if (SampleCheck(fixedPoint))
    {
      fixedPoints.push_back(fixedPoint);
    }
  }
  if (fixedPoints.empty())
  {
    return 0;
  }
  unsigned int numberOfSample = fixedPoints.size();
  for (int i = 0; i < m_FixedFeaturesMaps.size(); ++i)
  {
    const std::vector<unsigned int> & subsetOfFeatures = m_SubsetsOfFeatures[i];

    torch::Tensor fixedOutputTensor = torch::zeros({ numberOfSample, GetSubsetFeatures()[i] });
    torch::Tensor movingOutputTensor = torch::zeros({ numberOfSample, GetSubsetFeatures()[i] });
    for (unsigned int s = 0; s < numberOfSample; ++s)
    {
      const auto & fixedPoint = fixedPoints[s];
      fixedOutputTensor[s] = m_FixedFeaturesMaps[i].m_FeaturesMapsInterpolator->Evaluate(fixedPoint, subsetOfFeatures);
      movingOutputTensor[s] = m_MovingFeaturesMaps[i].m_FeaturesMapsInterpolator->Evaluate(
        this->TransformPoint(fixedPoint), subsetOfFeatures);
    }
    loss.m_Losses[i]->updateValue(fixedOutputTensor, movingOutputTensor);
  }
  return numberOfSample;
} // end ComputeValueStatic

/**
 * ******************* ComputeValueAndDerivativeStatic *******************
 */
template <typename TFixedImage, typename TMovingImage>
unsigned int
ImpactImageToImageMetric<TFixedImage, TMovingImage>::ComputeValueAndDerivativeStatic(
  const std::vector<FixedImagePointType> & fixedPointsTmp,
  LossPerThreadStruct &                    loss) const
{
  std::vector<FixedImagePointType> fixedPoints;
  fixedPoints.reserve(fixedPointsTmp.size());
  for (FixedImagePointType fixedPoint : fixedPointsTmp)
  {
    if (SampleCheck(fixedPoint))
    {
      fixedPoints.push_back(fixedPoint);
    }
  }
  if (fixedPoints.empty())
  {
    return 0;
  }
  unsigned int  numberOfSample = fixedPoints.size();
  const int     numNonZeroJacobianIndices = this->m_AdvancedTransform->GetNumberOfNonZeroJacobianIndices();
  torch::Tensor nonZeroJacobianIndices = torch::zeros({ numberOfSample, numNonZeroJacobianIndices }, torch::kLong);

  torch::Tensor transformsJacobian = torch::zeros(
    { numberOfSample, MovingImageDimension, static_cast<int64_t>(numNonZeroJacobianIndices) }, torch::kFloat32);

  TransformJacobianType      flatTransformJacobian; // class itk::Array2D<double>
  NonZeroJacobianIndicesType flatNonZeroJacobianIndices(
    numNonZeroJacobianIndices); // class std::vector<unsigned long,class std::allocator<uint64_t> >
  for (unsigned int s = 0; s < numberOfSample; ++s)
  {
    this->m_AdvancedTransform->GetJacobian(fixedPoints[s], flatTransformJacobian, flatNonZeroJacobianIndices);
    nonZeroJacobianIndices[s] =
      torch::from_blob(&(*flatNonZeroJacobianIndices.begin()), { numNonZeroJacobianIndices }, torch::kUInt64)
        .to(torch::kLong)
        .clone();
    transformsJacobian[s] = torch::from_blob(&(*flatTransformJacobian.begin()),
                                             { MovingImageDimension, numNonZeroJacobianIndices },
                                             torch::kDouble)
                              .to(torch::kFloat32)
                              .clone();
  }


  for (int i = 0; i < m_FixedFeaturesMaps.size(); ++i)
  {
    const std::vector<unsigned int> & subsetOfFeatures = m_SubsetsOfFeatures[i];

    MovingImagePointType mappedPoint;
    torch::Tensor        fixedOutputTensor = torch::zeros({ numberOfSample, GetSubsetFeatures()[i] }, torch::kFloat32);
    torch::Tensor        movingOutputTensor = torch::zeros({ numberOfSample, GetSubsetFeatures()[i] }, torch::kFloat32);
    torch::Tensor        movingDerivativeTensor =
      torch::zeros({ numberOfSample, GetSubsetFeatures()[i], MovingImageDimension }, torch::kFloat32);
    for (unsigned int s = 0; s < numberOfSample; ++s)
    {
      const auto & fixedPoint = fixedPoints[s];
      mappedPoint = this->TransformPoint(fixedPoint);
      fixedOutputTensor[s] = m_FixedFeaturesMaps[i].m_FeaturesMapsInterpolator->Evaluate(fixedPoint, subsetOfFeatures);
      movingOutputTensor[s] =
        m_MovingFeaturesMaps[i].m_FeaturesMapsInterpolator->Evaluate(mappedPoint, subsetOfFeatures);
      movingDerivativeTensor[s] =
        m_MovingFeaturesMaps[i].m_FeaturesMapsInterpolator->EvaluateDerivative(mappedPoint, subsetOfFeatures);
    }
    torch::Tensor jacobian = torch::bmm(movingDerivativeTensor, transformsJacobian);
    loss.m_Losses[i]->updateValueAndDerivativeInStaticMode(
      fixedOutputTensor, movingOutputTensor, jacobian, nonZeroJacobianIndices);
  }
  return numberOfSample;
} // end ComputeValueAndDerivativeStatic

/**
 * ******************* InitializeThreadingParameters *******************
 */
template <typename TFixedImage, typename TMovingImage>
void
ImpactImageToImageMetric<TFixedImage, TMovingImage>::InitializeThreadingParameters() const
{
  const ThreadIdType numberOfThreads = Self::GetNumberOfWorkUnits();

  m_LossThreadStruct.reset(new AlignedLossPerThreadStruct[numberOfThreads]);
  for (ThreadIdType i = 0; i < numberOfThreads; ++i)
  {
    m_LossThreadStruct[i].init(GetDistance(), GetLayersWeight());
  }
  m_LossThreadStructSize = numberOfThreads;

  const int numberOfParameters = this->GetNumberOfParameters();
  for (ThreadIdType i = 0; i < numberOfThreads; ++i)
  {
    m_LossThreadStruct[i].setNumberOfParameters(numberOfParameters);
  }

} // end InitializeThreadingParameters

/**
 * ******************* GetValue *******************
 */
template <typename TFixedImage, typename TMovingImage>
auto
ImpactImageToImageMetric<TFixedImage, TMovingImage>::GetValue(const ParametersType & parameters) const -> MeasureType
{
  MeasureType value{};
  if (GetMode() == "Jacobian")
  {
    this->BeforeThreadedGetValueAndDerivative(parameters);
    this->EvaluateOnline(value, nullptr);
    return value;
  }
  if (!this->GetUseMultiThread())
  {
    return GetValueSingleThreaded(parameters);
  }
  this->BeforeThreadedGetValueAndDerivative(parameters);
  this->LaunchGetValueThreaderCallback();
  this->AfterThreadedGetValue(value);
  return value;
} // end GetValue

/**
 * ******************* GetValueSingleThreaded *******************
 */
template <typename TFixedImage, typename TMovingImage>
auto
ImpactImageToImageMetric<TFixedImage, TMovingImage>::GetValueSingleThreaded(const ParametersType & parameters) const
  -> MeasureType
{
  this->BeforeThreadedGetValueAndDerivative(parameters);
  /** Initialize some variables. */
  auto & loss = m_LossThreadStruct[0];
  loss.reset();

  /** Get a handle to the sample container. */
  ImageSampleContainerPointer sampleContainer = this->GetImageSampler()->GetOutput();
  const unsigned long         sampleContainerSize = sampleContainer->size();


  std::vector<FixedImagePointType> fixedPoints;
  fixedPoints.reserve(sampleContainerSize);
  for (unsigned int i = 0; i < sampleContainerSize; ++i)
  {
    fixedPoints.push_back((*sampleContainer)[i].m_ImageCoordinates);
  }
  Superclass::m_NumberOfPixelsCounted = ComputeValueStatic(fixedPoints, loss);
  this->CheckNumberOfSamples();

  return loss.GetValue((m_NormalizeLosses ? &m_LossNormalization : nullptr));
} // end GetValueSingleThreaded

/**
 * ******************* ThreadedGetValue *******************
 */
template <typename TFixedImage, typename TMovingImage>
void
ImpactImageToImageMetric<TFixedImage, TMovingImage>::ThreadedGetValue(ThreadIdType threadId) const
{
  /** Get a handle to the sample container. */
  ImageSampleContainerPointer sampleContainer = this->GetImageSampler()->GetOutput();
  const unsigned long         sampleContainerSize = sampleContainer->Size();

  /** Get the samples for this thread. */
  const unsigned long nrOfSamplesPerThreads = static_cast<unsigned long>(
    std::ceil(static_cast<double>(sampleContainerSize) / static_cast<double>(Self::GetNumberOfWorkUnits())));

  const auto posBegin = std::min<size_t>(nrOfSamplesPerThreads * threadId, sampleContainerSize);
  const auto posEnd = std::min<size_t>(nrOfSamplesPerThreads * (threadId + 1), sampleContainerSize);

  /** Create iterator over the sample container. */
  const auto beginOfSampleContainer = sampleContainer->cbegin();
  const auto threaderBegin = beginOfSampleContainer + posBegin;
  const auto threaderEnd = beginOfSampleContainer + posEnd;

  /** Create variables to store intermediate results. circumvent false sharing */

  LossPerThreadStruct & loss = m_LossThreadStruct[threadId];
  loss.reset();
  std::vector<FixedImagePointType> fixedPoints;
  fixedPoints.reserve(nrOfSamplesPerThreads);
  for (auto threaderIter = threaderBegin; threaderIter != threaderEnd; ++threaderIter)
  {
    fixedPoints.push_back(threaderIter->m_ImageCoordinates);
  }
  loss.m_NumberOfPixelsCounted += ComputeValueStatic(fixedPoints, loss);
} // end ThreadedGetValue

/**
 * ******************* AfterThreadedGetValue *******************
 */
template <typename TFixedImage, typename TMovingImage>
void
ImpactImageToImageMetric<TFixedImage, TMovingImage>::AfterThreadedGetValue(MeasureType & value) const
{
  LossPerThreadStruct & loss = m_LossThreadStruct[0];
  for (ThreadIdType i = 1; i < Self::GetNumberOfWorkUnits(); ++i)
  {
    loss += m_LossThreadStruct[i];
  }
  Superclass::m_NumberOfPixelsCounted = loss.m_NumberOfPixelsCounted;
  /** Check if enough samples were valid. */
  this->CheckNumberOfSamples();

  /** Accumulate values. */
  value = loss.GetValue((m_NormalizeLosses ? &m_LossNormalization : nullptr));
} // end AfterThreadedGetValue

/**
 * ******************* GetDerivative *******************
 */
template <typename TFixedImage, typename TMovingImage>
void
ImpactImageToImageMetric<TFixedImage, TMovingImage>::GetDerivative(const ParametersType & parameters,
                                                                   DerivativeType &       derivative) const
{
  MeasureType dummyvalue{};
  this->GetValueAndDerivative(parameters, dummyvalue, derivative);
} // end GetDerivative


/**
 * ******************* GetValueAndDerivative *******************
 */
template <typename TFixedImage, typename TMovingImage>
void
ImpactImageToImageMetric<TFixedImage, TMovingImage>::GetValueAndDerivative(const ParametersType & parameters,
                                                                           MeasureType &          value,
                                                                           DerivativeType &       derivative) const
{
  if (GetMode() == "Jacobian")
  {
    this->BeforeThreadedGetValueAndDerivative(parameters);
    this->EvaluateOnline(value, &derivative);
    return;
  }

  /** Option for now to still use the single threaded code. */
  if (!this->GetUseMultiThread())
  {
    return GetValueAndDerivativeSingleThreaded(parameters, value, derivative);
  }

  this->BeforeThreadedGetValueAndDerivative(parameters);

  /** Launch multi-threading metric */
  this->LaunchGetValueAndDerivativeThreaderCallback();

  /** Gather the metric values and derivatives from all threads. */
  AfterThreadedGetValueAndDerivative(value, derivative);

} // end GetValueAndDerivative

/**
 * ******************* GetValueAndDerivativeSingleThreaded *******************
 */
template <typename TFixedImage, typename TMovingImage>
void
ImpactImageToImageMetric<TFixedImage, TMovingImage>::GetValueAndDerivativeSingleThreaded(
  const ParametersType & parameters,
  MeasureType &          value,
  DerivativeType &       derivative) const
{

  this->BeforeThreadedGetValueAndDerivative(parameters);

  /** Initialize some variables. */
  LossPerThreadStruct & loss = m_LossThreadStruct[0];
  loss.reset();

  /** Get a handle to the sample container. */
  ImageSampleContainerPointer sampleContainer = this->GetImageSampler()->GetOutput();
  const unsigned long         sampleContainerSize = sampleContainer->size();


  std::vector<FixedImagePointType> fixedPoints;
  fixedPoints.reserve(sampleContainerSize);
  for (unsigned int i = 0; i < sampleContainerSize; ++i)
  {
    fixedPoints.push_back((*sampleContainer)[i].m_ImageCoordinates);
  }
  Superclass::m_NumberOfPixelsCounted = ComputeValueAndDerivativeStatic(fixedPoints, loss);
  this->CheckNumberOfSamples();

  value = loss.GetValue((m_NormalizeLosses ? &m_LossNormalization : nullptr));
  derivative = loss.GetDerivative((m_NormalizeLosses ? &m_LossNormalization : nullptr));
} // end GetValueAndDerivativeSingleThreaded

/**
 * ******************* ThreadedGetValueAndDerivative *******************
 */
template <typename TFixedImage, typename TMovingImage>
void
ImpactImageToImageMetric<TFixedImage, TMovingImage>::ThreadedGetValueAndDerivative(ThreadIdType threadId) const
{
  LossPerThreadStruct & loss = m_LossThreadStruct[threadId];
  loss.reset();

  /** Get a handle to the sample container. */
  ImageSampleContainerPointer sampleContainer = this->GetImageSampler()->GetOutput();
  const unsigned long         sampleContainerSize = sampleContainer->Size();
  /** Get the samples for this thread. */
  const unsigned long nrOfSamplesPerThreads = static_cast<unsigned long>(
    std::ceil(static_cast<double>(sampleContainerSize) / static_cast<double>(Self::GetNumberOfWorkUnits())));

  const auto posBegin = std::min<size_t>(nrOfSamplesPerThreads * threadId, sampleContainerSize);
  const auto posEnd = std::min<size_t>(nrOfSamplesPerThreads * (threadId + 1), sampleContainerSize);

  /** Create iterator over the sample container. */
  const auto beginOfSampleContainer = sampleContainer->cbegin();
  const auto threaderBegin = beginOfSampleContainer + posBegin;
  const auto threaderEnd = beginOfSampleContainer + posEnd;

  std::vector<FixedImagePointType> fixedPoints;
  fixedPoints.reserve(nrOfSamplesPerThreads);
  for (auto threaderIter = threaderBegin; threaderIter != threaderEnd; ++threaderIter)
  {
    fixedPoints.push_back(threaderIter->m_ImageCoordinates);
  }
  loss.m_NumberOfPixelsCounted += ComputeValueAndDerivativeStatic(fixedPoints, loss);
} // end ThreadedGetValueAndDerivative

/**
 * ******************* AfterThreadedGetValueAndDerivative *******************
 */
template <typename TFixedImage, typename TMovingImage>
void
ImpactImageToImageMetric<TFixedImage, TMovingImage>::AfterThreadedGetValueAndDerivative(
  MeasureType &    value,
  DerivativeType & derivative) const
{
  LossPerThreadStruct & loss = m_LossThreadStruct[0];
  for (ThreadIdType i = 1; i < Self::GetNumberOfWorkUnits(); ++i)
  {
    loss += m_LossThreadStruct[i];
  }
  Superclass::m_NumberOfPixelsCounted = loss.m_NumberOfPixelsCounted;
  /** Check if enough samples were valid. */
  this->CheckNumberOfSamples();

  /** Accumulate values. */
  value = loss.GetValue((m_NormalizeLosses ? &m_LossNormalization : nullptr));
  derivative = loss.GetDerivative((m_NormalizeLosses ? &m_LossNormalization : nullptr));
} // end AfterThreadedGetValueAndDerivative

} // end namespace itk

#endif // end #ifndef _itkImpactImageToImageMetric_hxx
