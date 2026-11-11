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
#ifndef itkComputeJacobianTerms_hxx
#define itkComputeJacobianTerms_hxx

#include "itkComputeJacobianTerms.h"

#include <vnl/vnl_math.h>
#include <vnl/vnl_fastops.h>
#include <vnl/vnl_diag_matrix.h>
#include <vnl/vnl_sparse_matrix.h>

#include <itkMultiThreaderBase.h>

#include <algorithm>
#include <cassert>
#include <vector>

namespace itk
{

/**
 * ************************* Compute ************************
 */

template <typename TFixedImage, typename TTransform>
auto
ComputeJacobianTerms<TFixedImage, TTransform>::Compute() const -> Terms
{
  /** This function computes four terms needed for the automatic parameter
   * estimation. The equation number refers to the IJCV paper.
   * Term 1: TrC, which is the trace of the covariance matrix, needed in (34):
   *    C = 1/n \sum_{i=1}^n J_i^T J_i    (25)
   *    with n the number of samples, J_i the Jacobian of the i-th sample.
   * Term 2: TrCC, which is the Frobenius norm of C, needed in (60):
   *    ||C||_F^2 = trace( C^T C )
   * To compute equations (47) and (54) we need the four sub-terms:
   *    A: trace( J_j C J_j^T )  in (47)
   *    B: || J_j C J_j^T ||_F   in (47)
   *    C: || J_j ||_F^2         in (54)
   *    D: || J_j J_j^T ||_F     in (54)
   * Term 3: maxJJ, see (47)
   * Term 4: maxJCJ, see (54)
   */

  using FixedImagePointType = typename TFixedImage::PointType;
  using JacobianType = typename TTransform::JacobianType;
  using NumberOfParametersType = typename TTransform::NumberOfParametersType;
  using NonZeroJacobianIndicesType = typename TTransform::NonZeroJacobianIndicesType;

  /** Get samples. */
  const std::vector<ImageSampleType> samples = SampleFixedImageForJacobianTerms();
  const SizeValueType                nrofsamples = samples.size();
  const auto                         n = static_cast<double>(nrofsamples);

  /** Get the number of parameters. */
  const auto numberOfParameters = static_cast<unsigned int>(m_Transform->GetNumberOfParameters());

  static constexpr unsigned int outdim{ TTransform::OutputSpaceDimension };

  /** Variables for nonzerojacobian indices and the Jacobian. */
  const NumberOfParametersType sizejacind = m_Transform->GetNumberOfNonZeroJacobianIndices();
  NonZeroJacobianIndicesType   jacind(sizejacind);

  /** A transform may leave the Jacobian of a sample as it is, off the grid of a B-spline: the Jacobian
   * of that sample is then the one of the sample before it. Each loop over the samples starts with
   * the Jacobian the loop before it leaves, and walks the samples in their order.
   */
  JacobianType prevjacj(outdim, sizejacind, 0.0);
  assert((sizejacind > 0) && (jacind.front() == 0));

  using FreqPairType = std::pair<unsigned int, unsigned int>;
  std::vector<FreqPairType> difHist2;

  {
    /** `difHist` is a histogram of absolute parameterNrDifferences that
     * occur in the nonzerojacobianindex vectors.
     * `difHist2` is another way of storing the histogram, as a vector
     * of pairs. pair.first = Frequency, pair.second = parameterNrDifference.
     * This is useful for sorting.
     */
    std::vector<unsigned int> difHist(numberOfParameters);

    /** Try to guess the band structure of the covariance matrix.
     * A 'band' is a series of elements cov(p,q) with constant q-p.
     * In the loop below, on a few positions in the image the Jacobian
     * is computed. The nonzerojacobianindices are inspected to figure out
     * which values of q-p occur often. This is done by making a histogram.
     * The histogram is then sorted and the most occurring bands
     * are determined. The covariance elements in these bands will not
     * be stored in the sparse matrix structure 'cov', but in the band
     * matrix 'bandcov', which is much faster.
     * Only after the bandcov and cov have been filled (by looping over
     * all Jacobian measurements in the sample container, the bandcov
     * matrix is injected in the cov matrix, for easy further calculations,
     * and the bandcov matrix is deleted.
     */
    unsigned int onezero = 0;
    for (unsigned int s = 0; s < m_NumberOfBandStructureSamples; ++s)
    {
      /** Semi-randomly get some samples from the sample container. */
      const unsigned int samplenr = (s + 1) * nrofsamples / (m_NumberOfBandStructureSamples + 2 + onezero);
      onezero = 1 - onezero; // introduces semi-randomness

      /** Read fixed coordinates and get Jacobian J_j. */
      const FixedImagePointType & point = samples[samplenr].m_ImageCoordinates;
      m_Transform->GetJacobian(point, prevjacj, jacind);

      /** Skip invalid Jacobians in the beginning, if any. */
      if (sizejacind > 1 && jacind[0] == jacind[1])
      {
        continue;
      }

      /** Fill the histogram of parameter nr differences. */
      for (unsigned int i = 0; i < sizejacind; ++i)
      {
        const int jacindi = static_cast<int>(jacind[i]);
        for (unsigned int j = i; j < sizejacind; ++j)
        {
          const int jacindj = static_cast<int>(jacind[j]);
          difHist[static_cast<unsigned int>(std::abs(jacindj - jacindi))]++;
        }
      }
    }

    /** Copy the nonzero elements of the difHist to a vector pairs. */
    for (unsigned int p = 0; p < numberOfParameters; ++p)
    {
      const unsigned int freq = difHist[p];
      if (freq != 0)
      {
        difHist2.push_back(FreqPairType(freq, p));
      }
    }
  } // End of scope of difHist.

  /** Compute the number of bands. */
  const unsigned int bandcovsize = std::min(m_MaxBandCovSize, static_cast<unsigned int>(difHist2.size()));

  /** Maps parameterNrDifference (q-p) to colnr in bandcov. */
  std::vector<unsigned int> bandcovMap(numberOfParameters, bandcovsize);
  /** Maps colnr in bandcov to parameterNrDifference (q-p). */
  std::vector<unsigned int> bandcovMap2(bandcovsize, numberOfParameters);

  /** Sort the difHist2 based on the frequencies. */
  std::sort(difHist2.begin(), difHist2.end());

  /** Determine the bands that are expected to be most dominant. */
  auto difHist2It = difHist2.end();
  for (unsigned int b = 0; b < bandcovsize; ++b)
  {
    --difHist2It;
    bandcovMap[difHist2It->second] = b;
    bandcovMap2[b] = difHist2It->second;
  }

  using CovarianceValueType = double;
  using CovarianceMatrixType = vnl_matrix<CovarianceValueType>;
  using DiagCovarianceMatrixType = vnl_diag_matrix<CovarianceValueType>;

  /** Initialize covariance matrix. Sparse, diagonal, and band form. */
  vnl_sparse_matrix<CovarianceValueType> cov(numberOfParameters, numberOfParameters);
  DiagCovarianceMatrixType               diagcov(numberOfParameters, 0.0);

  /** The Jacobian the samples of term 1 leave. */
  JacobianType prevjacjOfMaxima;

  {
    /** Initialize band matrix. */
    CovarianceMatrixType bandcov(numberOfParameters, bandcovsize, 0.0);

    /**
     *    TERM 1
     *
     * Loop over image and compute Jacobian.
     * Compute C = 1/n \sum_i J_i^T J_i
     * Possibly apply scaling afterwards.
     *
     * An element of C is the sum of what the samples give it, taken in their order. The rows of C
     * are shared out between the threads: each one walks all the samples, in order, and sums the
     * rows that are its own, so that an element is summed as it is on one thread.
     */
    const auto          threader = MultiThreaderBase::New();
    const SizeValueType numberOfOwners = threader->GetNumberOfWorkUnits();
    threader->ParallelizeArray(
      0,
      numberOfOwners,
      [&](const SizeValueType owner) {
        JacobianType               jacj(prevjacj);
        NonZeroJacobianIndicesType jacind(sizejacind);
        /** The indices of no sample: the first sample starts a sum. */
        NonZeroJacobianIndicesType prevjacind(sizejacind, numberOfParameters);

        /** J'J of the samples in a row that have the same nonzero Jacobian indices: the rows of this thread. */
        CovarianceMatrixType jactjac(sizejacind, sizejacind, 0.0);

        /** Where the parameters of this thread are among the nonzero Jacobian indices. */
        std::vector<unsigned int> own;

        /** Update covariance matrix. */
        const auto updateCovariance = [&] {
          for (const unsigned int pi : own)
          {
            const unsigned int p = prevjacind[pi];
            for (unsigned int qi = 0; qi < sizejacind; ++qi)
            {
              const unsigned int q = prevjacind[qi];
              if (q >= p)
              {
                const double tempval = jactjac[pi][qi] / n;
                if (std::abs(tempval) > 1e-14)
                {
                  const unsigned int bandindex = bandcovMap[q - p];
                  if (bandindex < bandcovsize)
                  {
                    bandcov[p][bandindex] += tempval;
                  }
                  else
                  {
                    cov(p, q) += tempval;
                  }
                }
              }
            } // qi
          } // pi
        };

        for (const auto & sample : samples)
        {
          /** Read fixed coordinates and get Jacobian J_j. */
          const FixedImagePointType & point = sample.m_ImageCoordinates;
          m_Transform->GetJacobian(point, jacj, jacind);

          /** Skip invalid Jacobians in the beginning, if any. */
          if (sizejacind > 1 && jacind[0] == jacind[1])
          {
            continue;
          }

          /** With the indices of the sample before, J_j^T J_j adds to the sum; with others, the sum goes
           * to the covariance matrix and starts again.
           */
          const bool sameIndices = jacind == prevjacind;
          if (!sameIndices)
          {
            updateCovariance();
            prevjacind = jacind;
            own.clear();
            for (unsigned int pi = 0; pi < sizejacind; ++pi)
            {
              if (static_cast<unsigned int>(jacind[pi]) % static_cast<unsigned int>(numberOfOwners) == owner)
              {
                own.push_back(pi);
              }
            }
          }
          for (const unsigned int pi : own)
          {
            double * const row = jactjac[pi];
            for (unsigned int qi = 0; qi < sizejacind; ++qi)
            {
              if (jacind[qi] >= jacind[pi])
              {
                /** An element of J_j^T J_j, summed over the rows of J_j as vnl_fastops sums it. */
                double accum = 0;
                for (unsigned int k = 0; k < outdim; ++k)
                {
                  accum += jacj[k][pi] * jacj[k][qi];
                }
                row[qi] = sameIndices ? row[qi] + accum : accum;
              }
            }
          }
        } // end iter loop: end computation of covariance matrix

        /** Update covariance matrix once again to include last jactjac updates. */
        updateCovariance();

        if (owner == 0)
        {
          prevjacjOfMaxima = jacj;
        }

        /** Copy the bandmatrix into the sparse matrix.
         * \todo: perhaps work further with this bandmatrix instead.
         */
        for (unsigned int p = owner; p < numberOfParameters; p += numberOfOwners)
        {
          for (unsigned int b = 0; b < bandcovsize; ++b)
          {
            const double tempval = bandcov[p][b];
            if (std::abs(tempval) > 1e-14)
            {
              const unsigned int q = p + bandcovMap2[b];
              cov(p, q) = tempval;
            }
          }
        }
      },
      nullptr);
  } // End of scope of `bandcov`.

  /** Apply scales. the use of m_Scales maybe something wrong. */
  if (m_UseScales)
  {
    for (unsigned int p = 0; p < numberOfParameters; ++p)
    {
      cov.scale_row(p, 1.0 / m_Scales[p]);
    }
    /**  \todo: this might be faster with get_row instead of the iterator */
    cov.reset();
    while (cov.next())
    {
      const int col = cov.getcolumn();
      cov(cov.getrow(), col) /= m_Scales[col];
    }
  }

  /** Compute TrC = trace(C), and diagcov. */
  double TrC = 0.0;
  for (unsigned int p = 0; p < numberOfParameters; ++p)
  {
    // Do cov.get(p, p) instead of cov(p, p) to avoid creation of an entry that just has zero.
    const CovarianceValueType covpp = cov.get(p, p);
    TrC += covpp;
    diagcov[p] = covpp;
  }

  /**
   *    TERM 2
   *
   * Compute TrCC = ||C||_F^2.
   */
  cov.reset();
  double TrCC = 0.0;
  while (cov.next())
  {
    TrCC += vnl_math::sqr(cov.value());
  }

  /** Symmetry: multiply by 2 and subtract sumsqr(diagcov). */
  TrCC *= 2.0;
  TrCC -= diagcov.diagonal().squared_magnitude();

  /**
   *    TERM 3 and 4
   *
   * Compute maxJJ and maxJCJ
   * \li maxJJ = max_j [ ||J_j||_F^2 + 2\sqrt{2} || J_j J_j^T ||_F ]
   * \li maxJCJ = max_j [ Tr( J_j C J_j^T ) + 2\sqrt{2} || J_j C J_j^T ||_F ]
   */
  const double sqrt2 = std::sqrt(2.0);

  /** A maximum does not depend on the order it is taken in: the samples are shared out between the
   * threads, each with its own workspace. A thread walks the samples before its own for the
   * Jacobian they leave.
   */
  const auto                threader = MultiThreaderBase::New();
  const SizeValueType       numberOfChunks = threader->GetNumberOfWorkUnits();
  std::vector<double>       maxJJOfChunk(numberOfChunks, 0.0);
  std::vector<double>       maxJCJOfChunk(numberOfChunks, 0.0);
  threader->ParallelizeArray(
    0,
    numberOfChunks,
    [&](const SizeValueType chunk) {
      JacobianType               jacj(prevjacjOfMaxima);
      NonZeroJacobianIndicesType jacind(sizejacind);
      JacobianType               jacjjacj(outdim, outdim);
      JacobianType               jacjcov(outdim, sizejacind);
      DiagCovarianceMatrixType   diagcovsparse(sizejacind);
      JacobianType               jacjdiagcov(outdim, sizejacind);
      JacobianType               jacjdiagcovjacj(outdim, outdim);
      JacobianType               jacjcovjacj(outdim, outdim);
      /** Where each parameter is among the nonzero Jacobian indices of the sample; sizejacind elsewhere. */
      std::vector<SizeValueType> jacindExpanded(numberOfParameters, sizejacind);
      double                     maxJJ = 0.0;
      double                     maxJCJ = 0.0;

      const SizeValueType begin = chunk * nrofsamples / numberOfChunks;
      const SizeValueType end = (chunk + 1) * nrofsamples / numberOfChunks;
      for (SizeValueType s = 0; s < end; ++s)
      {
        /** Read fixed coordinates and get Jacobian. */
        const FixedImagePointType & point = samples[s].m_ImageCoordinates;
        m_Transform->GetJacobian(point, jacj, jacind);

        /** Apply scales, if necessary. */
        if (m_UseScales)
        {
          for (unsigned int pi = 0; pi < sizejacind; ++pi)
          {
            const unsigned int p = jacind[pi];
            jacj.scale_column(pi, 1.0 / m_Scales[p]);
          }
        }
        if (s < begin)
        {
          continue;
        }

        /** Compute 1st part of JJ: ||J_j||_F^2. */
        double JJ_j = vnl_math::sqr(jacj.frobenius_norm());

        /** Compute 2nd part of JJ: 2\sqrt{2} || J_j J_j^T ||_F. */
        vnl_fastops::ABt(jacjjacj, jacj, jacj);
        JJ_j += 2.0 * sqrt2 * jacjjacj.frobenius_norm();

        /** Max_j [JJ_j]. */
        maxJJ = std::max(maxJJ, JJ_j);

        /** Compute JCJ_j. */
        double JCJ_j = 0.0;

        /** J_j C = jacjC. */
        jacjcov.Fill(0.0);

        /** Store the nonzero Jacobian indices in a different format
         * and create the sparse diagcov.
         */
        for (unsigned int pi = 0; pi < sizejacind; ++pi)
        {
          const unsigned int p = jacind[pi];
          jacindExpanded[p] = pi;
          diagcovsparse[pi] = diagcov[p];
        }

        /** We below calculate jacjC = J_j cov^T, but later we will correct
         * for this using:
         * J C J' = J (cov + cov' - diag(cov')) J'.
         * (NB: cov now still contains only the upper triangular part of C)
         */
        for (unsigned int pi = 0; pi < sizejacind; ++pi)
        {
          /** Loop over row of the sparse cov matrix. */
          for (const auto & covRowEntry : cov.get_row(jacind[pi]))
          {
            const unsigned int q = covRowEntry.first;
            const unsigned int qi = jacindExpanded[q];

            if (qi < sizejacind)
            {
              /** If found, update the jacjC matrix. */
              const CovarianceValueType covElement = covRowEntry.second;
              for (unsigned int dx = 0; dx < outdim; ++dx)
              {
                jacjcov[dx][pi] += jacj[dx][qi] * covElement;
              } // dx
            } // if qi < sizejacind
          } // for covrow
        } // pi

        /** The indices of this sample leave the expanded format: only they were set. */
        for (unsigned int pi = 0; pi < sizejacind; ++pi)
        {
          jacindExpanded[jacind[pi]] = sizejacind;
        }

        /** J_j C J_j^T  = jacjCjacj.
         * But note that we actually compute J_j cov' J_j^T
         */
        vnl_fastops::ABt(jacjcovjacj, jacjcov, jacj);

        /** jacjCjacj = jacjCjacj+ jacjCjacj' - jacjdiagcovjacj */
        jacjdiagcov = jacj * diagcovsparse;
        vnl_fastops::ABt(jacjdiagcovjacj, jacjdiagcov, jacj);
        jacjcovjacj += jacjcovjacj.transpose();
        jacjcovjacj -= jacjdiagcovjacj;

        /** Compute 1st part of JCJ: Tr( J_j C J_j^T ). */
        for (unsigned int d = 0; d < outdim; ++d)
        {
          JCJ_j += jacjcovjacj[d][d];
        }

        /** Compute 2nd part of JCJ_j: 2 \sqrt{2} || J_j C J_j^T ||_F. */
        JCJ_j += 2.0 * sqrt2 * jacjcovjacj.frobenius_norm();

        /** Max_j [JCJ_j]. */
        maxJCJ = std::max(maxJCJ, JCJ_j);
      } // end loop over the samples of the chunk
      maxJJOfChunk[chunk] = maxJJ;
      maxJCJOfChunk[chunk] = maxJCJ;
    },
    nullptr);
  const double maxJJ = *std::max_element(maxJJOfChunk.begin(), maxJJOfChunk.end());
  const double maxJCJ = *std::max_element(maxJCJOfChunk.begin(), maxJCJOfChunk.end());

  /** Finalize progress information. */
  // progressObserver->PrintProgress( 1.0 );

  return Terms{ TrC, TrCC, maxJJ, maxJCJ };

} // end Compute()


/**
 * ************************* SampleFixedImageForJacobianTerms ************************
 */

template <typename TFixedImage, typename TTransform>
auto
ComputeJacobianTerms<TFixedImage, TTransform>::SampleFixedImageForJacobianTerms() const -> std::vector<ImageSampleType>
{
  /** Set up grid sampler. */
  const auto sampler = ImageGridSampler<TFixedImage>::New();
  sampler->SetInput(m_FixedImage);
  sampler->SetInputImageRegion(m_FixedImageRegion);
  sampler->SetMask(m_FixedImageMask);

  /** Determine grid spacing of sampler such that the desired
   * NumberOfJacobianMeasurements is achieved approximately.
   * Note that the actually obtained number of samples may be lower, due to masks.
   * This is taken into account at the end of this function.
   */
  sampler->SetNumberOfSamples(m_NumberOfJacobianMeasurements);

  /** Get samples and check the actually obtained number of samples. */
  sampler->Update();
  std::vector<ImageSampleType> & samples = Deref(sampler->GetOutput()).CastToSTLContainer();

  if (samples.empty())
  {
    itkExceptionMacro("No valid voxels (0/" << m_NumberOfJacobianMeasurements
                                            << ") found to estimate the AdaptiveStochasticGradientDescent parameters.");
  }
  return std::move(samples);

} // end SampleFixedImageForJacobianTerms()


} // end namespace itk

#endif // end #ifndef itkComputeJacobianTerms_hxx
