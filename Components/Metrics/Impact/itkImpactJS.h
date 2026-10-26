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
#ifndef itkImpactJS_h
#define itkImpactJS_h

/** The JavaScript IMPACT the browser build of the metric calls in place of libtorch: the object the page sets as
 * globalThis.konfaiImpactElastix (konfai-impact's API). Each call but impact_js_free is awaited through JSPI: elastix
 * is suspended until its promise settles. A refusal or an error of the JavaScript side is thrown as a JavaScript
 * exception, which ends the run with its message. */
extern "C"
{
  /** A level of the registration: `config` is its JSON (models, distances, mode, seed); returns its handle. */
  int
  impact_js_create(const char * config);

  /** The feature maps of `model` on one image (`side` 0 fixed, 1 moving), resampled on the model's grid `x` x `y` x
   * `z` (x fastest); `stats` [min, max, mean, sigma] and `direction` (row by row, rounded) of the original image.
   * Writes [C, x, y, z] per kept layer to `shapes` (room for `maxLayers`); returns the number of kept layers. */
  int
  impact_js_extract(int            handle,
                    int            side,
                    int            model,
                    const float *  image,
                    int            x,
                    int            y,
                    int            z,
                    const float *  stats,
                    const double * direction,
                    unsigned int * shapes,
                    int            maxLayers);

  /** The losses at `n` points over `layers` kept layers: `fixed` and `moving` hold each layer's continuous indices
   * (x, y, z) of the points in its fixed and moving maps; `channels`, when not null, each layer's drawn channels
   * (`counts` of them, one layer after the other). Writes each layer's sum of the points' losses to `values` and,
   * when `gradient`, d(loss of point i)/d(moving index of point i) to `gradients` [layers][n][3]. */
  int
  impact_js_evaluate(int                  handle,
                     int                  n,
                     int                  layers,
                     const float *        fixed,
                     const float *        moving,
                     const unsigned int * channels,
                     const unsigned int * counts,
                     int                  gradient,
                     double *             values,
                     float *              gradients);

  /** The level's resources released. */
  void
  impact_js_free(int handle);
}

#endif // itkImpactJS_h
