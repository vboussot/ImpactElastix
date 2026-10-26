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
#ifndef itkImpactJS_hxx
#define itkImpactJS_hxx

#include "itkImpactJS.h"

#include <emscripten.h>

// The JavaScript is stringified by the preprocessor: no template literal and no ??= (a trigraph) in it. Pointers
// are read with >>> so that an address past 2 GiB (MAXIMUM_MEMORY is 4 GiB) stays positive. The arrays are copied out
// of the heap before the first await: the caller's memory is its own again once the promise settles.

// clang-format off
EM_ASYNC_JS(int, impact_js_create, (const char * config), {
  const api = globalThis.konfaiImpactElastix;
  if (!api)
  {
    throw new Error("IMPACT: the page has not set globalThis.konfaiImpactElastix, the konfai-impact API the metric calls");
  }
  const level = await api.create(JSON.parse(UTF8ToString(config)));
  if (!Module.impactLevels)
  {
    Module.impactLevels = [];
  }
  const levels = Module.impactLevels;
  levels.push(level);
  return levels.length - 1;
});

EM_ASYNC_JS(int,
            impact_js_extract,
            (int            handle,
             int            side,
             int            model,
             const float *  image,
             int            x,
             int            y,
             int            z,
             const float *  stats,
             const double * direction,
             unsigned int * shapes,
             int            maxLayers),
            {
              const at = (pointer, bytes) => (pointer >>> 0) / bytes;
              const pixels = HEAPF32.slice(at(image, 4), at(image, 4) + x * y * z);
              const statistics = HEAPF32.slice(at(stats, 4), at(stats, 4) + 4);
              const cosines = HEAPF64.slice(at(direction, 8), at(direction, 8) + 9);
              const answer = await Module.impactLevels[handle].extract(
                side ? "moving" : "fixed", model, pixels, [ x, y, z ], statistics, cosines);
              if (answer.length % 4 !== 0 || answer.length / 4 > maxLayers)
              {
                throw new Error("IMPACT: extract gave " + answer.length + " shape entries, not 4 per kept layer");
              }
              HEAPU32.set(answer, at(shapes, 4));
              return answer.length / 4;
            });

EM_ASYNC_JS(int,
            impact_js_evaluate,
            (int                  handle,
             int                  n,
             int                  layers,
             const float *        fixed,
             const float *        moving,
             const unsigned int * channels,
             const unsigned int * counts,
             int                  gradient,
             double *             values,
             float *              gradients),
            {
              const at = (pointer, bytes) => (pointer >>> 0) / bytes;
              const size = layers * n * 3;
              const request = {
                n,
                fixed : HEAPF32.slice(at(fixed, 4), at(fixed, 4) + size),
                moving : HEAPF32.slice(at(moving, 4), at(moving, 4) + size),
                channels : null,
                gradient : gradient !== 0,
              };
              if (channels)
              {
                request.channels = [];
                let next = at(channels, 4);
                for (let l = 0; l < layers; l++)
                {
                  const count = HEAPU32[at(counts, 4) + l];
                  request.channels.push(HEAPU32.slice(next, next + count));
                  next += count;
                }
              }
              const answer = await Module.impactLevels[handle].evaluate(request);
              if (answer.values.length !== layers || (gradient && answer.gradient.length !== size))
              {
                throw new Error("IMPACT: evaluate gave other than one value and n x 3 gradients per kept layer");
              }
              HEAPF64.set(answer.values, at(values, 8));
              if (gradient)
              {
                HEAPF32.set(answer.gradient, at(gradients, 4));
              }
              return 0;
            });

EM_JS(void, impact_js_free, (int handle), {
  Module.impactLevels?.[handle]?.free?.();
  if (Module.impactLevels)
  {
    Module.impactLevels[handle] = null;
  }
});
// clang-format on

#endif // itkImpactJS_hxx
