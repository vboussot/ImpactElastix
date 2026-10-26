if(EMSCRIPTEN)
  # The browser build: linked in, no LibTorch. The models and the distances run in the page (itkImpactJS.h), and
  # ITKIMPACT gives only its torch-free headers and model configuration.
  ADD_ELXCOMPONENT(ImpactMetric OFF
   elxImpactMetric.cxx
   elxImpactMetric.h
   elxImpactMetric.hxx
   itkImpactImageToImageMetricJS.h
   itkImpactImageToImageMetricJS.hxx
   itkImpactJS.h
   itkImpactJS.hxx
   )
else()
  ADD_ELXCOMPONENT(ImpactMetric OFF PLUGIN
   elxImpactMetric.cxx
   elxImpactMetric.h
   elxImpactMetric.hxx
   itkImpactImageToImageMetric.h
   itkImpactImageToImageMetric.hxx
   )
endif()

if(USE_ImpactMetric AND EMSCRIPTEN)
    set(ITKIMPACT_INCLUDE_DIR "" CACHE PATH "ITKIMPACT include directory (itkImpactModelConfiguration.h, ...)")
    if(NOT EXISTS "${ITKIMPACT_INCLUDE_DIR}/itkImpactLossNormalization.h")
        message(FATAL_ERROR "USE_ImpactMetric is ON: set ITKIMPACT_INCLUDE_DIR to the include/ directory of "
            "ITKIMPACT 0.1.6 or later, not '${ITKIMPACT_INCLUDE_DIR}'.")
    endif()
    target_include_directories(ImpactMetric PRIVATE ${ITKIMPACT_INCLUDE_DIR})
    target_sources(ImpactMetric PRIVATE ${ITKIMPACT_INCLUDE_DIR}/../src/itkImpactModelConfiguration.cxx)
    target_compile_definitions(ImpactMetric PRIVATE ITK_IMPACT_WITHOUT_TORCH)
elseif(USE_ImpactMetric)
    find_package(Torch REQUIRED)
    # IMPACT losses, model configuration and online inference come from the ITKIMPACT
    # remote module (one-way dependency Elastix -> ITKIMPACT).
    set(ITKIMPACT_INCLUDE_DIR "" CACHE PATH
        "ITKIMPACT backend include directory (provides ImpactLoss.h, itkImpactModelConfiguration.h, ...)")
    # Check it before use: an unset ITKIMPACT_INCLUDE_DIR collapses the path below to
    # /src/itkImpactModelConfiguration.cxx, and CMake then reports a missing source file rather than
    # the missing setting that caused it.
    if(NOT EXISTS "${ITKIMPACT_INCLUDE_DIR}/ImpactLoss.h")
        message(FATAL_ERROR
            "USE_ImpactMetric is ON, so the ITKIMPACT backend is required, but "
            "ITKIMPACT_INCLUDE_DIR does not point at it: '${ITKIMPACT_INCLUDE_DIR}'. "
            "Clone https://github.com/InsightSoftwareConsortium/ITKIMPACT and set "
            "ITKIMPACT_INCLUDE_DIR to its include/ directory.")
    endif()
    target_include_directories(ImpactMetric PRIVATE ${ITKIMPACT_INCLUDE_DIR})
    target_sources(ImpactMetric PRIVATE ${ITKIMPACT_INCLUDE_DIR}/../src/itkImpactModelConfiguration.cxx)
    target_link_libraries(ImpactMetric PUBLIC ${TORCH_LIBRARIES})
endif()
