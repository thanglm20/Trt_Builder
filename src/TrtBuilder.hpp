#ifndef TrtBuilder_hpp
#define TrtBuilder_hpp

#include "../common/BatchStream.h"
#include "../common/EntropyCalibrator.h"

#include "../common/argsParser.h"
#include "../common/buffers.h"
#include "../common/common.h"
#include "../common/logger.h"
#include "../common/parserOnnxConfig.h"


#include "NvInfer.h"
#include <cuda_runtime_api.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>

using samplesCommon::SampleUniquePtr;

struct UserParams : public samplesCommon::OnnxSampleParams
{
    std::string networkName;
    std::string saveEngine;
    int nbCalBatches;               //!< The number of batches for calibration
    int batchCalib = 1;
    std::string calibrationBatches; //!< The path to calibration batches
    std::vector<std::string>  calibDir;
};

class TrtBuilder
{
public:
    TrtBuilder(const UserParams& params)
        : mParams(params)
        , mEngine(nullptr)
    {
    }

    //!
    //! \brief Function builds the network engine
    //!
    bool build(DataType dataType);


private:
    UserParams mParams; //!< The parameters for the sample.

    nvinfer1::Dims mInputDims;  //!< The dimensions of the input to the network.
    nvinfer1::Dims mOutputDims; //!< The dimensions of the output to the network.

    std::shared_ptr<nvinfer1::ICudaEngine> mEngine; //!< The TensorRT engine used to run the network

    //!
    //! \brief Parses an ONNX model for MNIST and creates a TensorRT network
    //!
    bool constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
    SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
    SampleUniquePtr<nvonnxparser::IParser>& parser, DataType dataType);

};


#endif