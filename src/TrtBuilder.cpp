#include "TrtBuilder.hpp"
#include "UserBatchStream.h"
//!
//! \brief Creates the network, configures the builder and creates the network engine
//!
//! \details This function creates the network by parsing the caffe model and builds
//!          the engine that will be used to run the model (mEngine)
//!
//! \return Returns true if the engine was created successfully and false otherwise
//!
bool TrtBuilder::build(DataType dataType)
{

    auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
    if (!builder)
    {
        return false;
    }
    if ((dataType == DataType::kINT8 && !builder->platformHasFastInt8())
        || (dataType == DataType::kHALF && !builder->platformHasFastFp16()))
    {
        std::cout << "Platform does not support data format" << std::endl;
        return false;
    }

    const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (!network)
    {
        return false;
    }

    auto config = SampleUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config)
    {
        return false;
    }

     auto parser
        = SampleUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, sample::gLogger.getTRTLogger()));
    if (!parser)
    {
        return false;
    }

    auto constructed = constructNetwork(builder, network, config, parser, dataType);
    if (!constructed)
    {
        return false;
    }

    // ASSERT(network->getNbInputs() == 1);
    // mInputDims = network->getInput(0)->getDimensions();
    // ASSERT(mInputDims.nbDims == 3);

    return true;
}

//!
//! \brief Uses a caffe parser to create the network and marks the
//!        output layers
//!
//! \param network Pointer to the network that will be populated with the network
//!
//! \param builder Pointer to the engine builder
//!
bool TrtBuilder::constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
    SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
    SampleUniquePtr<nvonnxparser::IParser>& parser, DataType dataType)
{

    auto parsed = parser->parseFromFile(locateFile(mParams.onnxFileName, mParams.dataDirs).c_str(),
        static_cast<int>(sample::gLogger.getReportableSeverity()));
    if (!parsed)
    {
        std::cout << "[ERROR] parseFromFile"  << std::endl;
        return false;
    }
    // for (auto& s : mParams.outputTensorNames)
    // {
    //     network->markOutput(*blobNameToTensor->find(s.c_str()));
    // }
    std::cout << "[INFO] Parsed ONNX file successfully" << std::endl;
    // Calibrator life time needs to last until after the engine is built.
    std::unique_ptr<IInt8Calibrator> calibrator;

    config->setAvgTimingIterations(1);
    config->setMinTimingIterations(1);
    config->setMaxWorkspaceSize(4_GiB);
    if (dataType == DataType::kHALF)
    {
        std::cout << "[INFO] DataType::kHALF successfully" << std::endl;
        config->setFlag(BuilderFlag::kFP16);
    }
    if (dataType == DataType::kINT8)
    {
        std::cout << "[INFO] BuilderFlag::kINT8 successfully" << std::endl;
        config->setFlag(BuilderFlag::kINT8);
    }
    builder->setMaxBatchSize(mParams.batchSize);

    if (dataType == DataType::kINT8)
    {

        UserBatchStream calibrationStream(
            mParams.batchCalib, mParams.nbCalBatches, mParams.calibrationBatches, mParams.calibDir);
        calibrator.reset(new Int8EntropyCalibrator2<UserBatchStream>(
            calibrationStream, 0, mParams.networkName.c_str(), mParams.inputTensorNames[0].c_str()));
        config->setInt8Calibrator(calibrator.get());     
    }

    if (mParams.dlaCore >= 0)
    {
        samplesCommon::enableDLA(builder.get(), config.get(), mParams.dlaCore);
        if (mParams.batchSize > builder->getMaxDLABatchSize())
        {
            sample::gLogError << "Requested batch size " << mParams.batchSize
                              << " is greater than the max DLA batch size of " << builder->getMaxDLABatchSize()
                              << ". Reducing batch size accordingly." << std::endl;
            return false;
        }
    }

    // CUDA stream used for profiling by the builder.
    auto profileStream = samplesCommon::makeCudaStream();
    if (!profileStream)
    {
        return false;
    }
    config->setProfileStream(*profileStream);

    
    SampleUniquePtr<IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
    if (!plan)
    {
        return false;
    }

    SampleUniquePtr<IRuntime> runtime{createInferRuntime(sample::gLogger.getTRTLogger())};
    if (!runtime)
    {
        return false;
    }

    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
        runtime->deserializeCudaEngine(plan->data(), plan->size()), samplesCommon::InferDeleter());
    if (!mEngine)
    {
        return false;
    }

    std::ofstream engineFile( mParams.saveEngine, std::ios::binary);
    engineFile.write(static_cast<char*>(plan->data()), plan->size());
    std::cout << "[INFO] Saved engine: " << mParams.saveEngine << std::endl;
    std::cout << "Model has been converted to TRT successfully\n";
    return true;
}
