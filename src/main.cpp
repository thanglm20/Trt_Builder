




#include "../common/BatchStream.h"
#include "../common/EntropyCalibrator.h"
#include "../common/argsParser.h"
#include "../common/buffers.h"
#include "../common/common.h"
#include "../common/logger.h"
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include "TrtBuilder.hpp"
#include "UserBatchStream.h"

UserParams initializeSampleParams()
{
    UserParams params;
    std::string pathModel = "D:\\ThangLM\\models\\220509";
    params.dataDirs.emplace_back(pathModel);
    params.onnxFileName = "RepVGG_A2-210603-050157-cocokp-edge321-o10s.pkl.epoch250.449x257.onnx";

    params.inputTensorNames.push_back("input_batch");
    params.outputTensorNames.push_back("cif");
    params.outputTensorNames.push_back("caf");
    params.batchSize = 1;
    // params.dlaCore = 32;
    params.int8 = true;
    params.fp16 = false;
    params.networkName = "AsillaPose";
    params.saveEngine = pathModel + "\\" + "RepVGG_A2-210603-050157-cocokp-edge321-o10s.pkl.epoch250.449x257_int8.engine";
    // for calib
    params.batchCalib = 1;
    params.nbCalBatches = 500;
    params.calibrationBatches = "batch_calibration";
    params.calibDir.push_back("D:\\ThangLM\\CalibFiles\\batch_calib_train");
    // params.calibDir.push_back("D:\\ThangLM\\CalibFiles\\val2017_calib");
    return params;
}


int main()
{
    std::cout << "============ Building TRT model from ONNX =============" << std::endl;
    sample::gLogInfo << "TensorRT version: " << NV_TENSORRT_MAJOR << "." << NV_TENSORRT_MINOR
        << "." << NV_TENSORRT_PATCH << std::endl;
    initLibNvInferPlugins(&sample::gLogger.getTRTLogger(), "");
    
    UserParams  params = initializeSampleParams(); 
    TrtBuilder builder(params);
    DataType datatype;
    if(params.int8)
        datatype = DataType::kINT8;
    else if(params.fp16)
        datatype = DataType::kHALF;
    else 
        datatype = DataType::kFLOAT;
    if (!samplesCommon::isDataTypeSupported(datatype))
    {
        sample::gLogWarning << "Skipping " 
                            << " Since the platform does not support this data type." << std::endl;
        return -1;
    }
    if (!builder.build(datatype))
    {
        perror("Build error\n"); 
    }
    std::cout << "DONE" << std::endl;
    return 0;
}