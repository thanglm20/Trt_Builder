/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef USER_BATCH_STREAM_H
#define USER_BATCH_STREAM_H

#include "NvInfer.h"
#include "../common/common.h"
#include "../common/BatchStream.h"
#include <algorithm>
#include <stdio.h>
#include <vector>




class UserBatchStream : public IBatchStream
{
public:
    UserBatchStream(
        int batchSize, int maxBatches, std::string prefix, std::string suffix, std::vector<std::string> directories)
        : mBatchSize(batchSize)
        , mMaxBatches(maxBatches)
        , mPrefix(prefix)
        , mSuffix(suffix)
        , mDataDir(directories)
    {
        FILE* file = fopen(locateFile(mPrefix + std::string("0") + mSuffix, mDataDir).c_str(), "rb");
        ASSERT(file != nullptr);
        int d[4];
        size_t readSize = fread(d, sizeof(int), 4, file);
        ASSERT(readSize == 4);
        mDims.nbDims = 4;  // The number of dimensions.
        mDims.d[0] = d[0]; // Batch Size
        mDims.d[1] = d[1]; // Channels
        mDims.d[2] = d[2]; // Height
        mDims.d[3] = d[3]; // Width
        std::cout << "==================================================" << std::endl;
        std::cout << "Calibration dimensions: "
                    << "\nbatchsize: " << mDims.d[0]
                    << "\nchannels: " << mDims.d[1]
                    << "\nheight: " << mDims.d[2]
                    << "\nwidth: " << mDims.d[3]
                    << std::endl;
        std::cout << "==================================================" << std::endl;

        ASSERT(mDims.d[0] > 0 && mDims.d[1] > 0 && mDims.d[2] > 0 && mDims.d[3] > 0);
        fclose(file);

        mImageSize = mDims.d[1] * mDims.d[2] * mDims.d[3];
        mBatch.resize(mBatchSize * mImageSize, 0);
        mLabels.resize(mBatchSize, 0);
        mFileBatch.resize(mDims.d[0] * mImageSize, 0);
        mFileLabels.resize(mDims.d[0], 0);
        reset(0);
    }

    UserBatchStream(int batchSize, int maxBatches, std::string prefix, std::vector<std::string> directories)
        : UserBatchStream(batchSize, maxBatches, prefix, ".batch", directories)
    {

    }


    // Resets data members
    void reset(int firstBatch) override
    {
        mBatchCount = 0;
        mFileCount = 0;
        mFileBatchPos = mDims.d[0];
        skip(firstBatch);
    }

    // Advance to next batch and return true, or return false if there is no batch left.
    bool next() override
    {
        if (mBatchCount == mMaxBatches)
        {
            return false;
        }

        for (int csize = 1, batchPos = 0; batchPos < mBatchSize; batchPos += csize, mFileBatchPos += csize)
        {
            ASSERT(mFileBatchPos > 0 && mFileBatchPos <= mDims.d[0]);
            if (mFileBatchPos == mDims.d[0] && !update())
            {
                return false;
            }

            // copy the smaller of: elements left to fulfill the request, or elements left in the file buffer.
            csize = std::min(mBatchSize - batchPos, mDims.d[0] - mFileBatchPos);
            std::copy_n(
                getFileBatch() + mFileBatchPos * mImageSize, csize * mImageSize, getBatch() + batchPos * mImageSize);
            std::copy_n(getFileLabels() + mFileBatchPos, csize, getLabels() + batchPos);
        }
        mBatchCount++;
        return true;
    }
    // Skips the batches
    void skip(int skipCount) override
    {
        if (mBatchSize >= mDims.d[0] && mBatchSize % mDims.d[0] == 0 && mFileBatchPos == mDims.d[0])
        {
            mFileCount += skipCount * mBatchSize / mDims.d[0];
            return;
        }

        int x = mBatchCount;
        for (int i = 0; i < skipCount; i++)
        {
            next();
        }
        mBatchCount = x;
    }

    float* getBatch() override
    {
        return mBatch.data();
    }

    float* getLabels() override
    {
        return mLabels.data();
    }

    int getBatchesRead() const override
    {
        return mBatchCount;
    }

    int getBatchSize() const override
    {
        return mBatchSize;
    }

    nvinfer1::Dims getDims() const override
    {
        return mDims;
    }

private:
    float* getFileBatch()
    {
        return mFileBatch.data();
    }

    float* getFileLabels()
    {
        return mFileLabels.data();
    }

    bool update()
    {
        
        std::string inputFileName = locateFile(mPrefix + std::to_string(mFileCount++) + mSuffix, mDataDir);
        std::cout << "Calib batch: " << inputFileName << std::endl; 
        FILE* file = fopen(inputFileName.c_str(), "rb");
        if (!file)
        {
            return false;
        }

        int d[4];
        size_t readSize = fread(d, sizeof(int), 4, file);
        ASSERT(readSize == 4);
        ASSERT(mDims.d[0] == d[0] && mDims.d[1] == d[1] && mDims.d[2] == d[2] && mDims.d[3] == d[3]);
        size_t readInputCount = fread(getFileBatch(), sizeof(float), mDims.d[0] * mImageSize, file);
        ASSERT(readInputCount == size_t(mDims.d[0] * mImageSize));
        size_t readLabelCount = fread(getFileLabels(), sizeof(float), mDims.d[0], file);
        ASSERT(readLabelCount == 0 || readLabelCount == size_t(mDims.d[0]));

        fclose(file);
        
        
        mFileBatchPos = 0;
        return true;
    }

    int mBatchSize{0};
    int mMaxBatches{0};
    int mBatchCount{0};
    int mFileCount{0};
    int mFileBatchPos{0};
    int mImageSize{0};
    std::vector<float> mBatch;         //!< Data for the batch
    std::vector<float> mLabels;        //!< Labels for the batch
    std::vector<float> mFileBatch;     //!< List of image files
    std::vector<float> mFileLabels;    //!< List of label files
    std::string mPrefix;               //!< Batch file name prefix
    std::string mSuffix;               //!< Batch file name suffix
    nvinfer1::Dims mDims;              //!< Input dimensions
    std::vector<std::string> mDataDir; //!< Directories where the files can be found
};
#endif
