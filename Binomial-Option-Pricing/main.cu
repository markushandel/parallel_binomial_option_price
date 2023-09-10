#include <iostream>
#include "option.h"
#include <fstream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <stdlib.h>
#include <stdio.h>
#include <iterator>
#include <numeric>
#include <sstream>

#define TRIANGLE_WIDTH 7
#define TRIANGLE_LENGTH 8
#define THREAD_AMOUNT 32
#define THREAD_SIZE 32
#define OPTION_OFFSET 5
#define STEP 255
#define STEP_OFFSET 320
#define START_WIDTH 8//this is b in the trapezoidal
#define END_WIDTH 4 //this is b in the trapezoidal
#define MARGINAL_STEP 8

using namespace std;

// Print out error:
#define error(args...)                           \
    {                                            \
        string _s = #args;                       \
        replace(_s.begin(), _s.end(), ',', ' '); \
        stringstream _ss(_s);                    \
        istream_iterator<string> _it(_ss);       \
        err(_it, args);                          \
    }

void err(istream_iterator<string> it)
{
}

template <typename T, typename... Args>
void err(istream_iterator<string> it, T a, Args... args)
{
    cout << *it << " = " << a << '\n';
    err(++it, args...);
}

// debug container
template <typename T>
void debugContainer(T &t)
{
    for (auto x : t)
    {
        cout << x << ' ';
    }
    cout << '\n';
}

// use 32 threads to calculate one option
//this is the newest version of Kernel Call
__global__ void kernel_call(float *in, float *out, int N)
{
    __shared__ float treeValues[STEP_OFFSET+1];
    float localValues[START_WIDTH];
    int thread_id = threadIdx.x; // get my id;
    int block_id = blockIdx.x;
    int num_threads = blockDim.x;
    int num_blocks = gridDim.x;

    for (int op = 0; op < N; op += num_blocks)
    {
        int offset = (op + block_id) * OPTION_OFFSET;
        double K = in[offset];
        double T = in[offset + 1];     // time-to-maturity
        double sigma = in[offset + 2]; // underlying volatility
        double r = in[offset + 3];     // risk-free rate
        double s = in[offset + 4];     // risk-free rate

        // printf("offset, K,T,sig,r,s: %d %.3f %.3f %.3f %.3f %.3f\n", offset, K, T, sigma, r, s);

        double deltaT = T / STEP;
        double up = exp(sigma * sqrt(deltaT)); // by how much go up when goes up
        double down = 1 / up;
        double p0;
        p0 = (exp(r * deltaT) - down) / (up - down); // RNP up
        double p1 = 1 - p0;                          // RNP down
        // get exercice values at expiration
        for (int i = thread_id; i <= STEP; i += num_threads)
        {
            // going up i times out of step times
            // simplfied for eu options
            treeValues[i] = [&](double ss)
            {
                return max(0.0, ss - K);
            }(s * pow(up, 2.0 * i - STEP));
            // getExerciseValue(s * pow(up, 2.0 * i - STEP), T, T, K);
        }

        //int cycles = STEP / MARGINAL_STEP;

        for(int times = 0;times<40;times++){

            for (int i = 0; i <= START_WIDTH; i ++)
            {
                localValues[i] = treeValues[i+thread_id*END_WIDTH];
            }
            
            // move to ealier times
            double t = T; // current time

            //__syncthreads();

            for (int j = MARGINAL_STEP-1; j >= 0; j--)
            {
                //int jOffset = j % 2 ? STEP_OFFSET : 0;
                //int jNoOffset = j % 2 ? 0 : STEP_OFFSET;
                t -= deltaT; // current time goes down by one timeStep
                            // considering we went up i times out of j times...
                for (int i = 0; i <= j; i++)
                {
                    double currentSpot = s * pow(up, 2 * i - j);
                    // exercise value at this time (this current spot, this time)
                    // eu options so X = 0;
                    // double exercise = 0;
                    // getExerciseValue(currentSpot, t, T, K);
                    //  at this node, tree values are...
                    //treeValues[i + jNoOffset] = (p0 * treeValues[i + 1 + jOffset] + p1 * treeValues[i + jOffset]) * exp(-r * deltaT);
                    localValues[i] = (p0 * localValues[i + 1] + p1 * localValues[i]) * exp(-r * deltaT);
                }
            }
            for(int i = 0; i <= END_WIDTH; i++)
            {
                treeValues[i+thread_id*END_WIDTH] = localValues[i];
            }
            __syncthreads();

        }


        // write data to out
        if (thread_id == 0)
        {
            // printf("treeValues[0]: %.3f %.3f %d\n", treeValues[0], treeValues[(STEP + 2)], op + block_id);
            out[op + block_id] = treeValues[0];
        }
    }
}

//this is the version before
__global__ void kernel_call_v1(float *in, float *out, int N)
{
    __shared__ float treeValues[STEP_OFFSET * 2];
    int thread_id = threadIdx.x; // get my id;
    int block_id = blockIdx.x;
    int num_threads = blockDim.x;
    int num_blocks = gridDim.x;

    for (int op = 0; op < N; op += num_blocks)
    {
        int offset = (op + block_id) * OPTION_OFFSET;
        double K = in[offset];
        double T = in[offset + 1];     // time-to-maturity
        double sigma = in[offset + 2]; // underlying volatility
        double r = in[offset + 3];     // risk-free rate
        double s = in[offset + 4];     

        // printf("offset, K,T,sig,r,s: %d %.3f %.3f %.3f %.3f %.3f\n", offset, K, T, sigma, r, s);

        double deltaT = T / STEP;
        double up = exp(sigma * sqrt(deltaT)); // by how much go up when goes up
        double down = 1 / up;
        double p0;
        p0 = (exp(r * deltaT) - down) / (up - down); // RNP up
        double p1 = 1 - p0;                          // RNP down
        // get exercice values at expiration
        for (int i = thread_id; i <= STEP; i += num_threads)
        {
            // going up i times out of step times
            // simplfied for eu options
            treeValues[i] = [&](double ss)
            {
                return max(0.0, ss - K);
            }(s * pow(up, 2.0 * i - STEP));
            // getExerciseValue(s * pow(up, 2.0 * i - STEP), T, T, K);
        }
        // move to ealier times
        double t = T; // current time

        __syncthreads();

        for (int j = STEP - 1; j >= 0; j--)
        {
            int jOffset = j % 2 ? STEP_OFFSET : 0;
            int jNoOffset = j % 2 ? 0 : STEP_OFFSET;
            t -= deltaT; // current time goes down by one timeStep
                         // considering we went up i times out of j times...
            for (int i = thread_id; i <= j; i += num_threads)
            {
                double currentSpot = s * pow(up, 2 * i - j);
                // exercise value at this time (this current spot, this time)
                // eu options so X = 0;
                // double exercise = 0;
                // getExerciseValue(currentSpot, t, T, K);
                //  at this node, tree values are...
                treeValues[i + jNoOffset] = (p0 * treeValues[i + 1 + jOffset] + p1 * treeValues[i + jOffset]) * exp(-r * deltaT);
            }
            __syncthreads();
        }
        // write data to out
        if (thread_id == 0)
        {
            // printf("treeValues[0]: %.3f %.3f %d\n", treeValues[0], treeValues[(STEP + 2)], op + block_id);
            out[op + block_id] = treeValues[STEP_OFFSET];
        }
    }
}

//this is the version before
__global__ void kernel_call_triangle(float *in, float *out, int N)
{
    int treeValueSize = TRIANGLE_LENGTH * THREAD_AMOUNT;
    __shared__ float treeValues[TRIANGLE_LENGTH * THREAD_AMOUNT];
    int thread_id = threadIdx.x; // get my id;
    int block_id = blockIdx.x;
    int num_blocks = gridDim.x;
    int totalNodesAtExpiration = STEP + 1;
    int nodesPerThread = totalNodesAtExpiration / THREAD_AMOUNT;
    int numberOfTriangles = (STEP) / (nodesPerThread - 1);
    float localTreeValues[TRIANGLE_LENGTH + 1];

    for (int op = 0; op < N; op += num_blocks)
    {
        int offset = (op + block_id) * OPTION_OFFSET;
        double K = in[offset];
        double T = in[offset + 1];     // time-to-maturity
        double sigma = in[offset + 2]; // underlying volatility
        double r = in[offset + 3];     // risk-free rate
        double s = in[offset + 4];     


        double deltaT = T / STEP;
        double up = exp(sigma * sqrt(deltaT)); // by how much go up when goes up
        double down = 1 / up;
        double p0;
        p0 = (exp(r * deltaT) - down) / (up - down); // RNP up
        double p1 = 1 - p0;                          // RNP down
                // this is the amount of triangles we will form for thread 0
        


        // First we calculate nodes at expiration. We only put the ones in shared mem that we need
        int startIndex = thread_id * nodesPerThread;    
        treeValues[startIndex] = [&](double ss)
            {
                return max(0.0, ss - K);
            }(s * pow(up, 2.0 * startIndex - STEP));

        // We try to reduce the amount of sharedmem calls so we also calculate the the ones for tree vals at startIndex again
        for (int i = startIndex; i < startIndex + nodesPerThread; i++) {
            localTreeValues[startIndex % nodesPerThread] = [&](double ss)
            {
                return max(0.0, ss - K);
            }(s * pow(up, 2.0 * i - STEP));
        }

        // PHASE A do as much as you can for one single thread
        for(int j = 1; j < TRIANGLE_WIDTH; j++)
            {
                // This is the val we need to put in shared mem
                treeValues[startIndex + j] = (p0 * localTreeValues[1] + p1 * localTreeValues[0]) * exp(-r * deltaT);

                // each col there will be less rows we can immeadiately calc
                for(int m = 0; m < nodesPerThread - j; m++)
                {
                    localTreeValues[m] = (p0 * localTreeValues[m + 1] + p1 * localTreeValues[m]) * exp(-r * deltaT);
                }
            }
        
        __syncthreads();

        // PHASE B 
        for(int j = 1; j < TRIANGLE_WIDTH; j++)
        {
            localTreeValues[nodesPerThread - 1] = (p0 * treeValues[(startIndex + nodesPerThread + j - 1) % (treeValueSize)] + p1 * localTreeValues[nodesPerThread - 1]) * exp(-r * deltaT);

            for(int m = nodesPerThread - 2; m >= nodesPerThread - j; m--)
                {
                    localTreeValues[m] = (p0 * localTreeValues[m + 1] + p1 * localTreeValues[m]) * exp(-r * deltaT);
                }
        }

        for (int k = 1; k < numberOfTriangles; k++)
        {   
            // PHASE A
            for(int j = 0; j < TRIANGLE_WIDTH; j++)
            {
                // This is the val we need to put in shared mem
                treeValues[startIndex + j] = (p0 * localTreeValues[1] + p1 * localTreeValues[0]) * exp(-r * deltaT);

                // each col there will be less rows we can immeadiately calc
                for(int m = 0; m < nodesPerThread - j; m++)
                {
                    localTreeValues[m] = (p0 * localTreeValues[m + 1] + p1 * localTreeValues[m]) * exp(-r * deltaT);
                }
            }

            // we sync here so that the vals in shared mem are not overwritten
            __syncthreads();

            // PHASE B
            for(int j = 1; j < TRIANGLE_WIDTH; j++)
            {
                localTreeValues[nodesPerThread - 1] = (p0 * treeValues[(startIndex + nodesPerThread + j - 1) % (treeValueSize)] + p1 * localTreeValues[nodesPerThread - 1]) * exp(-r * deltaT);

                for(int m = nodesPerThread - 2; m >= nodesPerThread - j; m--)
                    {
                        localTreeValues[m] = (p0 * localTreeValues[m + 1] + p1 * localTreeValues[m]) * exp(-r * deltaT);
                    }
            }

            // __syncthreads();
        }

        // write data to out
        if (thread_id == 0)
        {
            // printf("treeValues[0]: %.3f %.3f %d\n", treeValues[0], treeValues[(STEP + 2)], op + block_id);
            out[op + block_id] = localTreeValues[0];
        }
    }
}



void runGPUPricing(vector<EuropeanCall> &calls, vector<double> &res, int N, int numBlocks)
{
    // GPU version
    float *host_in, *host_out;
    float *dev_in, *dev_out;

    // prepare data
    // create buffer on host
    host_in = (float *)malloc(N * sizeof(EuropeanCall));
    host_out = (float *)malloc(N * sizeof(float));

    for (int i = 0; i < N; i++)
    {
        // we have 5 parameters
        int offset = i * 5;
        host_in[offset + 0] = calls[i].K;
        host_in[offset + 1] = calls[i].T;
        host_in[offset + 2] = calls[i].sigma;
        host_in[offset + 3] = calls[i].r;
        host_in[offset + 4] = 1400;
    }

    cudaError_t err = cudaMalloc(&dev_in, N * sizeof(EuropeanCall));
    if (err != cudaSuccess)
    {
        cout << "Dev Memory not allocated" << endl;
        exit(-1);
    }

    err = cudaMalloc(&dev_out, N * sizeof(float));
    if (err != cudaSuccess)
    {
        cout << "Dev Memory not allocated" << endl;
        exit(-1);
    }

    // transfer the data
    cudaMemcpy(dev_in, host_in, N * sizeof(EuropeanCall), cudaMemcpyHostToDevice);

    cudaEvent_t st2, et2;
    cudaEventCreate(&st2);
    cudaEventCreate(&et2);

    cudaEventRecord(st2);
    // kernel call
    kernel_call_triangle<<<numBlocks, 32>>>(dev_in, dev_out, N);
    cudaEventRecord(et2);

    // copy result out
    cudaMemcpy(host_out, dev_out, N * sizeof(float), cudaMemcpyDeviceToHost);

    // error(host_out[0], host_out[1], host_out[2]);
    // cout << host_out[0] << ' ' << host_out[1] << ' ' << host_out[2] << endl;

    for (int i = 0; i < N; i++)
    {
        res[i] = host_out[i];
    }

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, st2, et2);
    // cout << "GPU Kernel time running " << numThreads << " threads, time: " << milliseconds << " ms" << endl;
    cout << N << "," << numBlocks << "," << milliseconds << "," << endl;
    // free all resources
    free(host_in);
    free(host_out);
    cudaFree(dev_in);
    cudaFree(dev_out);
}

void runPricing(vector<EuropeanCall> &calls, vector<double> &res, int N)
{

    auto start = std::chrono::high_resolution_clock::now();

    // for each call option, calculate the binomial price
    for (int i = 0; i < N; i++)
    {
        // cout << "calc optoin " << i << endl;
        res[i] = calls[i].getBinomialTreeValue(1400, STEP);
    }

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> kernel_time = end - start;
    std::cout << "Kernel time: for " << N << " " << kernel_time.count() << " seconds\n";
}

int main(int argc, const char *argv[])
{
    // Examples
    ExtendibleCall exted(73, 0.25, 0.25, 0.02);
    std::cout << "Extendible Call: " << exted.getBinomialTreeValue(75, 1000) << std::endl;
    // EuropeanCall europCall(1470.37, 0.5, 0.12, 0.02);
    // std::cout << "European Call: " << europCall.getBlackScholesValue(1400) << std::endl;
    // EuropeanPut europPut(63.75, 1, 0.2, 0.04);
    // double val = europPut.getBlackScholesValue(75);
    // std::cout << "European put: " << val << std::endl;

    // CompoundEuropeanCall compEurCall(2.5, 3.0 / 12, 0.25, 0.01);
    // std::cout << "Compound Call" << compEurCall.getBinomialTreeValue(80, 30) << std::endl;

    // ReloadableCallOption reloadCall(110, 2, 0.32, 0.0195);
    // std::cout << " Price of reloadable call with initial price 100: " << reloadCall.getBinomialTreeValue(100, 200);

    int N = 5000, Test = 5000;
    vector<EuropeanCall> eucalls;
    // eucalls.reserve(N);
    vector<double> res(N), gpuRes(N);
    for (int i = 0; i < N; i++)
    {
        eucalls.push_back(EuropeanCall(1470.37 - i, 0.5, 0.12, 0.02));
    }

    // baseline version
    for (int n = N; n <= Test; n += 1000)
    {
        runPricing(eucalls, res, n);
    }

    cout << "GPU section" << endl;
    cout << "dataSize,numThreads,gpuKernelTime,wallClockTime" << endl;
    for (int blocks = 32; blocks <= 32; blocks *= 2)
    {
        for (int n = 1000; n <= Test; n += 1000)
        {
            runGPUPricing(eucalls, gpuRes, n, blocks);
        }
    }


    // compare result value
    bool matched = true;
    for (int i = 0; i < Test; i++)
    {
        // error(i, gpuRes[i], res[i]);
        if (abs(gpuRes[i] - res[i]) > 0.01)
        {
            matched = false;
            break;
        }
    }
    cout << (matched ? "YES" : "NO") << '\n';

    return 0;
}
