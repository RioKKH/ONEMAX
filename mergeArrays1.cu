#include <iostream>
#include <cuda_runtime.h>

__global__ void mergeArraysKernel(float* A, float* B, float* C, int n, int i, int j)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // Warpダイバージェンスを避けるための条件
    bool isA = (idx < i) || (idx >= j);
    // bool isB = (idx >= i) && (idx < j);

    if (idx < n) {
        C[idx] = isA ? A[idx] : B[idx];
    }
}

int main()
{
    int n = 1000; // 例としての要素数
    int i = 300; // i番目まではAの要素をCにコピー
    // i < idx < jの範囲はBの要素をCにコピー
    int j = 700; // j番目以降はAの要素をCにコピー

    float* h_A = new float[n];
    float* h_B = new float[n];
    float* h_C = new float[n];

    // データの初期化
    for (int i = 0; i < n; ++i) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
        h_C[i] = 0.0f;
    }

    float* d_A;
    float* d_B;
    float* d_C;

    cudaMalloc(&d_A, n * sizeof(float));
    cudaMalloc(&d_B, n * sizeof(float));
    cudaMalloc(&d_C, n * sizeof(float));

    cudaMemcpy(d_A, h_A, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, n * sizeof(float), cudaMemcpyHostToDevice);

    // カーネルの起動
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    mergeArraysKernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, n, i, j);

    cudaMemcpy(h_C, d_C, n * sizeof(float), cudaMemcpyDeviceToHost);

    // 結果の表示やその他の処理
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}
    

