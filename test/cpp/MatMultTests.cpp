#include "kmeans.hpp"
#include "gtest/gtest.h"

class MatMultTests : public testing::Test {

 protected: 

  /* @brief constructor */
  MatMultTests() {
  }

  /* @brief destructor */
  virtual ~MatMultTests() {
  }

  /* @brief this method will be called before each test is run. You
     should define it if you need to initialize the varaibles.
     Otherwise, this can be skipped. */
  virtual void SetUp() {}

  /* @brief this method will be called after each test is run.
     You should define it if there is cleanup work to do. Otherwise,
     you don't have to provide it. */
  virtual void TearDown() {}
};


TEST_F(MatMultTests, SIFT) {

  string fileName("mat-sift");
  int m = 898790;
  int n = 128;
  int k = 256;
  int err = 0;

  /* allocate data */
  float * data = (float *)malloc(m*n*sizeof(float));
  float * centers = (float *)malloc(k*n*sizeof(float));
  float * result = (float *)malloc(m*k*sizeof(float));
  float * resultCublas = (float *)malloc(m*k*sizeof(float));

  /* read matrix from file */
  FILE * fid = fopen(fileName.c_str(), "rb");
  int nread = fread(data, sizeof(float), m*n, fid);
  ASSERT_EQ(nread, m*n);
  fclose(fid);

  /* initialize centers to 1 */
  for (int i = 0; i<k*n; ++i) centers[i] = (float)1;

  /* allocate device space for the various arrays */
  float * dev_data, *dev_centers, *dev_result;
  int factor = TILESIZE*N_UNROLL_FLOAT;
  int m_padded = ((m + factor - 1) / factor)*factor;

  int nBytes = m_padded*n*sizeof(float);
  cudaMalloc((void**)&dev_data, nBytes);
  cudaMemset(dev_data, 0, nBytes);
  cudaMemcpy(dev_data, data, m*n*sizeof(float), cudaMemcpyHostToDevice);

  nBytes = n*k*sizeof(float);
  cudaMalloc((void**)&dev_centers, nBytes);
  cudaMemcpy(dev_centers, centers, nBytes, cudaMemcpyHostToDevice);

  nBytes = m*k*sizeof(float);
  cudaMalloc((void**)&dev_result, nBytes);
  cudaMemset(dev_result, 0, nBytes);

  /* run MatMatMultF */
  err = MatMatMultF(m, n, dev_data, k, dev_centers, dev_result);
  cudaMemcpy(result, dev_result, nBytes, cudaMemcpyDeviceToHost);

  /* run CUBLAS SGEMM */
  float one = 1.f;
  float zero = 0.f;
  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
	  k, m, n, (const float *)&one,
	  (const float *)dev_centers, k,
	  (const float *)dev_data, n,
	  (const float *)&zero, (float *)dev_result, k);
  cudaMemcpy(resultCublas, dev_result, nBytes, cudaMemcpyDeviceToHost);

  /* check results */
  for (int i = 0; i < m; ++i) {
	  for (int j = 0; j < k; ++j) {
		  int index = i*k + j;
		  if (result[index] == 0 && resultCublas[index] == 0) continue;
		  else {
			  float err = fabs(result[index] - resultCublas[index]) / fabs(result[index]);
			  if (err >= 1.e-6 || result[index] == 0)
				  printf("i=%d, j=%d : %1.5g, %1.5g, err=%1.5g\n", i, j, result[index], resultCublas[index], err);
			  ASSERT_LT(err, 1.e-6);
		  }
	  }
  }

  /* free data */
  if (dev_data) cudaFree(dev_data);
  if (dev_centers) cudaFree(dev_centers);
  if (dev_result) cudaFree(dev_result);

  if (data) free(data);
  if (centers) free(centers);
  if (result) free(result);
  if (resultCublas) free(resultCublas);
  cublasDestroy(handle);
}

TEST_F(MatMultTests, HOG) {
  
  string fileName("mat-hog");
  int m = 796160;
  int n = 324;
  int k = 256;
  int err = 0;

  /* allocate data */
  float * data = (float *) malloc(m*n*sizeof(float));
  float * centers = (float *) malloc(k*n*sizeof(float));
  float * result = (float *)malloc(m*k*sizeof(float));
  float * resultCublas = (float *)malloc(m*k*sizeof(float));

  /* read matrix from file */
  FILE * fid = fopen(fileName.c_str(),"rb");
  int nread = fread(data, sizeof(float), m*n, fid);
  ASSERT_EQ(nread, m*n);
  fclose(fid);

  /* initialize centers to 1 */
  for (int i=0; i<k*n; ++i) centers[i] = (float)1;

  /* allocate device space for the various arrays */
  float * dev_data, * dev_centers, * dev_result;
  int factor = TILESIZE*N_UNROLL_FLOAT;
  int m_padded = ((m + factor - 1)/factor)*factor;

  int nBytes = m_padded*n*sizeof(float);
  cudaMalloc((void**)&dev_data,nBytes);
  cudaMemset(dev_data,0,nBytes);
  cudaMemcpy(dev_data, data, m*n*sizeof(float), cudaMemcpyHostToDevice);

  nBytes = n*k*sizeof(float);
  cudaMalloc((void**)&dev_centers,nBytes);
  cudaMemcpy(dev_centers, centers, nBytes, cudaMemcpyHostToDevice);

  nBytes = m*k*sizeof(float);
  cudaMalloc((void**)&dev_result,nBytes);
  cudaMemset(dev_result,0,nBytes);

  /* run MatMatMultF */
  err = MatMatMultF(m,n,dev_data,k,dev_centers,dev_result);
  cudaMemcpy(result, dev_result, nBytes, cudaMemcpyDeviceToHost);
	  
  /* run CUBLAS SGEMM */
  float one = 1.f;
  float zero = 0.f;
  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
	  k, m, n, (const float *)&one,
	  (const float *)dev_centers, k,
	  (const float *)dev_data, n,
	  (const float *)&zero, (float *)dev_result, k);
  cudaMemcpy(resultCublas, dev_result, nBytes, cudaMemcpyDeviceToHost);

  /* check results */
  for (int i = 0; i < m; ++i) {
	  for (int j = 0; j < k; ++j) {
		  int index = i*k + j;
		  if (result[index] == 0 && resultCublas[index] == 0) continue;
		  else {
			  float err = fabs(result[index] - resultCublas[index]) / fabs(result[index]);
			  if (err >= 1.e-6 || result[index] == 0)
				  printf("i=%d, j=%d : %1.5g, %1.5g, err=%1.5g\n", i, j, result[index], resultCublas[index], err);
			  ASSERT_LT(err, 1.e-6);
		  }
	  }
  }

  /* free data */
  if (dev_data) cudaFree(dev_data);
  if (dev_centers) cudaFree(dev_centers);
  if (dev_result) cudaFree(dev_result);

  if (data) free(data);
  if (centers) free(centers);
  if (result) free(result);
  if (resultCublas) free(resultCublas);
  cublasDestroy(handle);
}


int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  int err = RUN_ALL_TESTS();
  int keepOpen = getchar();
  return err;
}
