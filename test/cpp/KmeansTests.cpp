#include "kmeans.hpp"
#include "gtest/gtest.h"

class KmeansTests : public testing::Test {

 protected: 

  /* @brief constructor */
  KmeansTests() {
  }

  /* @brief destructor */
  virtual ~KmeansTests() {
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


TEST_F(KmeansTests, SIFT_CUBLAS) {

  string fileName("mat-sift");
  int m = 898790;
  int n = 128;
  int k = 1024;
  int nIters = 50;
  kmeansErrorStatus err = KMEANS_SUCCESS;
  int useCublas = 1;

  /* allocate data */
  float * data = (float *) malloc(m*n*sizeof(float));
  float * centers = (float *) malloc(k*n*sizeof(float));

  /* read matrix from file */
  FILE * fid = fopen(fileName.c_str(),"rb");
  int nread = fread(data, sizeof(float), m*n, fid);
  ASSERT_EQ(nread, m*n);
  fclose(fid);

  /* initialize centers to 0 */
  memset(centers, 0, sizeof(float)*k*n);

  /* run kmeans */
  err = computeKmeansF(data,m,n,k,1.e-5,nIters,1,0,useCublas,centers);
  if (err!=KMEANS_SUCCESS) {
    cout << "Kmeans(CUBLAS) internal error '" << kmeansGetErrorString(err) << "'" <<  endl;
  }

  /* free data */
  if (data) free(data);
  if (centers) free(centers);
}


TEST_F(KmeansTests, SIFT) {

  string fileName("mat-sift");
  int m = 898790;
  int n = 128;
  int k = 1024;
  int nIters = 50;
  kmeansErrorStatus err = KMEANS_SUCCESS;
  int useCublas = 0;

  /* allocate data */
  float * data = (float *) malloc(m*n*sizeof(float));
  float * centers = (float *) malloc(k*n*sizeof(float));

  /* read matrix from file */
  FILE * fid = fopen(fileName.c_str(),"rb");
  int nread = fread(data, sizeof(float), m*n, fid);
  ASSERT_EQ(nread, m*n);
  fclose(fid);

  /* initialize centers to 0 */
  memset(centers, 0, sizeof(float)*k*n);

  /* run kmeans */
  err = computeKmeansF(data,m,n,k,1.e-5,nIters,1,0,useCublas,centers);
  if (err!=KMEANS_SUCCESS) {
    cout << "Kmeans internal error '" << kmeansGetErrorString(err) << "'" <<  endl;
  }

  /* free data */
  if (data) free(data);
  if (centers) free(centers);
}

TEST_F(KmeansTests, HOG_CUBLAS) {
  
  string fileName("mat-hog");
  int m = 796160;
  int n = 324;
  int k = 1024;
  int nIters = 50;
  kmeansErrorStatus err = KMEANS_SUCCESS;
  int useCublas = 1;

  /* allocate data */
  float * data = (float *) malloc(m*n*sizeof(float));
  float * centers = (float *) malloc(k*n*sizeof(float));

  /* read matrix from file */
  FILE * fid = fopen(fileName.c_str(),"rb");
  int nread = fread(data, sizeof(float), m*n, fid);
  ASSERT_EQ(nread, m*n);
  fclose(fid);

  /* initialize centers to 0 */
  memset(centers, 0, sizeof(float)*k*n);

  /* run kmeans */
  err = computeKmeansF(data,m,n,k,1.e-5,nIters,1,0,useCublas,centers);
  if (err!=KMEANS_SUCCESS) {
    cout << "Kmeans(CUBLAS) internal error '" << kmeansGetErrorString(err) << "'" <<  endl;
  }

  /* free data */
  if (data) free(data);
  if (centers) free(centers);
}


TEST_F(KmeansTests, HOG) {
  
  string fileName("mat-hog");
  int m = 796160;
  int n = 324;
  int k = 1024;
  int nIters = 50;
  kmeansErrorStatus err = KMEANS_SUCCESS;
  int useCublas = 0;

  /* allocate data */
  float * data = (float *) malloc(m*n*sizeof(float));
  float * centers = (float *) malloc(k*n*sizeof(float));

  /* read matrix from file */
  FILE * fid = fopen(fileName.c_str(),"rb");
  int nread = fread(data, sizeof(float), m*n, fid);
  ASSERT_EQ(nread, m*n);
  fclose(fid);

  /* initialize centers to 0 */
  memset(centers, 0, sizeof(float)*k*n);

  /* run kmeans */
  err = computeKmeansF(data,m,n,k,1.e-5,nIters,1,0,useCublas,centers);
  if (err!=KMEANS_SUCCESS) {
    cout << "Kmeans internal error '" << kmeansGetErrorString(err) << "'" <<  endl;
  }

  /* free data */
  if (data) free(data);
  if (centers) free(centers);
}


int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  int err = RUN_ALL_TESTS();
  int keepOpen = getchar();
  return err;
}
