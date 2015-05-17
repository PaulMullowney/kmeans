#include "kmeans.hpp"
#include "gtest/gtest.h"

int my_argc;
char** my_argv;

class KmeansTests : public testing::Test {

 protected: 

  string fileName;
  int m;
  int n;
  int k;
  int nIters;
  int useCublas;
  float rtol;
  
  /* @brief constructor */
  KmeansTests() {
  }

  /* @brief destructor */
  virtual ~KmeansTests() {
  }

  /* @brief this method will be called before each test is run. You
     should define it if you need to initialize the varaibles.
     Otherwise, this can be skipped. */
  virtual void SetUp() {
    if (my_argc>1) {
      fileName = string(my_argv[1]);
      m = atoi(my_argv[2]);
      n = atoi(my_argv[3]);
      k = atoi(my_argv[4]);
      nIters = atoi(my_argv[5]);
      useCublas = atoi(my_argv[6]);
      rtol = atof(my_argv[7]);
    } else {
      fileName = string("mat-sift");
      m = 898790;
      n = 128;
      k = 256;
      nIters = 50;
      useCublas = 1;
      rtol = 1.e-5;
    }
  }

  /* @brief this method will be called after each test is run.
     You should define it if there is cleanup work to do. Otherwise,
     you don't have to provide it. */
  virtual void TearDown() {}
};


TEST_F(KmeansTests, TEST) {

  kmeansErrorStatus err = KMEANS_SUCCESS;

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
  err = computeKmeansF(data,m,n,k,rtol,nIters,1,0,useCublas,centers);
  if (err!=KMEANS_SUCCESS) {
    cout << "Kmeans(CUBLAS) internal error '" << kmeansGetErrorString(err) << "'" <<  endl;
  }

  /* free data */
  if (data) free(data);
  if (centers) free(centers);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  if (argc>1) {
    my_argc = argc;
    my_argv = argv;
  }
  int err = RUN_ALL_TESTS();
  return err;
}

