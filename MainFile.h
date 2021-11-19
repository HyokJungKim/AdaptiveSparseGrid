#pragma once
#include <iostream>
#include <vector>
#include <fstream>
#include "CL_ASG_Header_Simple.h"

template <class T> std::vector<std::vector<T>> read_tsv_noflatten(
  const std::string& in_string, int Nrow, int Ncol) {

  std::ifstream DataFile;

  std::vector<std::vector<T>> out_vec(Nrow, std::vector<T>(Ncol));

  DataFile.open(in_string.c_str());

  for (int ii = 0; ii < Nrow; ii++) {
    for (int jj = 0; jj < Ncol; jj++) {
       DataFile >> out_vec[ii][jj];
    }
  }

  DataFile.close();

  return out_vec;
}

template <class T> std::vector<T> read_tsv_flatten(
  const std::string& in_string, int Nrow, int Ncol) {

  std::ifstream DataFile;

  std::vector<T> out_vec(Nrow * Ncol);

  DataFile.open(in_string.c_str());

  for (int ii = 0; ii < Nrow; ii++) {
    for (int jj = 0; jj < Ncol; jj++) {
      DataFile >> out_vec[ii * Ncol + jj];
    }
  }

  DataFile.close();

  return out_vec;
}

vd TestModule(vd& in_params, const v2i& in_maps, std::function<double(vd&)> inObj, const int in_NS) {
    int testNum = 200;

    vd test_intp(testNum);
    v2d test_state(testNum, vd(in_NS, 0.5));

    std::uniform_real_distribution<double> test_rnd2(0.0, 1.0);

    for (int ii = 0; ii < testNum; ii++) {
        for (int jj = 0; jj < in_NS; jj++) {
          test_state[ii][jj] = test_rnd2(gen_random);
        }
        test_intp[ii] = EvaluateNoGPU(test_state[ii], in_params, in_maps, in_NS);
    }

    std::chrono::high_resolution_clock::time_point t1, t2;
    t1 = std::chrono::high_resolution_clock::now();

    vd test_out(testNum);
    for (int ii = 0; ii < testNum; ii++) {
    test_out[ii] = inObj(test_state[ii]);
    }

    t2 = std::chrono::high_resolution_clock::now();
    auto duration1 = std::chrono::duration_cast<std::chrono::milliseconds> (t2 - t1).count();

    vd err_sizes(3, 0.0);
    double tempval;

    for (int ii = 0; ii < testNum; ii++) {
        // Size of the Error
        err_sizes[0] += fabs((test_intp[ii] - test_out[ii]) / test_out[ii]);

        tempval = fabs(test_intp[ii] - test_out[ii]);

        err_sizes[1] += fabs(tempval);
        err_sizes[2] += std::max(std::log(tempval), -15.0);
    }

    err_sizes[0] /= static_cast<double>(testNum);
    err_sizes[1] /= static_cast<double>(testNum);
    err_sizes[2] /= static_cast<double>(testNum);

    std::cout << "Average Relative Err. Size: " << err_sizes[0] << "\n";
    std::cout << "Average Absolute Err. Size: " << err_sizes[1] << "\n";
    std::cout << "Average Log Err. Size: " << err_sizes[2] << "\n";

    return err_sizes;
}