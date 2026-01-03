#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <string>
#include <tuple>

#include "shvetsova_k_rad_sort_batch_merge/common/include/common.hpp"
#include "shvetsova_k_rad_sort_batch_merge/mpi/include/ops_mpi.hpp"
#include "shvetsova_k_rad_sort_batch_merge/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace shvetsova_k_rad_sort_batch_merge {

class ShvetsovaKRadSortBatchMergeRunFuncTestsProcesses : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return test_param;
  }

 protected:
  void SetUp() override {
    TestType param = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());

    std::string abs_path = ppc::util::GetAbsoluteTaskPath(PPC_ID_shvetsova_k_rad_sort_batch_merge, param + ".txt");

    std::ifstream file(abs_path);
    if (!file.is_open()) {
      std::cerr << "ERROR: Cannot open file: " << abs_path << '\n';
      return;
    }

    size_t n = 0;
    file >> n;

    input_data_.resize(n);
    expect_res_.resize(n);

    for (size_t i = 0; i < n; ++i) {
      file >> input_data_[i];
    }

    for (size_t i = 0; i < n; ++i) {
      file >> expect_res_[i];
    }

    file.close();
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.size() != expect_res_.size()) {
      std::cerr << "Size mismatch: expected " << expect_res_.size() << ", got " << output_data.size() << '/n';
      return true;
    }

    // Для целых чисел проверяем точное равенство
    for (size_t i = 0; i < output_data.size(); ++i) {
      if (output_data[i] != expect_res_[i]) {
        std::cerr << "Mismatch at index " << i << ": expected " << expect_res_[i] << ", got " << output_data[i] << '/n';
        return true;
      }
    }
    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expect_res_;
};

namespace {

TEST_P(ShvetsovaKRadSortBatchMergeRunFuncTestsProcesses, DataFromTest) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 5> kTestParam = {"test1", "test2", "test3", "test4", "test5"};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<ShvetsovaKRadSortBatchMergeMPI, InType>(
                                               kTestParam, PPC_SETTINGS_shvetsova_k_rad_sort_batch_merge),
                                           ppc::util::AddFuncTask<ShvetsovaKRadSortBatchMergeSEQ, InType>(
                                               kTestParam, PPC_SETTINGS_shvetsova_k_rad_sort_batch_merge));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = ShvetsovaKRadSortBatchMergeRunFuncTestsProcesses::PrintFuncTestName<
    ShvetsovaKRadSortBatchMergeRunFuncTestsProcesses>;

INSTANTIATE_TEST_SUITE_P(RadSortBatchMergeTest, ShvetsovaKRadSortBatchMergeRunFuncTestsProcesses, kGtestValues,
                         kPerfTestName);

}  // namespace

}  // namespace shvetsova_k_rad_sort_batch_merge
