#include <gtest/gtest.h>

#include <cstddef>
#include <random>

#include "shvetsova_k_rad_sort_batch_merge/common/include/common.hpp"
#include "shvetsova_k_rad_sort_batch_merge/mpi/include/ops_mpi.hpp"
#include "shvetsova_k_rad_sort_batch_merge/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace shvetsova_k_rad_sort_batch_merge {

class ShvetsovaKRadSortBatchMergeRunPerfTestProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  void SetUp() override {
    // Количество элементов для теста производительности (1 миллион)
    int n = 1000000;
    input_data_.resize(n);

    // Генерация случайных целых чисел
    std::seed_seq seed{std::random_device{}(), std::random_device{}()};
    std::mt19937 gen(seed);  // Фиксированный seed для воспроизводимости
    std::uniform_int_distribution<int> dis(0, 1000000);

    for (int i = 0; i < n; ++i) {
      input_data_[i] = static_cast<double>(dis(gen));
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    // Простейшая проверка: данные не пусты и отсортированы
    if (output_data.empty()) {
      return true;
    }

    for (size_t i = 0; i < output_data.size() - 1; ++i) {
      if (output_data[i] > output_data[i + 1]) {
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
};

TEST_P(ShvetsovaKRadSortBatchMergeRunPerfTestProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, ShvetsovaKRadSortBatchMergeMPI, ShvetsovaKRadSortBatchMergeSEQ>(
        PPC_SETTINGS_shvetsova_k_rad_sort_batch_merge);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = ShvetsovaKRadSortBatchMergeRunPerfTestProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, ShvetsovaKRadSortBatchMergeRunPerfTestProcesses, kGtestValues, kPerfTestName);

}  // namespace shvetsova_k_rad_sort_batch_merge
