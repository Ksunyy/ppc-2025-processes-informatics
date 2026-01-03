#pragma once

#include <vector>

#include "shvetsova_k_rad_sort_batch_merge/common/include/common.hpp"
#include "task/include/task.hpp"

namespace shvetsova_k_rad_sort_batch_merge {

class ShvetsovaKRadSortBatchMergeSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }

  explicit ShvetsovaKRadSortBatchMergeSEQ(const InType &in);

 private:
  std::vector<int> data_;

  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  // доп функции //
  static void RadixSort(std::vector<int> &vec);
  static void BatcherOddEvenMergeSort(std::vector<int> &vec, int left, int right);
  static void ExecuteBatcherStep(std::vector<int> &vec, int left, int n, int p, int k);
  static void CompareAndSwap(std::vector<int> &vec, int i, int j);
};

}  // namespace shvetsova_k_rad_sort_batch_merge
