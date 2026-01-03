#pragma once

#include <utility>
#include <vector>

#include "shvetsova_k_rad_sort_batch_merge/common/include/common.hpp"
#include "task/include/task.hpp"

namespace shvetsova_k_rad_sort_batch_merge {

class ShvetsovaKRadSortBatchMergeMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }

  explicit ShvetsovaKRadSortBatchMergeMPI(const InType &in);

 private:
  std::vector<int> data_;

  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  // Распределение данных
  static void CreateDistribution(int proc_count, int size, std::vector<int> &counts, std::vector<int> &displs);

  // MPI операции
  static void ScatterData(const std::vector<int> &data, std::vector<int> &local, const std::vector<int> &counts,
                          const std::vector<int> &displs, int rank);

  static void GatherAndBroadcast(std::vector<int> &data, const std::vector<int> &local, const std::vector<int> &counts,
                                 const std::vector<int> &displs, int rank);

  // Поразрядная сортировка
  static void RadixSortWithNegatives(std::vector<int> &arr);
  static void RadixSort(std::vector<int> &arr);
  static void CountingSort(std::vector<int> &arr, int digit_place);
  static int GetDigit(int num, int digit_place);

  // Вспомогательные функции для поразрядной сортировки
  static void CountDigits(const std::vector<int> &arr, int digit_place, std::vector<int> &count);
  static void AccumulateCounts(std::vector<int> &count);
  static void BuildOutputArray(const std::vector<int> &arr, int digit_place, std::vector<int> &count,
                               std::vector<int> &output);
  static void CopyBackToArray(std::vector<int> &arr, const std::vector<int> &output);

  // Функции для обработки отрицательных чисел
  static std::vector<int> ExtractNegatives(const std::vector<int> &arr);
  static std::vector<int> ExtractNonNegatives(const std::vector<int> &arr);
  static void SortSeparatedArrays(std::vector<int> &negatives, std::vector<int> &non_negatives);
  static void CombineSortedArrays(std::vector<int> &arr, std::vector<int> &negatives, std::vector<int> &non_negatives);
  static void ReverseAndRestoreSign(std::vector<int> &negatives);

  // Сеть Бэтчера
  static std::vector<std::pair<int, int>> GenerateBatcherComparators(int num_processes);
  static void AddComparator(std::vector<std::pair<int, int>> &comparators, int a, int b);
  static std::vector<int> MergeTwoSorted(const std::vector<int> &a, const std::vector<int> &b);

  // Вспомогательные функции для сети Бэтчера
  static void ProcessPowerStep(int power, int num_processes, std::vector<std::pair<int, int>> &comparators);
  static void ProcessKStep(int power, int k, int num_processes, std::vector<std::pair<int, int>> &comparators);
  static void ProcessJStep(int power, int k, int j, int num_processes, std::vector<std::pair<int, int>> &comparators);

  // Вспомогательные функции для обработки компараторов
  static void ProcessBatcherComparators(const std::vector<std::pair<int, int>> &comparators, int rank,
                                        const std::vector<int> &counts, std::vector<int> &local);
  static void ProcessComparator(const std::pair<int, int> &comparator, int rank, const std::vector<int> &counts,
                                std::vector<int> &local);
  static void MergeAndDistribute(std::vector<int> &local, const std::vector<int> &received, int rank, int a, int b);

  // Вспомогательная функция для слияния
  static void AddRemainingElements(std::vector<int> &result, const std::vector<int> &source, size_t &index);
};

}  // namespace shvetsova_k_rad_sort_batch_merge
