from scipy.sparse import load_npz
import matplotlib.pyplot as plt

# 加载稀疏矩阵
tail_history = load_npz('../data/GDELT/history/tail_history_435.npz')
rel_history = load_npz('../data/GDELT/history/rel_history_435.npz')

# 查看基本信息
print("Tail History Shape:", tail_history.shape)
print("Relation History Shape:", rel_history.shape)

# 查看非零元素数量
print("Tail History Non-zero elements:", tail_history.nnz)
print("Relation History Non-zero elements:", rel_history.nnz)

# # 可视化稀疏矩阵
# plt.spy(tail_history, markersize=1)
# plt.title("Tail History Sparse Matrix")
# plt.show()
#
# plt.spy(rel_history, markersize=1)
# plt.title("Relation History Sparse Matrix")
# plt.show()

# 查看非零元素
row_indices, col_indices = tail_history.nonzero()
values = tail_history.data

for i in range(len(row_indices)):  # 打印前10个非零元素
    print(f"Row: {row_indices[i]}, Col: {col_indices[i]}, Value: {values[i]}")
pass