"""
简单的4个agents测试 - 直接看结果
"""
import numpy as np
from CEA_CG_run2 import cea_allocation

print("CG-CEA测试 - 4个agents")
print("="*70)
print()

# 测试1
print("测试1: a0=0.25, a1=0.25, a2=0.25, a3=0.25, E=20, D=100")
claims = np.array([25.0, 25.0, 25.0, 25.0])
estate = 20.0
result = cea_allocation(claims, estate)
print(f"Claims: {claims}")
print(f"Result: {result}")
print(f"Sum: {result.sum()}")
print()

# 测试2
print("测试2: a0=0.30, a1=0.25, a2=0.25, a3=0.20, E=20, D=100")
claims = np.array([30.0, 25.0, 25.0, 20.0])
estate = 20.0
result = cea_allocation(claims, estate)
print(f"Claims: {claims}")
print(f"Result: {result}")
print(f"Sum: {result.sum()}")
print()

# 测试3
print("测试3: a0=0.40, a1=0.30, a2=0.20, a3=0.10, E=20, D=100")
claims = np.array([40.0, 30.0, 20.0, 10.0])
estate = 20.0
result = cea_allocation(claims, estate)
print(f"Claims: {claims}")
print(f"Result: {result}")
print(f"Sum: {result.sum()}")
print()

# 测试4
print("测试4: a0=0.10, a1=0.30, a2=0.30, a3=0.30, E=20, D=100")
claims = np.array([10.0, 30.0, 30.0, 30.0])
estate = 20.0
result = cea_allocation(claims, estate)
print(f"Claims: {claims}")
print(f"Result: {result}")
print(f"Sum: {result.sum()}")
print()

# 测试5 - 不同的E值
print("测试5: a0=0.25, a1=0.25, a2=0.25, a3=0.25, E=25, D=100")
claims = np.array([25.0, 25.0, 25.0, 25.0])
estate = 25.0
result = cea_allocation(claims, estate)
print(f"Claims: {claims}")
print(f"Result: {result}")
print(f"Sum: {result.sum()}")
print()

# 测试6
print("测试6: a0=0.25, a1=0.25, a2=0.25, a3=0.25, E=30, D=100")
claims = np.array([25.0, 25.0, 25.0, 25.0])
estate = 30.0
result = cea_allocation(claims, estate)
print(f"Claims: {claims}")
print(f"Result: {result}")
print(f"Sum: {result.sum()}")
print()
