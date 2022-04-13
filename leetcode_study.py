from typing import List,Optional
import pdb
# class Solution:
#     def twoSum(self, numbers: List[int]) -> List[int]:
#         left = 0
#         pdb.set_trace()
#         for second in range(1,len(numbers)):
#             if numbers[second] != numbers[left]:
#                 left += 1
#                 numbers[left] = numbers[second]
#         return numbers[:left+1]

# object = Solution()
# print(object.twoSum([0,1,0]))

#[-1,0,1,2,-1,-4]
# class Solution:
#     def threeSum(self, nums: List[int]) -> List[List[int]]:
#         nums.sort()
#         result = []
#         # pdb.set_trace()
#         for index,i in enumerate(nums):
#             d = {}
#             for second in range(index+1,len(nums)):
#                 two_sum = -(i)
#                 first_num = nums[second]
#                 second_num = two_sum - first_num
#                 if  first_num not in d:
#                     d[second_num] = second
#                 else:
#                     zero_list = [i,first_num,second_num]
#                     zero_list.sort()
#                     if zero_list not in result:
#                         result.append(zero_list)
#         return result

# object = Solution()
# print(object.threeSum([1,2,-2,-1]))


# class Solution:
#     def minSubArrayLen(self, target, nums):
#         total = 0
#         left = 0
#         min_length = len(nums)
#         for index,value in enumerate(nums):
#             total += value
#             while total >= target:
#                 length = index - left+1
#                 if length <= min_length:
#                     min_length = length
#                 total -= nums[left]
#                 left += 1
#         return 0 if sum(nums) < target else min_length


# object = Solution()
# print(object.minSubArrayLen(7,[2,3,1,2,4,3]))

# class Solution:
#     def numSubarrayProductLessThanK(self, nums: List[int], k: int) -> int:
#         total = 1
#         left = 0
#         k_nums = 0
#         list_value = []
#         for index,value in enumerate(nums):
#             total *= value
#             while total >= k and left<=index:
#                 total = total//(nums[left])
#                 left += 1
#             # list_value.append(value)
#             # print(list_value)
#             k_nums += (index-left+1)
#         return k_nums

# class Solution:
#     def numSubarrayProductLessThanK(self, nums: List[int], k: int) -> int:
#         total=1
#         left=0
#         ans=0
#         for right,num in enumerate(nums):
#             total*=num
#             while left<=right and total>=k:
#                 total//=nums[left]
#                 left+=1
#             ans+=(right-left+1)
#         return ans


# object = Solution()
# print(object.numSubarrayProductLessThanK([10,5,2,6],100))

# import re
# class Solution:
#     def countAndSay(self, n: int) -> str:
#         first = '1'
#         for i in range(1,n+1):
#             if i == 1:
#                 result = '1'
#             else:
#                 split_num = re.findall(r'([0-9])(\1*)',first)
#                 first = ''
#                 for j in split_num:
#                     sum_num = str((j[0]+j[1]).count(j[0]))+j[0]
#                     first += sum_num
#             i += 1
#         return first
# object = Solution()
# print(object.countAndSay(5))


# class Solution:
#     def findComplement(self, num: int) -> int:
#         num_bin = bin(num)[2:]
#         other_bin = 2**len(num_bin)-1
#         result = num^other_bin
#         return result
# object = Solution()
# print(object.findComplement(5))

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
# class Solution:
#     def kthSmallest(self, root, k):
#         """
#         :type root: TreeNode
#         :type k: int
#         :rtype: int
#         """
#         def gen(r):
#             if r is not None:
#                 yield from gen(r.left)
#                 yield r.val
#                 yield from gen(r.right)
        
#         it = gen(root)
#         for _ in range(k):
#             ans = next(it)
#         return ans

# object = Solution()
# print(object.kthSmallest([3,1,4,null,2], k = 1))
# class Solution:
#     def numSubarrayProductLessThanK(self, nums: List[int], k: int) -> int:
#         total=1
#         left=0
#         ans=0
#         pdb.set_trace()
#         for right,num in enumerate(nums):
#             total*=num
#             while left<=right and total>=k:
#                 total//=nums[left]
#                 left+=1
#             ans+=(right-left+1)
#         return ans
# object = Solution()
# print(object.numSubarrayProductLessThanK([10,5,2,6],100))
# def yield_test(n):  
#     for i in range(n):  
#         yield call(i)  
#         print("i=",i)  
#     #做一些其它的事情      
#     print("do something.")      
#     print("end.")  

# def call(i):  
#     return i*2
# pdb.set_trace()
# yield_test(5)
# #使用for循环  
# for i in yield_test(5):  
#     print(i,",")
# object = Solution()
# print(object.addOperators("123", 6))
# class Solution:
#     def subarraySum(self, nums, k):
#         ret = pre_sum = 0
#         pdb.set_trace()
#         pre_dict = {0: 1}
#         for i in nums:
#             pre_sum += i
#             ret += pre_dict.get(pre_sum - k, 0)
#             pre_dict[pre_sum] = pre_dict.get(pre_sum, 0) + 1
#         return ret
# object = Solution()
# print(object.subarraySum([1,2,3],3))

# class Solution:
#     def minMoves(self, nums: List[int]) -> int:
#         min_num = min(nums)
#         res = 0
#         for num in nums:
#             res += num - min_num
#             print(res)
#             print(min_num)
#         return res

# object = Solution()
# print(object.minMoves([-1,2,3]))

# class Solution:
#     def findMaxLength(self, nums: List[int]) -> int:
#         left = 0
#         res = 0
#         for index,num in enumerate(nums):
#             right = index+1
#             new = nums[left:right+1]
#             while new.count(0) != new.count(1) and left <= right:
#                 print(nums)
#                 print(left)
#                 nums.pop(nums[left])
#                 left+=1
#             res = len(new)
#         return res


# object = Solution()
# print(object.findMaxLength([0,1,1]))


# class Solution:
#     def pivotIndex(self, nums: List[int]) -> int:
#         for i in range(len(nums)):
#             mid = nums[i]
#             if i == 0:
#                 pre_sum = 0
#             else:
#                 pre_sum = sum(nums[0:i])
#             suf_sum = sum(nums[i+1:len(nums)])
#             if pre_sum == suf_sum:
#                 # pdb.set_trace()
#                 ans = i
#                 break
#             else:
#                 ans = -1
#         return ans
# object = Solution()
# print(object.pivotIndex([-1,-1,-1,-1,0,0]))


# class NumMatrix:

#     def __init__(self, matrix: List[List[int]]):
#         # 二维的前缀和
#         # s[i][j]表示以ij为右下角的矩阵和
#         # s[i][j] = s[i][j-1] + s[i-1][j] - s[i-1][j-1] + matrix[i][j]
#         m, n = len(matrix), len(matrix[0])
#         s = [[0]*(n) for _ in range(m)]
#         s[0][0] = matrix[0][0]
#         for j in range(1, n):
#             s[0][j] = s[0][j-1] + matrix[0][j]
#         for i in range(1, m):
#             s[i][0] = s[i-1][0] + matrix[i][0]
#         for i in range(1, m):
#             for j in range(1, n):
#                 s[i][j] = s[i][j-1] + s[i-1][j] - s[i-1][j-1] + matrix[i][j]
#         pdb.set_trace()
#         self.s = s

#     def sumRegion(self, row1: int, col1: int, row2: int, col2: int) -> int:
#         a = self.s[row2][col2]
#         b = self.s[row1-1][col2] if row1 > 0 else 0
#         c = self.s[row2][col1-1] if col1 > 0 else 0
#         d = self.s[row1-1][col1-1] if row1 > 0 and col1 > 0 else 0
#         return a-b-c+d

# # Your NumMatrix object will be instantiated and called as such:
# obj = NumMatrix([[[3,0,1,4,2],[5,6,3,2,1],[1,2,0,1,5],[4,1,0,1,7],[1,0,3,0,5]]])
# param_1 = obj.sumRegion(row1,col1,row2,col2)

# ["NumMatrix","sumRegion","sumRegion","sumRegion"]
# [[[[3,0,1,4,2],[5,6,3,2,1],[1,2,0,1,5],[4,1,0,1,7],[1,0,3,0,5]]],[2,1,4,3],[1,1,2,2],[1,2,2,4]]

a=[[1,2,3,4],[9,8]]
def ss(lie):
    for i in lie:
        b=i+1
        yield b
print([list(ss(i)) for i in a])