## homework1 
把field_name改成了**主讲或助教**，在get_filtered_records函数里先按“或”对field_name进行切割，得到列表变量fields_list.  
定义了breakoutout变量，作为循环结束的判断条件  
使用**三重循环**，最外层对每一条record遍历，中间层对fields_list中要查找的内容遍历，最内层对通过current_field in fields_list得到的人员列表中的人员en_name进行检查，若==field_value，就记breakoutout为True，退回到最外层循环，对下一条record遍历

## homework2
换了一个参数量比较小的模型，输出效果不太理想，凑合看吧